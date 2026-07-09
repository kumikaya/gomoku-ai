//! 训练器
//!
//! `TrainConfig` 配置训练超参数，`Trainer` 执行完整的 AlphaZero 训练循环。

use crate::game::board::D4Symmetry;
use crate::inference::InferenceServer;
use crate::network::transformer::GomokuNetwork;
use crate::network::transformer::policy_out_dim;
use crate::selfplay::{PlayRecord, SelfPlayConfig, self_play};
use crate::training::buffer::RolloutBuffer;
use crate::training::lr_schedule::LrSchedule;
use crate::training::metrics::{BatchStats, EpochStats, TrainingLogger};

use crate::eval::{BaselineServer, EloTracker, EvalConfig, MatchRunner};

use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::ModuleOptimizer;
use burn::store::ModuleRecord;
use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams},
    tensor::{Device, FloatDType, Int, Tensor, activation::log_softmax},
};
use futures::task::SpawnExt;
use futures_executor::ThreadPool;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::path::PathBuf;
use std::sync::Arc;

// ── 配置 ──

pub struct TrainConfig {
    pub num_simulations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
    pub mini_batches_per_iteration: usize,
    pub num_iterations: usize,
    pub learning_rate: f64,
    pub value_loss_weight: f32,
    pub save_every: usize,
    pub model_dir: PathBuf,
    pub buffer_capacity: usize,
    pub max_grad_norm: f32,
    /// 从指定 checkpoint 路径恢复训练（None 则创建新模型）
    pub checkpoint: Option<PathBuf>,

    /// 最终学习率比例（相对于 `learning_rate`），cosine 衰减的底值。
    pub lr_final_ratio: f32,
    // ── 评估配置 ──
    /// 是否启用对抗评估（每 `eval_every` 轮与 baseline 对战）
    pub eval_enabled: bool,
    /// 评估对弈局数
    pub eval_num_games: usize,
    /// 评估时 MCTS 模拟次数（建议比训练时高）
    pub eval_num_simulations: usize,
    /// 晋升阈值：当前模型胜率超过此值替换 baseline
    pub eval_promotion_threshold: f64,
    /// 每隔多少轮评估一次
    pub eval_every: usize,
    /// 随机种子（固定可复现实验结果）
    pub random_seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_simulations: 64,
            games_per_iteration: 64,
            batch_size: 256,
            mini_batches_per_iteration: 256,
            num_iterations: 100,
            learning_rate: 1e-3,
            value_loss_weight: 1.0,
            save_every: 5,
            model_dir: PathBuf::from("checkpoints"),
            buffer_capacity: 160000,
            max_grad_norm: 1.0,
            checkpoint: None,
            lr_final_ratio: 0.1,
            eval_enabled: true,
            eval_num_games: 100,
            eval_num_simulations: 64,
            eval_promotion_threshold: 0.55,
            eval_every: 5,
            random_seed: 42,
        }
    }
}

// ── 训练器 ──

pub struct Trainer {
    config: TrainConfig,
    device: Device,
    buffer: RolloutBuffer<PlayRecord>,
    metrics_logger: TrainingLogger,
    pool: ThreadPool,
}

impl Trainer {
    pub fn new(config: TrainConfig, device: Device) -> Self {
        let cap = config.buffer_capacity;
        let metrics_logger = TrainingLogger::new(&config.model_dir);
        let pool = ThreadPool::new().expect("Failed to create thread pool");
        Self {
            config,
            device,
            buffer: RolloutBuffer::new(cap),
            metrics_logger,
            pool,
        }
    }

    // ── 模型持久化 ──

    fn model_path(&self, label: &str) -> PathBuf {
        self.config.model_dir.join(format!("gomoku_{}", label))
    }

    fn load_model(&self, autodiff_device: &Device) -> GomokuNetwork {
        if let Some(ref ckpt_path) = self.config.checkpoint {
            match ModuleRecord::load(ckpt_path) {
                Ok(record) => {
                    println!("Loaded checkpoint from {:?}", ckpt_path);
                    return GomokuNetwork::new(autodiff_device).load_record(record);
                }
                Err(e) => {
                    eprintln!(
                        "Warning: failed to load checkpoint {:?}: {}. Creating new model.",
                        ckpt_path, e
                    );
                }
            }
        }
        println!("Creating new model.");
        GomokuNetwork::new(autodiff_device)
    }

    fn save_model(&self, model: &GomokuNetwork, label: &str) {
        let path = self.model_path(label);
        if let Err(e) = model.clone().valid().save_file(&path) {
            eprintln!("Warning: failed to save model {}: {}", label, e);
        } else {
            println!("Model saved: {:?}", path);
        }
    }

    // ── 训练循环 ──

    pub fn train(&mut self) {
        std::fs::create_dir_all(&self.config.model_dir).ok();

        let train_device = self.device.clone().autodiff();
        let mut model = self.load_model(&train_device);

        let mut optim = AdamConfig::new().init();
        if self.config.max_grad_norm > 0.0 {
            optim = optim
                .with_grad_clipping(GradientClippingConfig::Norm(self.config.max_grad_norm).init());
        }

        let inference_server = InferenceServer::new(model.clone().valid(), self.device.clone());

        // 评估基础设施
        let eval_config = EvalConfig {
            num_games: self.config.eval_num_games,
            num_simulations: self.config.eval_num_simulations,
            eval_every: self.config.eval_every,
        };
        let match_runner = if self.config.eval_enabled {
            Some(MatchRunner::new(eval_config))
        } else {
            println!("  Eval disabled.");
            None
        };

        // BaselineServer 长期持有 baseline 的 GPU 线程，只在晋升时热更新
        let baseline_server = if self.config.eval_enabled {
            Some(BaselineServer::new(
                &model,
                self.config.model_dir.clone(),
                &self.device,
            ))
        } else {
            None
        };

        let mut elo = EloTracker::new();

        let mut master_rng = StdRng::seed_from_u64(self.config.random_seed);
        println!("  Random seed: {}", self.config.random_seed);

        let lr_sched = LrSchedule::new(self.config.num_iterations, self.config.lr_final_ratio);
        println!(
            "  LR schedule: final_ratio={:.2}, base_lr={:.6}",
            self.config.lr_final_ratio, self.config.learning_rate,
        );

        for iteration in 0..self.config.num_iterations {
            let epoch = iteration + 1;

            println!(
                "========== Iteration {}/{} ==========",
                epoch, self.config.num_iterations
            );

            let lr = lr_sched.lr(iteration, self.config.learning_rate);

            self.run_self_play(&inference_server, iteration, &mut master_rng);

            let buffer_size = self.buffer.len();
            println!(
                "  Training: mini_batches={}, buffer size={}",
                self.config.mini_batches_per_iteration, buffer_size,
            );

            if buffer_size < self.config.batch_size {
                println!("    Buffer too small, skipping.");
                continue;
            }

            self.run_training_batches(
                &mut model,
                &mut optim,
                &train_device,
                lr,
                &mut master_rng,
                epoch,
            );

            inference_server.update_model(model.clone().valid());

            // ── 对抗评估 ──
            if self.config.eval_enabled && epoch % self.config.eval_every == 0 {
                if let (Some(runner), Some(bs)) = (&match_runner, &baseline_server) {
                    println!();
                    println!("  === Tournament Evaluation (iter {}) ===", epoch);
                    let result = runner.run_match(
                        &self.pool,
                        &inference_server,
                        &bs.server(),
                        master_rng.random::<u64>(),
                    );
                    result.print();

                    let wr = result.win_rate_current();
                    let new_elo = elo.update(epoch, wr);
                    println!("  Elo: {:.1} (baseline=1500)", new_elo);

                    if wr > self.config.eval_promotion_threshold {
                        println!(
                            "  >>> Model promoted! Win rate {:.1}% > threshold {:.0}%",
                            wr * 100.0,
                            self.config.eval_promotion_threshold * 100.0,
                        );
                        bs.promote(&model);
                    } else {
                        println!(
                            "  Win rate {:.1}% <= threshold {:.0}%, baseline unchanged",
                            wr * 100.0,
                            self.config.eval_promotion_threshold * 100.0,
                        );
                    }
                }
            }

            // 每个 epoch 结束后保存
            if epoch % self.config.save_every == 0 || epoch == self.config.num_iterations {
                self.save_model(&model, &format!("epoch_{}", epoch));
            }
        }

        if self.config.eval_enabled {
            elo.print_history();
        }

        println!("Training complete!");
    }

    // ── 自对弈 ──

    /// minizero 风格训练进度衰减：0%–50%→1.0, 50%–75%→0.5, 75%–100%→0.25
    fn decayed_temperature(iteration: usize, num_iterations: usize) -> f32 {
        let ratio = iteration as f32 / num_iterations.max(1) as f32;
        if ratio < 0.5 {
            1.0
        } else if ratio < 0.75 {
            0.5
        } else {
            0.25
        }
    }

    fn run_self_play<R: RngExt>(
        &mut self,
        inference_server: &InferenceServer,
        iteration: usize,
        master_rng: &mut R,
    ) {
        let total = self.config.games_per_iteration;

        let temp = Self::decayed_temperature(iteration, self.config.num_iterations);
        if iteration == 0 {
            println!("  Temperature: {:.2}", temp);
        }

        let pb = Arc::new(ProgressBar::new(total as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Self-play: {bar:40.cyan/blue} {pos}/{len} ({eta})")
                .unwrap(),
        );

        let sp_config = SelfPlayConfig {
            num_simulations: self.config.num_simulations,
            select_temperature: temp,
            ..Default::default()
        };

        let base_seed = master_rng.random::<u64>();

        let mut handles = Vec::with_capacity(total);
        for game_i in 0..total {
            let seed = base_seed.wrapping_add(game_i as u64);
            let pb = Arc::clone(&pb);
            let sp_config = sp_config.clone();
            let inf = inference_server.clone();

            // 关键：spawn 到线程池，handle 可以并发等待
            let handle = self
                .pool
                .spawn_with_handle(async move {
                    let mut game_rng = StdRng::seed_from_u64(seed);
                    let records = self_play(&inf, &sp_config, &mut game_rng).await.records;
                    pb.inc(1);
                    records
                })
                .expect("spawn selfplay task");

            handles.push(handle);
        }
        let results: Vec<Vec<PlayRecord>> =
            futures_executor::block_on(futures::future::join_all(handles));
        let all_records: Vec<PlayRecord> = results.into_iter().flatten().collect();

        pb.finish_and_clear();

        let added = self.buffer.extend(all_records);
        println!("  Buffer: {} samples (+{})", self.buffer.len(), added);
    }

    // ── 训练阶段 ──

    fn run_training_batches<R: RngExt>(
        &mut self,
        model: &mut GomokuNetwork,
        optim: &mut ModuleOptimizer,
        train_device: &Device,
        lr: f64,
        rng: &mut R,
        epoch: usize,
    ) -> EpochStats {
        let mut stats = EpochStats::default();
        let identity_prob = 1.0 / D4Symmetry::COUNT as f32;
        let symmetry = D4Symmetry::new(model.board_size);
        let npos = policy_out_dim(model.board_size);
        let encode_len = model.board_size * model.board_size;

        let batch_size = self.config.batch_size;
        let mini_batches_per_iteration = self.config.mini_batches_per_iteration;
        let num_batches = mini_batches_per_iteration.min(self.buffer.len() / batch_size);
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Training: {bar:40.green/dim} {pos}/{len} batches ({eta})")
                .unwrap(),
        );
        // 加权有放回采样，每 batch 独立采样一次
        let weights: Vec<_> = self.buffer.iter().map(|i| i.sample_weight).collect();

        for _ in 0..num_batches {
            let chunk: Vec<_> = self
                .buffer
                .sample_batch(batch_size, &weights, rng)
                .collect();
            let mini_batch: Vec<PlayRecord> = chunk
                .into_iter()
                .map(|record| {
                    let (state, policy) =
                        symmetry.random_augment(&record.state, &record.policy, rng, identity_prob);
                    PlayRecord {
                        state,
                        policy,
                        value: record.value,
                        ..Default::default()
                    }
                })
                .collect();
            let (flat_states, flat_policies, flat_values) = Self::flatten_batch(&mini_batch, npos);

            let state_tensor = Tensor::<2, Int>::from_data(
                burn::tensor::TensorData::new(flat_states, [batch_size as i32, encode_len as i32]),
                train_device,
            );
            let policy_target = Tensor::<1>::from_floats(flat_policies.as_slice(), train_device)
                .reshape([batch_size, npos]);
            let value_target_tensor =
                Tensor::<1>::from_floats(flat_values.as_slice(), train_device)
                    .reshape([batch_size, 1]);
            let (policy_logits, value_pred) = model.forward(state_tensor);

            let log_probs = log_softmax(policy_logits.clone(), 1);

            // ── 统计（detached） ──
            let entropy = {
                let log_probs = log_probs.clone().detach();
                let probs = log_probs.clone().exp();
                -(probs * log_probs).sum_dim(1).mean().into_scalar::<f32>()
            };

            let val_pred: Vec<f32> = value_pred
                .clone()
                .reshape([batch_size])
                .cast(FloatDType::F32)
                .into_data()
                .to_vec()
                .unwrap();
            stats.all_value_preds.extend(val_pred);
            stats.all_value_targets.extend(flat_values.iter());

            // ── 损失 ──
            let policy_loss = -(log_probs * policy_target.clone()).sum_dim(1).mean();
            let policy_loss_scalar: f32 = policy_loss.clone().into_scalar();

            let mse = MseLoss::new();
            let value_loss = mse.forward(
                value_pred.clone(),
                value_target_tensor.clone(),
                Reduction::Mean,
            );
            let value_loss_scalar: f32 = value_loss.clone().into_scalar();

            let loss = value_loss * self.config.value_loss_weight + policy_loss;
            let total_loss_scalar: f32 = loss.clone().into_scalar();

            // ── 记录每个 batch 的指标到日志文件 ──
            self.metrics_logger.log_batch(
                epoch,
                total_loss_scalar,
                policy_loss_scalar,
                value_loss_scalar,
                entropy,
            );

            stats.push(BatchStats {
                total_loss: total_loss_scalar,
                policy_loss: policy_loss_scalar,
                value_loss: value_loss_scalar,
                entropy,
                batch_size,
            });

            // 反向传播 + 参数更新
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            *model = optim.step(lr.into(), model.clone(), grads);

            pb.inc(1);
        }

        pb.finish_and_clear();

        let explained_var =
            Self::explained_variance(&stats.all_value_preds, &stats.all_value_targets);

        // 记录 epoch 级别汇总指标
        self.metrics_logger.log_epoch_summary(epoch, explained_var);

        println!(
            "  {} batches: avg_loss={:.4}, policy_loss={:.4}, value_loss={:.4}, explained_var={:.3}, avg_entropy={:.4}",
            stats.num_batches,
            stats.avg_total_loss(),
            stats.avg_policy_loss(),
            stats.avg_value_loss(),
            explained_var,
            stats.avg_entropy(),
        );

        stats
    }

    // ── 数据辅助 ──

    fn flatten_batch(batch: &[PlayRecord], npos: usize) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
        let batch_size = batch.len();
        let encode_len = batch.first().map(|r| r.state.len()).unwrap_or(0);
        let mut states = Vec::with_capacity(batch_size * encode_len);
        let mut policies = Vec::with_capacity(batch_size * npos);
        let mut values = Vec::with_capacity(batch_size);

        for record in batch {
            states.extend(&record.state);
            policies.extend(&record.policy);
            values.push(record.value);
        }
        (states, policies, values)
    }

    // ── 评估指标 ──

    fn explained_variance(predictions: &[f32], targets: &[f32]) -> f32 {
        let n = predictions.len();
        if n < 2 {
            return 0.0;
        }
        let nf = n as f32;
        let mean_target: f32 = targets.iter().sum::<f32>() / nf;
        let var_target: f32 = targets
            .iter()
            .map(|t| (t - mean_target).powi(2))
            .sum::<f32>()
            / (nf - 1.0);
        if var_target < 1e-10 {
            return 1.0;
        }
        let var_residual: f32 = targets
            .iter()
            .zip(predictions)
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f32>()
            / (nf - 1.0);
        1.0 - var_residual / var_target
    }
}
