//! 训练器
//!
//! `TrainConfig` 配置训练超参数，`Trainer` 执行完整的 AlphaZero 训练循环。

use crate::game::board::{D4Symmetry, ENCODE_LEN, NUM_POSITIONS};
use crate::inference::InferenceServer;
use crate::network::transformer::GomokuNetwork;
use crate::selfplay::{PlayRecord, SelfPlayConfig, self_play};
use crate::training::buffer::RolloutBuffer;
use crate::training::lr_schedule::LrSchedule;

use crate::eval::{BaselineManager, EloTracker, EvalConfig, MatchRunner};

use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::ModuleOptimizer;
use burn::store::ModuleRecord;
use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams},
    tensor::{Device, FloatDType, Int, Tensor, activation::log_softmax},
};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::path::PathBuf;

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
    /// KataGo Playout Cap：启用后每步在 `[min, max]` 内均匀随机模拟次数。
    pub playout_cap_enabled: bool,
    /// 下界比例（相对于 `num_simulations`）。
    pub playout_cap_min_ratio: f32,
    /// LR warmup 比例：前 `lr_warmup_ratio` 比例迭代内线性 warmup，之后 cosine 衰减到 `lr_final_ratio`。
    /// 0.05 = 前 5% 迭代 warmup。设为 0 表示仅 cosine 衰减（无 warmup）。
    pub lr_warmup_ratio: f32,
    /// 最终学习率比例（相对于 `learning_rate`），cosine 衰减的底值。
    pub lr_final_ratio: f32,
    /// 窗口化 buffer 加权采样：近期数据采样权重乘数。
    /// 0.0 = 关（均匀采样），推荐 0.5~1.0。
    pub buffer_recent_bonus: f32,
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
            num_simulations: 32,
            games_per_iteration: 64,
            batch_size: 512,
            mini_batches_per_iteration: 100,
            num_iterations: 100,
            learning_rate: 1e-3,
            value_loss_weight: 0.5,
            save_every: 5,
            model_dir: PathBuf::from("checkpoints"),
            buffer_capacity: 80000,
            max_grad_norm: 2.0,
            checkpoint: None,
            playout_cap_enabled: false,
            playout_cap_min_ratio: 0.25,
            lr_warmup_ratio: 0.05,
            lr_final_ratio: 0.1,
            buffer_recent_bonus: 0.0,
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
    buffer: RolloutBuffer,
}

impl Trainer {
    pub fn new(config: TrainConfig, device: Device) -> Self {
        let cap = config.buffer_capacity;
        Self {
            config,
            device,
            buffer: RolloutBuffer::new(cap),
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

        let mut optim = AdamWConfig::new().init();
        if self.config.max_grad_norm > 0.0 {
            optim = optim
                .with_grad_clipping(GradientClippingConfig::Norm(self.config.max_grad_norm).init());
        }

        let inference_server = InferenceServer::new(model.clone().valid(), self.device.clone());

        // 评估基础设施
        let eval_config = EvalConfig {
            num_games: self.config.eval_num_games,
            num_simulations: self.config.eval_num_simulations,
            promotion_threshold: self.config.eval_promotion_threshold,
            eval_every: self.config.eval_every,
        };
        let match_runner = if self.config.eval_enabled {
            Some(MatchRunner::new(eval_config))
        } else {
            println!("  Eval disabled.");
            None
        };
        let baseline_mgr = BaselineManager::new(self.config.model_dir.clone());
        let mut elo = EloTracker::new();

        // 初始模型作为 baseline（如果尚未存在）
        if self.config.eval_enabled && baseline_mgr.load_baseline(&self.device).is_none() {
            baseline_mgr.promote(&model);
        }

        let mut master_rng = StdRng::seed_from_u64(self.config.random_seed);
        println!("  Random seed: {}", self.config.random_seed);

        let lr_sched = LrSchedule::new(
            self.config.num_iterations,
            self.config.lr_warmup_ratio,
            self.config.lr_final_ratio,
        );
        println!(
            "  LR schedule: warmup_ratio={:.2}, final_ratio={:.2}, base_lr={:.6}",
            self.config.lr_warmup_ratio, self.config.lr_final_ratio, self.config.learning_rate,
        );

        for iteration in 0..self.config.num_iterations {
            println!(
                "========== Iteration {}/{} ==========",
                iteration + 1,
                self.config.num_iterations
            );

            let lr = lr_sched.lr(iteration, self.config.learning_rate);

            self.run_self_play(&inference_server, iteration, &mut master_rng);

            let buffer_size = self.buffer.len();
            println!(
                "  Training: mini_batches={}, buffer size={}, recent_bonus={:.1}",
                self.config.mini_batches_per_iteration,
                buffer_size,
                self.config.buffer_recent_bonus
            );

            if buffer_size < self.config.batch_size {
                println!("    Buffer too small, skipping.");
                continue;
            }

            self.run_training_batches(&mut model, &mut optim, &train_device, lr, &mut master_rng);

            inference_server.update_model(model.clone().valid());

            let epoch = iteration + 1;

            // ── 对抗评估 ──
            if self.config.eval_enabled && epoch % self.config.eval_every == 0 {
                if let Some(ref runner) = match_runner {
                    if let Some(baseline) = baseline_mgr.load_baseline(&self.device) {
                        println!();
                        println!("  === Tournament Evaluation (iter {}) ===", epoch);
                        let result = runner.run_match(
                            model.clone().valid(),
                            baseline,
                            self.device.clone(),
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
                            baseline_mgr.promote(&model);
                        } else {
                            println!(
                                "  Win rate {:.1}% <= threshold {:.0}%, baseline unchanged",
                                wr * 100.0,
                                self.config.eval_promotion_threshold * 100.0,
                            );
                        }
                    } else {
                        println!("  Baseline missing, promoting current model.");
                        baseline_mgr.promote(&model);
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

        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Self-play: {bar:40.cyan/blue} {pos}/{len} ({eta})")
                .unwrap(),
        );

        if self.config.playout_cap_enabled {
            println!(
                "  Playout Cap: enabled (sims in [{:.0}%, 100%] of {}), target_weight ∝ sims",
                self.config.playout_cap_min_ratio * 100.0,
                self.config.num_simulations,
            );
        }

        let sp_config = SelfPlayConfig {
            num_simulations: self.config.num_simulations,
            select_temperature: temp,
            playout_cap_enabled: self.config.playout_cap_enabled,
            playout_cap_min_ratio: self.config.playout_cap_min_ratio,
        };

        let base_seed = master_rng.random::<u64>();

        let all_records: Vec<PlayRecord> = (0..total)
            .into_par_iter()
            .flat_map(|game_i| {
                let seed = base_seed.wrapping_add(game_i as u64);
                let mut game_rng = StdRng::seed_from_u64(seed);
                let game = self_play(inference_server, &sp_config, &mut game_rng);
                pb.inc(1);
                game.records
            })
            .collect();

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
    ) -> (f32, usize) {
        let mut total_loss = 0.0_f32;
        let mut total_steps: usize = 0;
        let identity_prob = 1.0 / D4Symmetry::COUNT as f32;

        let num_batches = self.config.mini_batches_per_iteration;
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Training: {bar:40.green/dim} {pos}/{len} batches ({eta})")
                .unwrap(),
        );

        let mut total_entropy_sum = 0.0_f32;
        let mut total_entropy_count: usize = 0;
        let mut all_value_preds: Vec<f32> = Vec::new();
        let mut all_value_targets: Vec<f32> = Vec::new();

        for _ in 0..num_batches {
            // 每次 mini-batch 独立从 buffer 采样 batch_size 个样本
            let batch_size = self.config.batch_size.min(self.buffer.len());
            let all_indices = self.buffer.sample(rng, self.config.buffer_recent_bonus);
            let chunk = &all_indices[..batch_size];

            // 透传 target_weight 用于策略损失加权
            let mut batch_target_weights = Vec::with_capacity(chunk.len());
            let mini_batch: Vec<PlayRecord> = chunk
                .iter()
                .map(|&i| {
                    let record = self.buffer.get(i);
                    batch_target_weights.push(record.target_weight);
                    let (state, policy) = D4Symmetry::random_augment(
                        &record.state,
                        &record.policy,
                        rng,
                        identity_prob,
                    );
                    PlayRecord {
                        state,
                        policy,
                        value: record.value,
                        target_weight: record.target_weight,
                    }
                })
                .collect();
            let batch_len = mini_batch.len();

            let (flat_states, flat_policies, flat_values) = Self::flatten_batch(&mini_batch);

            let state_tensor = Tensor::<2, Int>::from_data(
                burn::tensor::TensorData::new(flat_states, [batch_len as i32, ENCODE_LEN as i32]),
                train_device,
            );
            let policy_target = Tensor::<1>::from_floats(flat_policies.as_slice(), train_device)
                .reshape([batch_len, NUM_POSITIONS]);
            let value_target_tensor =
                Tensor::<1>::from_floats(flat_values.as_slice(), train_device)
                    .reshape([batch_len, 1]);
            let weight_tensor =
                Tensor::<1>::from_floats(batch_target_weights.as_slice(), train_device)
                    .unsqueeze_dim(1);

            let (policy_logits, value_pred) = model.forward(state_tensor);

            let log_probs = log_softmax(policy_logits.clone(), 1);

            // 统计
            {
                let log_probs = log_probs.clone().detach();
                let probs = log_probs.clone().exp();
                let entropy = -(probs * log_probs).sum_dim(1).mean().into_scalar::<f32>();
                total_entropy_sum += entropy;
                total_entropy_count += 1;

                let val_pred: Vec<f32> = value_pred
                    .clone()
                    .reshape([batch_len])
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec()
                    .unwrap();
                all_value_preds.extend(val_pred);
                all_value_targets.extend(flat_values.iter());
            }

            // 策略损失：按 target_weight 加权聚合（低质量样本对策略梯度贡献小）
            let per_sample_policy_loss = -(log_probs * policy_target.clone()).sum_dim(1); // [batch, 1]

            let total_weight = weight_tensor.clone().sum();
            let weighted_policy_loss =
                (per_sample_policy_loss * weight_tensor).sum() / total_weight;

            let mse = MseLoss::new();
            let value_loss = mse.forward(
                value_pred.clone(),
                value_target_tensor.clone(),
                Reduction::Mean,
            );

            let loss = weighted_policy_loss + value_loss * self.config.value_loss_weight;
            let scalar: f32 = loss.clone().into_scalar();
            total_loss += scalar;
            total_steps += 1;

            // 反向传播 + 参数更新
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            *model = optim.step(lr, model.clone(), grads);

            pb.inc(1);
        }

        let avg_loss = if total_steps > 0 {
            total_loss / total_steps as f32
        } else {
            0.0
        };
        let avg_entropy = if total_entropy_count > 0 {
            total_entropy_sum / total_entropy_count as f32
        } else {
            0.0
        };
        let explained_var = Self::explained_variance(&all_value_preds, &all_value_targets);
        pb.finish_and_clear();

        println!(
            "  {} batches: avg_loss={:.4}, explained_var={:.3}, avg_entropy={:.4}",
            total_steps, avg_loss, explained_var, avg_entropy,
        );

        (total_loss, total_steps)
    }

    // ── 数据辅助 ──

    fn flatten_batch(batch: &[PlayRecord]) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
        let batch_size = batch.len();
        let mut states = Vec::with_capacity(batch_size * ENCODE_LEN);
        let mut policies = Vec::with_capacity(batch_size * NUM_POSITIONS);
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
