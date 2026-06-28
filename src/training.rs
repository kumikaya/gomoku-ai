//! AlphaZero 训练循环
//!
//! 自对弈 → 收集数据 → 采样训练时随机 D4 增强 →
//! （KL 早停 + 自适应学习率） → 更新参数 → 重复。

use crate::game::board::{D4Symmetry, ENCODE_CHANNELS, NUM_POSITIONS};
use crate::network::residual::{BOARD_SIZE, GomokuNetwork, INPUT_CHANNELS};
use crate::selfplay::{PlayRecord, SelfPlayConfig, self_play};

use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams},
    store::ModuleRecord,
    tensor::{Device, Tensor, activation::log_softmax},
};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::PathBuf;

pub struct TrainConfig {
    pub num_simulations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
    /// buffer 内每个样本被重复训练的遍数
    pub epochs: usize,
    pub num_iterations: usize,
    pub learning_rate: f64,
    pub value_loss_weight: f32,
    /// 每 N 轮保存一次模型权重
    pub save_every: usize,
    /// 模型保存目录
    pub model_dir: PathBuf,
    /// KL 散度目标值（AlphaZero 默认 0.02）
    pub kl_targ: f32,
    /// 经验回放缓冲区最大容量（FIFO 自动淘汰）
    pub buffer_capacity: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_simulations: 200,
            games_per_iteration: 64,
            batch_size: 512,
            epochs: 6,
            num_iterations: 100,
            learning_rate: 1e-3,
            value_loss_weight: 1.0,
            save_every: 5,
            model_dir: PathBuf::from("checkpoints"),
            kl_targ: 0.02,
            buffer_capacity: 12000,
        }
    }
}

pub struct Trainer {
    config: TrainConfig,
    device: Device,
    /// 固定容量 FIFO 缓冲区（自动淘汰最旧数据）
    replay_buffer: VecDeque<PlayRecord>,
    /// 自适应学习率倍率（根据 KL 散度动态调整）
    lr_multiplier: f32,
}

impl Trainer {
    pub fn new(config: TrainConfig, device: Device) -> Self {
        Self {
            config,
            device,
            replay_buffer: VecDeque::new(),
            lr_multiplier: 1.0,
        }
    }

    /// 模型文件路径（不含扩展名，Burn 会自动加 .bpk）
    fn model_path(&self, label: &str) -> PathBuf {
        self.config.model_dir.join(format!("gomoku_{}", label))
    }

    /// 从磁盘加载模型（若存在），否则创建新模型。
    fn load_or_create_model(&self, autodiff_device: &Device) -> GomokuNetwork {
        let latest_path = self.model_path("latest");
        if let Ok(record) = ModuleRecord::load(&latest_path) {
            println!("Loaded existing model from disk.");
            return GomokuNetwork::new(autodiff_device).load_record(record);
        }

        let init_path = self.model_path("initial");
        if let Ok(record) = ModuleRecord::load(&init_path) {
            println!("Loaded initial model from disk.");
            return GomokuNetwork::new(autodiff_device).load_record(record);
        }

        println!("Creating new model.");
        GomokuNetwork::new(autodiff_device)
    }

    /// 保存模型到 `checkpoints/gomoku_{label}.bpk`
    fn save_model(&self, model: &GomokuNetwork, label: &str) {
        let path = self.model_path(label);
        if let Err(e) = model.clone().valid().save_file(&path) {
            eprintln!("Warning: failed to save model {}: {}", label, e);
        } else {
            println!("Model saved: {:?}", path);
        }
    }

    /// AlphaZero 训练循环
    ///
    /// ## 训练流程
    ///
    /// 每次迭代包含两个阶段：
    ///
    /// ### 1. 自对弈（Self-Play）
    /// 使用 valid() 去掉自动微分追踪的推理网络进行 MCTS 引导的自对弈，
    /// 原始对局记录直接存入 FIFO 缓冲区（不扩充，保持数据多样性）。
    ///
    /// ### 2. 训练（Training）
    /// 从 FIFO 缓冲区随机采样批次，每个样本随机应用一种 D4 对称变换后，
    /// 计算策略损失（交叉熵）和价值损失（MSE）。
    /// 每个 mini-batch 后用 KL 散度监控策略更新幅度，若 KL 过大则提前终止 epoch。
    /// 学习率根据 KL 散度自适应调整。
    pub fn train(&mut self) {
        std::fs::create_dir_all(&self.config.model_dir).ok();

        let train_device = self.device.clone().autodiff();

        let mut model = self.load_or_create_model(&train_device);
        let mut optim = AdamConfig::new().init();

        // 保存初始模型
        self.save_model(&model, "initial");

        for iteration in 0..self.config.num_iterations {
            println!(
                "========== Iteration {}/{} ==========",
                iteration + 1,
                self.config.num_iterations
            );

            // ============================================================
            //  1. 自对弈（原始数据直接入 buffer，训练时随机增强）
            // ============================================================
            let inference_network = model.clone().valid();
            println!("  Self-play: {} games...", self.config.games_per_iteration);
            let sp_config = SelfPlayConfig {
                num_simulations: self.config.num_simulations,
                temperature: 1.0,
                temperature_decay_steps: 30,
            };

            let all_records: Vec<PlayRecord> = (0..self.config.games_per_iteration)
                .into_par_iter()
                .flat_map(|_| {
                    let game = self_play(&inference_network, &sp_config, self.device.clone());
                    println!(
                        "    Game finished: {} steps, winner: {:?}",
                        game.num_steps(),
                        game.winner
                    );
                    // 原始记录直接存入 buffer，训练时再做随机 D4 增强
                    game.records
                })
                .collect();

            // FIFO 固定容量：自动淘汰最旧数据
            let added = all_records.len();
            for record in all_records {
                if self.replay_buffer.len() >= self.config.buffer_capacity {
                    self.replay_buffer.pop_front();
                }
                self.replay_buffer.push_back(record);
            }
            println!(
                "  Buffer: {} samples (added {} this iteration)",
                self.replay_buffer.len(),
                added
            );

            // ============================================================
            //  2. 训练
            // ============================================================
            let buffer_size = self.replay_buffer.len();
            println!(
                "  Training: epochs={}, buffer size={}, lr_multiplier={:.3}",
                self.config.epochs, buffer_size, self.lr_multiplier
            );

            if buffer_size < self.config.batch_size {
                println!(
                    "    Buffer too small ({} < batch_size {}), skipping training.",
                    buffer_size, self.config.batch_size
                );
                continue;
            }

            let mut total_loss = 0.0_f32;
            let mut total_steps: usize = 0;

            let mut rng = rand::rng();

            for epoch in 0..self.config.epochs {
                // 每 epoch 打乱索引（不移动实际数据）
                let n = self.replay_buffer.len();
                let mut indices: Vec<usize> = (0..n).collect();
                indices.shuffle(&mut rng);

                let mut epoch_loss = 0.0_f32;
                let mut epoch_steps: usize = 0;
                let mut epoch_kl_sum = 0.0_f32;
                let mut epoch_kl_count: usize = 0;
                let mut all_value_preds: Vec<f32> = Vec::new();
                let mut all_value_targets: Vec<f32> = Vec::new();

                let epoch_effective_lr = self.config.learning_rate * self.lr_multiplier as f64;
                let identity_prob = 1.0 / D4Symmetry::COUNT as f32;
                for chunk in indices.chunks(self.config.batch_size) {
                    let mini_batch: Vec<PlayRecord> = chunk
                        .iter()
                        .map(|&i| {
                            let record = &self.replay_buffer[i];
                            // 训练时随机 D4 增强：每个样本独立随机选一种对称变换
                            let (state, policy) = D4Symmetry::random_augment(
                                &record.state,
                                &record.policy,
                                &mut rng,
                                identity_prob,
                            );
                            PlayRecord {
                                state,
                                policy,
                                value: record.value,
                            }
                        })
                        .collect();
                    let batch_len = mini_batch.len();

                    let (flat_states, flat_policies, flat_values) =
                        Self::flatten_batch(&mini_batch);

                    let state_tensor =
                        Tensor::<1>::from_floats(flat_states.as_slice(), &train_device).reshape([
                            batch_len,
                            INPUT_CHANNELS,
                            BOARD_SIZE,
                            BOARD_SIZE,
                        ]);
                    let policy_target =
                        Tensor::<1>::from_floats(flat_policies.as_slice(), &train_device)
                            .reshape([batch_len, NUM_POSITIONS]);
                    let value_target_tensor =
                        Tensor::<1>::from_floats(flat_values.as_slice(), &train_device)
                            .reshape([batch_len, 1]);

                    // ---- 前向传播（训练用） ----
                    let state_for_new = state_tensor.clone();
                    let (policy_logits, value_pred) = model.forward(state_tensor);

                    // 保存旧策略分布用于 KL 计算
                    let log_probs = log_softmax(policy_logits.clone(), 1);
                    let old_log_probs = log_probs.clone();

                    // ---- 损失计算 ----
                    let policy_loss = -(log_probs * policy_target.clone()).sum_dim(1).mean();

                    let mse = MseLoss::new();
                    let val_pred_for_loss = value_pred.clone();
                    let value_loss = mse.forward(
                        val_pred_for_loss,
                        value_target_tensor.clone(),
                        Reduction::Mean,
                    );

                    let value_weight_tensor =
                        Tensor::<1>::from_floats([self.config.value_loss_weight], &train_device);
                    let loss = policy_loss + value_loss * value_weight_tensor.unsqueeze();

                    let scalar: f32 = loss.clone().into_scalar();
                    epoch_loss += scalar;
                    epoch_steps += 1;

                    // ---- 反向传播 + 参数更新 ----
                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    model = optim.step(epoch_effective_lr, model, grads);

                    // ---- 新策略分布（更新后） + KL 散度 ----
                    let (new_policy_logits, new_value_pred) = model.forward(state_for_new);
                    let new_log_probs = log_softmax(new_policy_logits, 1);

                    let kl = Self::compute_kl(old_log_probs, new_log_probs);
                    epoch_kl_sum += kl;
                    epoch_kl_count += 1;

                    // 收集价值预测用于 explained variance
                    let new_val_pred: Vec<f32> = new_value_pred
                        .reshape([batch_len])
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    all_value_preds.extend(new_val_pred);
                    all_value_targets.extend(flat_values.iter());

                    // ---- KL 早停 ----
                    if kl > self.config.kl_targ * 4.0 {
                        println!(
                            "    Epoch {} early stop: KL={:.5} > kl_targ*4={:.5}",
                            epoch + 1,
                            kl,
                            self.config.kl_targ * 4.0
                        );
                        break;
                    }
                }

                let epoch_avg_loss = if epoch_steps > 0 {
                    epoch_loss / epoch_steps as f32
                } else {
                    0.0
                };

                let avg_kl = if epoch_kl_count > 0 {
                    epoch_kl_sum / epoch_kl_count as f32
                } else {
                    0.0
                };

                let explained_var = Self::explained_variance(&all_value_preds, &all_value_targets);

                println!(
                    "    Epoch {}/{}: {} steps, avg_loss={:.4}, avg_kl={:.5}, explained_var={:.3}",
                    epoch + 1,
                    self.config.epochs,
                    epoch_steps,
                    epoch_avg_loss,
                    avg_kl,
                    explained_var
                );

                total_loss += epoch_loss;
                total_steps += epoch_steps;

                // ---- 自适应学习率 ----
                if avg_kl > self.config.kl_targ * 2.0 && self.lr_multiplier > 0.1 {
                    self.lr_multiplier /= 1.5;
                    println!(
                        "    lr_multiplier ↓ {:.3} (KL={:.5} > target*2={:.5})",
                        self.lr_multiplier,
                        avg_kl,
                        self.config.kl_targ * 2.0
                    );
                } else if avg_kl < self.config.kl_targ / 2.0 && self.lr_multiplier < 10.0 {
                    self.lr_multiplier *= 1.5;
                    println!(
                        "    lr_multiplier ↑ {:.3} (KL={:.5} < target/2={:.5})",
                        self.lr_multiplier,
                        avg_kl,
                        self.config.kl_targ / 2.0
                    );
                }
            }

            let avg_loss = if total_steps > 0 {
                total_loss / total_steps as f32
            } else {
                0.0
            };
            println!("  Average loss: {:.4}", avg_loss);

            // 定期保存
            let epoch = iteration + 1;
            if epoch % self.config.save_every == 0 || epoch == self.config.num_iterations {
                self.save_model(&model, &format!("epoch_{}", epoch));
                model
                    .clone()
                    .valid()
                    .save_file(self.model_path("latest"))
                    .unwrap_or_else(|e| eprintln!("Warning: failed to save latest: {}", e));
            }
        }

        // 保存最终模型
        self.save_model(&model, "final");
        println!("Training complete!");
    }

    // ============================================================
    //  张量辅助
    // ============================================================

    /// 将一批 PlayRecord 扁平化为三个独立浮点数组。
    fn flatten_batch(batch: &[PlayRecord]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let batch_size = batch.len();
        let mut states = Vec::with_capacity(batch_size * ENCODE_CHANNELS * NUM_POSITIONS);
        let mut policies = Vec::with_capacity(batch_size * NUM_POSITIONS);
        let mut values = Vec::with_capacity(batch_size);

        for record in batch {
            states.extend(&record.state);
            policies.extend(&record.policy);
            values.push(record.value);
        }

        (states, policies, values)
    }

    // ============================================================
    //  KL 散度
    // ============================================================

    /// 计算两个策略分布之间的 KL 散度（标量）。
    ///
    /// KL(p_old || p_new) = Σ p_old * (log p_old - log p_new)
    ///
    /// 输入为 log-softmax 后的对数概率，形状 [batch, NUM_POSITIONS]。
    fn compute_kl(old_log_probs: Tensor<2>, new_log_probs: Tensor<2>) -> f32 {
        let old_probs = old_log_probs.clone().exp();
        let kl_per_sample = (old_probs * (old_log_probs - new_log_probs)).sum_dim(1);
        kl_per_sample.mean().into_scalar()
    }

    // ============================================================
    //  Explained Variance
    // ============================================================

    /// 计算价值预测的 explained variance。
    ///
    /// `explained_var = 1 - Var(target - prediction) / Var(target)`
    ///
    /// - 接近 1.0：预测与真实值高度一致
    /// - 接近 0.0：预测几乎等于瞎猜
    /// - 负数：预测比瞎猜还差
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
            return 1.0; // 目标值几乎相同，视为完美解释
        }

        let var_residual: f32 = targets
            .iter()
            .zip(predictions.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f32>()
            / (nf - 1.0);

        1.0 - var_residual / var_target
    }
}
