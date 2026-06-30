//! 训练器
//!
//! `TrainConfig` 配置训练超参数，`Trainer` 执行完整的 AlphaZero 训练循环。

use crate::game::board::{D4Symmetry, ENCODE_CHANNELS, NUM_POSITIONS};
use crate::inference::InferenceServer;
use crate::network::residual::{BOARD_SIZE, GomokuNetwork, INPUT_CHANNELS};
use crate::selfplay::{PlayRecord, SelfPlayConfig, self_play};

use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams},
    store::ModuleRecord,
    tensor::{Device, FloatDType, Tensor, activation::log_softmax},
};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::PathBuf;

// ── 配置 ──

pub struct TrainConfig {
    pub num_simulations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub num_iterations: usize,
    pub learning_rate: f64,
    pub value_loss_weight: f32,
    pub save_every: usize,
    pub model_dir: PathBuf,
    pub kl_targ: f32,
    pub buffer_capacity: usize,
    pub max_grad_norm: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_simulations: 200,
            games_per_iteration: 64,
            batch_size: 512,
            epochs: 2,
            num_iterations: 100,
            learning_rate: 1e-3,
            value_loss_weight: 1.0,
            save_every: 5,
            model_dir: PathBuf::from("checkpoints"),
            kl_targ: 0.02,
            buffer_capacity: 12000,
            max_grad_norm: 1.0,
        }
    }
}

// ── 训练器 ──

pub struct Trainer {
    config: TrainConfig,
    device: Device,
    replay_buffer: VecDeque<PlayRecord>,
}

impl Trainer {
    pub fn new(config: TrainConfig, device: Device) -> Self {
        Self {
            config,
            device,
            replay_buffer: VecDeque::new(),
        }
    }

    // ── 模型持久化 ──

    fn model_path(&self, label: &str) -> PathBuf {
        self.config.model_dir.join(format!("gomoku_{}", label))
    }

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
        let mut model = self.load_or_create_model(&train_device);

        let mut optim = AdamConfig::new().init();
        if self.config.max_grad_norm > 0.0 {
            optim = optim
                .with_grad_clipping(GradientClippingConfig::Norm(self.config.max_grad_norm).init());
        }

        self.save_model(&model, "initial");

        let inference_server = InferenceServer::new(model.clone().valid(), self.device.clone());

        for iteration in 0..self.config.num_iterations {
            println!(
                "========== Iteration {}/{} ==========",
                iteration + 1,
                self.config.num_iterations
            );

            self.run_self_play(&inference_server);

            let buffer_size = self.replay_buffer.len();
            println!(
                "  Training: epochs={}, buffer size={}",
                self.config.epochs, buffer_size
            );

            if buffer_size < self.config.batch_size {
                println!("    Buffer too small, skipping.");
                continue;
            }

            let (total_loss, total_steps) =
                self.run_training_epochs(&mut model, &mut optim, &train_device);

            let avg_loss = if total_steps > 0 {
                total_loss / total_steps as f32
            } else {
                0.0
            };
            println!("  Average loss: {:.4}", avg_loss);

            inference_server.update_model(model.clone().valid());

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

        self.save_model(&model, "final");
        println!("Training complete!");
    }

    // ── 自对弈 ──

    fn run_self_play(&mut self, inference_server: &InferenceServer) {
        println!("  Self-play: {} games...", self.config.games_per_iteration);

        let sp_config = SelfPlayConfig {
            num_simulations: self.config.num_simulations,
        };

        let all_records: Vec<PlayRecord> = (0..self.config.games_per_iteration)
            .into_par_iter()
            .flat_map(|_| {
                let game = self_play(inference_server, &sp_config);
                println!(
                    "    Game finished: {} steps, winner: {:?}",
                    game.num_steps(),
                    game.winner
                );
                game.records
            })
            .collect();

        let added = all_records.len();
        for record in all_records {
            if self.replay_buffer.len() >= self.config.buffer_capacity {
                self.replay_buffer.pop_front();
            }
            self.replay_buffer.push_back(record);
        }
        println!(
            "  Buffer: {} samples (+{})",
            self.replay_buffer.len(),
            added
        );
    }

    // ── 训练阶段 ──

    fn run_training_epochs(
        &mut self,
        model: &mut GomokuNetwork,
        optim: &mut burn::optim::ModuleOptimizer,
        train_device: &Device,
    ) -> (f32, usize) {
        let lr = self.config.learning_rate;
        let mut total_loss = 0.0_f32;
        let mut total_steps: usize = 0;
        let mut rng = rand::rng();
        let identity_prob = 1.0 / D4Symmetry::COUNT as f32;

        for epoch in 0..self.config.epochs {
            let n = self.replay_buffer.len();
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);

            let mut epoch_loss = 0.0_f32;
            let mut epoch_steps: usize = 0;
            let mut epoch_entropy_sum = 0.0_f32;
            let mut epoch_entropy_count: usize = 0;
            let mut all_value_preds: Vec<f32> = Vec::new();
            let mut all_value_targets: Vec<f32> = Vec::new();

            for chunk in indices.chunks(self.config.batch_size) {
                let mini_batch: Vec<PlayRecord> = chunk
                    .iter()
                    .map(|&i| {
                        let record = &self.replay_buffer[i];
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

                let (flat_states, flat_policies, flat_values) = Self::flatten_batch(&mini_batch);

                let state_tensor = Tensor::<1>::from_floats(flat_states.as_slice(), train_device)
                    .reshape([batch_len, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE]);
                let policy_target =
                    Tensor::<1>::from_floats(flat_policies.as_slice(), train_device)
                        .reshape([batch_len, NUM_POSITIONS]);
                let value_target_tensor =
                    Tensor::<1>::from_floats(flat_values.as_slice(), train_device)
                        .reshape([batch_len, 1]);

                let state_for_new = state_tensor.clone();

                let (policy_logits, value_pred) = model.forward(state_tensor);

                let log_probs = log_softmax(policy_logits.clone(), 1);
                let policy_loss = -(log_probs * policy_target.clone()).sum_dim(1).mean();

                let mse = MseLoss::new();
                let value_loss = mse.forward(
                    value_pred.clone(),
                    value_target_tensor.clone(),
                    Reduction::Mean,
                );

                let value_weight_tensor =
                    Tensor::<1>::from_floats([self.config.value_loss_weight], train_device);
                let loss = policy_loss + value_loss * value_weight_tensor.unsqueeze();

                let scalar: f32 = loss.clone().into_scalar();
                epoch_loss += scalar;
                epoch_steps += 1;

                // 反向传播 + 参数更新
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, model);
                *model = optim.step(lr, model.clone(), grads);

                // 更新后统计
                let (new_policy_logits, new_value_pred) = model.forward(state_for_new);
                let new_log_probs = log_softmax(new_policy_logits.clone(), 1);
                let probs = new_log_probs.clone().exp();
                let entropy = -(probs * new_log_probs)
                    .sum_dim(1)
                    .mean()
                    .into_scalar::<f32>();
                epoch_entropy_sum += entropy;
                epoch_entropy_count += 1;

                let new_val_pred: Vec<f32> = new_value_pred
                    .reshape([batch_len])
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec()
                    .unwrap();
                all_value_preds.extend(new_val_pred);
                all_value_targets.extend(flat_values.iter());
            }

            let avg_loss = if epoch_steps > 0 {
                epoch_loss / epoch_steps as f32
            } else {
                0.0
            };
            let avg_entropy = if epoch_entropy_count > 0 {
                epoch_entropy_sum / epoch_entropy_count as f32
            } else {
                0.0
            };
            let explained_var = Self::explained_variance(&all_value_preds, &all_value_targets);

            println!(
                "    Epoch {}/{}: {} steps, avg_loss={:.4}, explained_var={:.3}, avg_entropy={:.4}",
                epoch + 1,
                self.config.epochs,
                epoch_steps,
                avg_loss,
                explained_var,
                avg_entropy,
            );

            total_loss += epoch_loss;
            total_steps += epoch_steps;
        }

        (total_loss, total_steps)
    }

    // ── 数据辅助 ──

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
