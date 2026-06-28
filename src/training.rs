//! AlphaZero 训练循环
//!
//! 自对弈 → 收集数据 → 采样训练（含 D4 数据增强） → 更新参数 → 重复。

use crate::game::board::{D4Symmetry, NUM_POSITIONS};
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
use rayon::prelude::*;
use std::path::PathBuf;

pub struct TrainConfig {
    pub num_simulations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
    pub train_steps: usize,
    pub num_iterations: usize,
    pub learning_rate: f64,
    pub value_loss_weight: f32,
    /// 每 N 轮保存一次模型权重
    pub save_every: usize,
    /// 模型保存目录
    pub model_dir: PathBuf,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_simulations: 200,
            games_per_iteration: 4,
            batch_size: 128,
            train_steps: 25,
            num_iterations: 100,
            learning_rate: 1e-3,
            value_loss_weight: 1.0,
            save_every: 10,
            model_dir: PathBuf::from("checkpoints"),
        }
    }
}

pub struct Trainer {
    config: TrainConfig,
    device: Device,
    replay_buffer: Vec<PlayRecord>,
}

impl Trainer {
    /// 以 50% 概率保留原样（不做增强）
    const AUGMENT_IDENTITY_PROB: f32 = 0.50;

    pub fn new(config: TrainConfig, device: Device) -> Self {
        Self {
            config,
            device,
            replay_buffer: Vec::new(),
        }
    }

    /// 模型文件路径（不含扩展名，Burn 会自动加 .bpk）
    fn model_path(&self, label: &str) -> PathBuf {
        self.config.model_dir.join(format!("gomoku_{}", label))
    }

    /// 从磁盘加载模型（若存在），否则创建新模型。
    fn load_or_create_model(&self, autodiff_device: &Device) -> GomokuNetwork {
        // 尝试加载 latest 模型
        let latest_path = self.model_path("latest");
        if latest_path.exists() {
            if let Ok(record) = ModuleRecord::load(&latest_path) {
                println!("Loaded existing model from disk.");
                return GomokuNetwork::new(autodiff_device).load_record(record);
            }
        }

        // 尝试加载 initial 模型
        let init_path = self.model_path("initial");
        if init_path.exists() {
            if let Ok(record) = ModuleRecord::load(&init_path) {
                println!("Loaded initial model from disk.");
                return GomokuNetwork::new(autodiff_device).load_record(record);
            }
        }

        println!("Creating new model.");
        GomokuNetwork::new(autodiff_device)
    }

    /// 保存模型到 `checkpoints/gomoku_{label}.bpk`
    fn save_model(&self, model: &GomokuNetwork, label: &str) {
        let path = self.model_path(label);
        // valid() 去除 autodiff 追踪后才能保存到磁盘
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
    /// 使用 valid() 去掉自动微分追踪的推理网络进行 MCTS 引导的自对弈。
    ///
    /// ### 2. 训练（Training）
    /// 从缓冲区随机采样批次，计算策略损失（交叉熵）和价值损失（MSE）。
    ///
    /// ## 设备管理
    ///
    /// 参照 Burn 官方示例（custom-training-loop, mnist），训练模型直接在 autodiff
    /// 设备上创建，推理时通过 `valid()` 获取无自动微分的版本。这样避免了 fork 的
    /// Conv2d 兼容性问题。
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

            // 1. 自对弈 —— 并行执行多局游戏，克隆推理网络给每个线程
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
                    game.records
                })
                .collect();

            self.replay_buffer.extend(all_records);
            let max_cap = 5000;
            if self.replay_buffer.len() > max_cap {
                let drain = self.replay_buffer.len() - max_cap;
                self.replay_buffer.drain(0..drain);
            }

            // 2. 训练
            println!(
                "  Training: {} steps, buffer size={}",
                self.config.train_steps,
                self.replay_buffer.len()
            );

            let mut total_loss = 0.0;

            for step in 0..self.config.train_steps {
                let batch = Self::sample_batch(&self.replay_buffer, self.config.batch_size);
                if batch.is_empty() {
                    continue;
                }

                // 1. 数据增强 + 扁平化
                let (aug_states, aug_policies, aug_values) = Self::augment_batch(&batch);
                let batch_size = batch.len();

                // 2. 构造张量
                let state_tensor = Tensor::<1>::from_floats(aug_states.as_slice(), &train_device)
                    .reshape([batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE]);
                let policy_target =
                    Tensor::<1>::from_floats(aug_policies.as_slice(), &train_device)
                        .reshape([batch_size, NUM_POSITIONS]);
                let value_target = Tensor::<1>::from_floats(aug_values.as_slice(), &train_device)
                    .reshape([batch_size, 1]);

                // 3. 前向 + 损失
                let loss = Self::train_step(
                    &model,
                    state_tensor,
                    policy_target,
                    value_target,
                    self.config.value_loss_weight,
                    &train_device,
                );

                let scalar: f32 = loss.clone().into_scalar();
                total_loss += scalar;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(self.config.learning_rate, model, grads);

                if step % 5 == 0 {
                    println!(
                        "    Step {}: avg_loss={:.4}",
                        step,
                        total_loss / (step + 1) as f32
                    );
                }
            }

            let avg_loss = total_loss / self.config.train_steps as f32;
            println!("  Average loss: {:.4}", avg_loss);

            // 定期保存
            let epoch = iteration + 1;
            if epoch % self.config.save_every == 0 || epoch == self.config.num_iterations {
                self.save_model(&model, &format!("epoch_{}", epoch));
                // 同时更新 latest — valid() 去 autodiff 后保存
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

    fn sample_batch(buffer: &[PlayRecord], batch_size: usize) -> Vec<PlayRecord> {
        if buffer.is_empty() {
            return vec![];
        }
        use rand::prelude::IndexedRandom;
        let mut rng = rand::rng();
        let actual = batch_size.min(buffer.len());
        buffer.sample(&mut rng, actual).cloned().collect()
    }

    /// 对一批样本做 D4 数据增强，转化为扁平浮点数组（供构造张量使用）。
    ///
    /// 以 50% 概率对每个样本保留原样，否则随机应用旋转/翻转。
    /// 五子棋在这些 D4 变换下保持局面不变性，等价于将训练数据扩充 8 倍。
    fn augment_batch(batch: &[PlayRecord]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let batch_size = batch.len();

        let mut states: Vec<f32> = Vec::with_capacity(batch_size * INPUT_CHANNELS * NUM_POSITIONS);
        let mut target_policies: Vec<f32> = Vec::with_capacity(batch_size * NUM_POSITIONS);
        let mut target_values: Vec<f32> = Vec::with_capacity(batch_size);
        let mut rng = rand::rng();

        for record in batch {
            let (aug_state, aug_policy) = D4Symmetry::random_augment(
                &record.state,
                &record.policy,
                &mut rng,
                Self::AUGMENT_IDENTITY_PROB,
            );
            states.extend(&aug_state);
            target_policies.extend(&aug_policy);
            target_values.push(record.value);
        }

        (states, target_policies, target_values)
    }

    /// 单步训练：前向传播 → 计算损失 → 反向传播
    ///
    /// 这是一个纯函数，只负责张量计算。数据增强和批处理由调用方负责。
    ///
    /// ## 损失函数
    ///
    /// 总损失 = 策略损失 + value_weight × 价值损失
    ///
    /// - **策略损失（交叉熵）**：`-Σ target_p * log(softmax(logits))`
    ///   MCTS 搜索得到的策略分布作为目标，引导网络学会输出更准确的先验概率。
    ///
    /// - **价值损失（MSE）**：`(pred_value - actual_outcome)²`
    ///   让网络学会准确评估当前局面的胜负概率。
    fn train_step(
        model: &GomokuNetwork,
        state_tensor: Tensor<4>,
        policy_target: Tensor<2>,
        value_target: Tensor<2>,
        value_weight: f32,
        autodiff_device: &Device,
    ) -> Tensor<1> {
        let (policy_logits, value_pred) = model.forward(state_tensor);

        // 策略损失：交叉熵
        let log_probs = log_softmax(policy_logits, 1);
        let policy_loss = -(log_probs * policy_target).sum_dim(1).mean();

        // 价值损失：MSE
        let mse = MseLoss::new();
        let value_loss = mse.forward(value_pred, value_target, Reduction::Mean);

        let value_weight_tensor = Tensor::<1>::from_floats([value_weight], autodiff_device);
        policy_loss + value_loss * value_weight_tensor.unsqueeze()
    }
}
