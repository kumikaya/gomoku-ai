//! AlphaZero 训练循环
//!
//! 自对弈 → 收集数据 → 采样训练 → 更新参数 → 重复。

use crate::game::board::NUM_POSITIONS;
use crate::network::residual::GobangNetwork;
use crate::selfplay::{PlayRecord, SelfPlayConfig, self_play};

use burn::nn::loss::{MseLoss, Reduction};
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams},
    tensor::{Device, Tensor, activation::log_softmax},
};
use rayon::prelude::*;

pub struct TrainConfig {
    pub num_simulations: usize,
    pub games_per_iteration: usize,
    pub batch_size: usize,
    pub train_steps: usize,
    pub num_iterations: usize,
    pub learning_rate: f64,
    pub value_loss_weight: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_simulations: 200,
            games_per_iteration: 4,
            batch_size: 64,
            train_steps: 50,
            num_iterations: 100,
            learning_rate: 1e-3,
            value_loss_weight: 1.0,
        }
    }
}

pub struct Trainer {
    config: TrainConfig,
    device: Device,
    replay_buffer: Vec<PlayRecord>,
}

impl Trainer {
    pub fn new(config: TrainConfig, device: Device) -> Self {
        Self {
            config,
            device,
            replay_buffer: Vec::new(),
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
        let train_device = self.device.clone().autodiff();

        // 直接在 autodiff 设备上创建训练模型
        let mut model = GobangNetwork::new(&train_device);
        let mut optim = AdamConfig::new().init();

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

                let loss =
                    Self::train_step(&model, &batch, self.config.value_loss_weight, &train_device);

                let scalar: f32 = loss.clone().into_scalar();
                total_loss += scalar;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(self.config.learning_rate, model, grads);

                if step % 10 == 0 {
                    println!(
                        "    Step {}: avg_loss={:.4}",
                        step,
                        total_loss / (step + 1) as f32
                    );
                }
            }

            let avg_loss = total_loss / self.config.train_steps as f32;
            println!("  Average loss: {:.4}", avg_loss);

            // 训练完成，准备下一轮：clone 出推理副本给自对弈用，
            // 原始模型继续在 autodiff 设备上训练（避免 fork 往返丢失 require_grad）
            // 无需额外操作——model 已在 autodiff 设备上，下一轮 clone().valid() 获取推理网络即可
        }
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

    /// 单步训练：前向传播 → 计算损失 → 反向传播
    ///
    /// ## 损失函数
    ///
    /// 总损失 = 策略损失 + value_weight × 价值损失
    ///
    /// - **策略损失（交叉熵）**：`-Σ target_p * log(softmax(logits))`
    ///   MCTS 搜索得到的策略分布作为目标，引导网络学会输出更准确的先验概率。
    ///   这里是 AlphaZero 的核心创新——用 MCTS 改进策略作为监督信号。
    ///
    /// - **价值损失（MSE）**：`(pred_value - actual_outcome)²`
    ///   让网络学会准确评估当前局面的胜负概率。
    fn train_step(
        model: &GobangNetwork,
        batch: &[PlayRecord],
        value_weight: f32,
        autodiff_device: &Device,
    ) -> Tensor<1> {
        let batch_size = batch.len();

        let mut states: Vec<f32> = Vec::with_capacity(batch_size * 4 * NUM_POSITIONS);
        let mut target_policies: Vec<f32> = Vec::with_capacity(batch_size * NUM_POSITIONS);
        let mut target_values: Vec<f32> = Vec::with_capacity(batch_size);

        for record in batch {
            states.extend(&record.state);
            target_policies.extend(&record.policy);
            target_values.push(record.value);
        }

        let state_tensor = Tensor::<1>::from_floats(states.as_slice(), autodiff_device)
            .reshape([batch_size, 4, 15, 15]);

        let policy_target = Tensor::<1>::from_floats(target_policies.as_slice(), autodiff_device)
            .reshape([batch_size, NUM_POSITIONS]);

        let value_target = Tensor::<1>::from_floats(target_values.as_slice(), autodiff_device)
            .reshape([batch_size, 1]);

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
