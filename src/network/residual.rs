//! 残差块与 AlphaZero 五子棋网络
//!
//! 输入：[batch, 1, 15, 15]  单通道棋盘编码（-1=对方, 0=空, 1=己方）
//! 输出：
//!   - 策略 logits：[batch, 225]  各落子点未归一化分数
//!   - 局势价值：  [batch, 1]  范围 [-1, 1]（Tanh）

use burn::tensor::activation::{relu, tanh};
use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, conv::Conv2d,
        conv::Conv2dConfig,
    },
    tensor::{Device, FloatDType, Tensor},
};

/// 棋盘大小
pub const BOARD_SIZE: usize = 15;
/// 输入通道数（单通道：-1=对方, 0=空, 1=己方）
pub const INPUT_CHANNELS: usize = 1;
/// 残差块隐藏层通道数
pub const RES_CHANNELS: usize = 128;
/// 策略输出维度
pub const POLICY_OUT: usize = BOARD_SIZE * BOARD_SIZE;

// ============================================================
//  配置
// ============================================================

#[derive(Debug, Clone)]
pub struct GomokuNetworkConfig {
    pub input_channels: usize,
    pub board_size: usize,
    pub num_res_blocks: usize,
    pub res_channels: usize,
}

impl Default for GomokuNetworkConfig {
    fn default() -> Self {
        Self {
            input_channels: INPUT_CHANNELS,
            board_size: BOARD_SIZE,
            num_res_blocks: 5,
            res_channels: RES_CHANNELS,
        }
    }
}

// ============================================================
//  残差块
// ============================================================

#[derive(Module, Debug)]
pub struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl ResidualBlock {
    pub fn new(channels: usize, device: &Device) -> Self {
        let conv1 = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(channels).init(device);
        let conv2 = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(false)
            .init(device);
        let bn2 = BatchNormConfig::new(channels).init(device);

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
        }
    }

    /// 残差块前向传播
    ///
    /// 结构：x → Conv → BN → ReLU → Conv → BN → +residual → ReLU
    ///
    /// 跳跃连接（Skip Connection）：将输入 x 直接加到第二个 BN 的输出上，
    /// 解决深层网络的梯度消失问题，使梯度可以通过恒等映射直接回传。
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let residual = x.clone();

        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = relu(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);

        let x = x + residual;
        relu(x)
    }
}

// ============================================================
//  AlphaZero 完整网络
// ============================================================

#[derive(Module, Debug)]
pub struct GomokuNetwork {
    conv_in: Conv2d,
    bn_in: BatchNorm,
    res_blocks: Vec<ResidualBlock>,
    policy_conv: Conv2d,
    policy_bn: BatchNorm,
    policy_fc: Linear,
    value_conv: Conv2d,
    value_bn: BatchNorm,
    value_fc1: Linear,
    value_fc2: Linear,
}

impl GomokuNetwork {
    pub fn new(device: &Device) -> Self {
        Self::with_config(&GomokuNetworkConfig::default(), device)
    }

    /// 根据配置构建 AlphaZero 网络。
    ///
    /// 架构设计要点：
    /// - **输入卷积** Conv(3×3, input_channels→res_channels)：将 4 通道棋盘编码映射到 128 通道特征空间
    /// - **残差块** × `num_res_blocks`（默认 5 个）：核心特征提取模块，每个块保持通道数不变
    /// - **策略头**：先用 1×1 卷积降维到 2 通道，展平后全连接到 225 维 logits
    /// - **价值头**：先用 1×1 卷积降维到 1 通道，展平后经 64 维隐藏层到标量价值
    ///
    /// 两个输出头共享残差骨干网络，确保特征表示同时服务于走子决策和局势评估。
    pub fn with_config(config: &GomokuNetworkConfig, device: &Device) -> Self {
        let board_sq = config.board_size * config.board_size;
        let res_c = config.res_channels;

        let conv_in = Conv2dConfig::new([config.input_channels, res_c], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(false)
            .init(device);
        let bn_in = BatchNormConfig::new(res_c).init(device);

        let mut res_blocks = Vec::with_capacity(config.num_res_blocks);
        for _ in 0..config.num_res_blocks {
            res_blocks.push(ResidualBlock::new(res_c, device));
        }

        let policy_conv = Conv2dConfig::new([res_c, 2], [1, 1])
            .with_bias(false)
            .init(device);
        let policy_bn = BatchNormConfig::new(2).init(device);
        let policy_fc = LinearConfig::new(2 * board_sq, board_sq)
            .with_bias(true)
            .init(device);

        let value_conv = Conv2dConfig::new([res_c, 1], [1, 1])
            .with_bias(false)
            .init(device);
        let value_bn = BatchNormConfig::new(1).init(device);
        let value_fc1 = LinearConfig::new(board_sq, 64).with_bias(true).init(device);
        let value_fc2 = LinearConfig::new(64, 1).with_bias(true).init(device);

        Self {
            conv_in,
            bn_in,
            res_blocks,
            policy_conv,
            policy_bn,
            policy_fc,
            value_conv,
            value_bn,
            value_fc1,
            value_fc2,
        }
    }

    /// AlphaZero 网络前向传播（f16 半精度计算）
    ///
    /// ## 网络结构
    ///
    /// ```text
    /// 输入 [batch, 1, 15, 15]
    ///   │
    ///   ▼
    /// Conv(3×3, 128) → BN → ReLU          ← 输入卷积层
    ///   │
    ///   ▼
    /// ResidualBlock × 5                    ← 残差骨干（共享特征提取）
    ///   │
    ///   ├──────────┬──────────┐
    ///   ▼          ▼          ▼
    /// 策略头                    价值头
    /// Conv(1×1, 2)             Conv(1×1, 1)
    /// → BN → ReLU              → BN → ReLU
    /// → Flatten [batch, 450]    → Flatten [batch, 225]
    /// → Linear(450, 225)        → Linear(225, 64) → ReLU
    /// → policy_logits           → Linear(64, 1) → Tanh
    /// [batch, 225]              → value [batch, 1]
    /// ```
    ///
    /// 两个输出头的含义：
    /// - **策略 logits**：225 维，每个位置对应一个落子点的未归一化分数；
    ///   在 MCTS 中经掩码 Softmax 后作为先验概率 P(s, a)
    /// - **局势价值**：1 维，范围 [-1, 1]（Tanh 激活）；
    ///   +1 表示当前玩家必胜，-1 表示必败，0 表示均势
    pub fn forward(&self, state: Tensor<4>) -> (Tensor<2>, Tensor<2>) {
        let batch = state.dims()[0];
        let board_sq = POLICY_OUT;

        // 转为 f16 半精度计算
        let state = state.cast(FloatDType::F16);

        let mut x = self.conv_in.forward(state);
        x = self.bn_in.forward(x);
        x = relu(x);

        for block in &self.res_blocks {
            x = block.forward(x);
        }

        // 策略头
        let p = self.policy_conv.forward(x.clone());
        let p = self.policy_bn.forward(p);
        let p = relu(p);
        let p = p.reshape([batch, 2 * board_sq]);
        let policy_logits = self.policy_fc.forward(p);

        // 价值头
        let v = self.value_conv.forward(x);
        let v = self.value_bn.forward(v);
        let v = relu(v);
        let v = v.reshape([batch, board_sq]);
        let v = self.value_fc1.forward(v);
        let v = relu(v);
        let v = self.value_fc2.forward(v);
        let value = tanh(v);

        // 转回 f32 输出
        let policy_logits = policy_logits.cast(FloatDType::F32);
        let value = value.cast(FloatDType::F32);

        (policy_logits, value)
    }
}
