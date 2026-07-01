//! CNN (ResNet) 架构的五子棋网络 — 对齐 minizero AlphaZero
//!
//! 输入：[batch, num_input_channels, board_size, board_size]
//!   - 3 通道：0=空, 1=黑, 2=白（one-hot 由上层编码）
//! 输出：
//!   - 策略 logits：[batch, board_size²]  各落子点未归一化分数
//!   - 局势价值：  [batch, 1]  范围 [-1, 1]（Tanh）
//!
//! 架构（对齐 minizero `alphazero_network.py`）：
//!   Conv2d(in_channels → hidden_channels, 3×3) + BN + ReLU
//!   → N × ResidualBlock (Conv 3×3 + BN + ReLU + Conv 3×3 + BN, skip-connection)
//!   → Policy head: Conv2d(hidden → 1, 1×1) + BN + ReLU → Flatten → Linear → logits
//!   → Value head:  Conv2d(hidden → 1, 1×1) + BN + ReLU → Flatten → FC → ReLU → FC → Tanh

use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, conv::Conv2dConfig,
        norm::BatchNormConfig,
    },
    tensor::{
        Device, Tensor,
        activation::{relu, tanh},
    },
};

// ============================================================
//  Residual Block
// ============================================================

#[derive(Module, Debug)]
pub struct ResidualBlock {
    conv1: burn::nn::conv::Conv2d,
    bn1: burn::nn::norm::BatchNorm,
    conv2: burn::nn::conv::Conv2d,
    bn2: burn::nn::norm::BatchNorm,
    dropout: Dropout,
}

impl ResidualBlock {
    pub fn new(num_channels: usize, dropout: f64, device: &Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([num_channels, num_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            bn1: BatchNormConfig::new(num_channels).init(device),
            conv2: Conv2dConfig::new([num_channels, num_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            bn2: BatchNormConfig::new(num_channels).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let input = x.clone();
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = relu(x);
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.dropout.forward(x);
        relu(input + x)
    }
}

// ============================================================
//  GomokuCNN
// ============================================================

#[derive(Module, Debug)]
pub struct GomokuCNN {
    #[module(skip)]
    board_size: usize,
    // Initial conv
    conv: burn::nn::conv::Conv2d,
    bn: burn::nn::norm::BatchNorm,
    dropout_initial: Dropout,
    // Residual blocks
    blocks: Vec<ResidualBlock>,
    // Policy head
    policy_conv: burn::nn::conv::Conv2d,
    policy_bn: burn::nn::norm::BatchNorm,
    policy_fc: Linear,
    // Value head
    value_conv: burn::nn::conv::Conv2d,
    value_bn: burn::nn::norm::BatchNorm,
    value_fc1: Linear,
    value_fc2: Linear,
}

#[derive(Debug, Clone)]
pub struct GomokuCNNConfig {
    pub board_size: usize,
    pub num_input_channels: usize,
    pub num_hidden_channels: usize,
    pub num_blocks: usize,
    pub num_value_hidden_channels: usize,
    pub dropout: f64,
}

impl Default for GomokuCNNConfig {
    fn default() -> Self {
        Self {
            board_size: 8,
            num_input_channels: 3,
            num_hidden_channels: 64,
            num_blocks: 4,
            num_value_hidden_channels: 64,
            dropout: 0.1,
        }
    }
}

impl GomokuCNN {
    pub fn with_config(config: &GomokuCNNConfig, device: &Device) -> Self {
        let bs = config.board_size;
        let n_hidden = config.num_hidden_channels;
        let n_in = config.num_input_channels;
        let npos = bs * bs;

        // 初始卷积
        let conv = Conv2dConfig::new([n_in, n_hidden], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let bn = BatchNormConfig::new(n_hidden).init(device);
        let dropout_initial = DropoutConfig::new(config.dropout).init();

        // 残差块
        let blocks: Vec<ResidualBlock> = (0..config.num_blocks)
            .map(|_| ResidualBlock::new(n_hidden, config.dropout, device))
            .collect();

        // 策略头
        let policy_conv = Conv2dConfig::new([n_hidden, 1], [1, 1]).init(device);
        let policy_bn = BatchNormConfig::new(1).init(device);
        let policy_fc = LinearConfig::new(npos, npos).init(device);

        // 价值头
        let value_conv = Conv2dConfig::new([n_hidden, 1], [1, 1]).init(device);
        let value_bn = BatchNormConfig::new(1).init(device);
        let value_fc1 = LinearConfig::new(npos, config.num_value_hidden_channels).init(device);
        let value_fc2 = LinearConfig::new(config.num_value_hidden_channels, 1).init(device);

        Self {
            board_size: bs,
            conv,
            bn,
            dropout_initial,
            blocks,
            policy_conv,
            policy_bn,
            policy_fc,
            value_conv,
            value_bn,
            value_fc1,
            value_fc2,
        }
    }

    /// 前向传播。
    ///
    /// 输入 [batch, channels, height, width] → 策略 [batch, board_size²] + 价值 [batch, 1]
    pub fn forward(&self, input: Tensor<4>) -> (Tensor<2>, Tensor<2>) {
        let bs = self.board_size;
        let npos = bs * bs;

        // ── Shared trunk ──
        let mut x = self.conv.forward(input);
        x = self.bn.forward(x);
        x = relu(x);
        x = self.dropout_initial.forward(x);

        for block in &self.blocks {
            x = block.forward(x);
        }

        // ── 策略头 ──
        let p = self.policy_conv.forward(x.clone());
        let p = self.policy_bn.forward(p);
        let p = relu(p);
        let [batch, _c, _h, _w] = p.dims();
        let p = p.reshape([batch, npos]);
        let policy_logits = self.policy_fc.forward(p);

        // ── 价值头 ──
        let v = self.value_conv.forward(x);
        let v = self.value_bn.forward(v);
        let v = relu(v);
        let v = v.reshape([batch, npos]);
        let v = self.value_fc1.forward(v);
        let v = relu(v);
        let v = self.value_fc2.forward(v);
        let value = tanh(v);

        (policy_logits, value)
    }
}
