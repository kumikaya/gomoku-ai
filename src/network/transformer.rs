//! Transformer Decoder 架构的五子棋网络
//!
//! 输入：[batch, BOARD_SIZE²]  i32 (0=空, 1=黑, 2=白)
//! 输出：
//!   - 策略 logits：[batch, BOARD_SIZE²]  各落子点未归一化分数
//!   - 局势价值：  [batch, 1]  范围 [-1, 1]（Tanh）
//!
//! 架构：
//!   ContentEmbedding(3 → d_model) + Pos2DEmbed
//!   → N × TransformerBlock (self-attn + FFN, post-LN)
//!   → Policy head (per-position Linear) + Value head (mean pool + MLP → Tanh)

use crate::game::board::BOARD_SIZE;
use crate::network::pos_encoding::{Pos2DEmbed, Pos2DEmbedConfig};

use burn::tensor::activation::{relu, tanh};
use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    },
    tensor::{Device, Int, Tensor},
};

/// Token 种类数（0=空, 1=黑, 2=白）
pub const NUM_TOKENS: usize = 3;

pub fn policy_out_dim(board_size: usize) -> usize {
    board_size * board_size
}

// ============================================================
//  配置
// ============================================================

#[derive(Debug, Clone)]
pub struct GomokuNetworkConfig {
    pub board_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub dropout: f64,
}

impl Default for GomokuNetworkConfig {
    fn default() -> Self {
        Self {
            board_size: BOARD_SIZE,
            d_model: 128,
            d_ff: 512,
            n_heads: 4,
            n_layers: 4,
            dropout: 0.1,
        }
    }
}

// ============================================================
//  Transformer Block（post-LN, decoder-only self-attention）
// ============================================================

#[derive(Module, Debug)]
pub struct TransformerBlock {
    ln1: LayerNorm,
    self_attn: MultiHeadAttention,
    dropout1: Dropout,
    ln2: LayerNorm,
    ffn_w1: Linear,
    ffn_w2: Linear,
    dropout2: Dropout,
}

impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, n_heads: usize, dropout: f64, device: &Device) -> Self {
        let ln1 = LayerNormConfig::new(d_model).init(device);
        let self_attn = MultiHeadAttentionConfig::new(d_model, n_heads)
            .with_dropout(dropout)
            .init(device);
        let dropout1 = DropoutConfig::new(dropout).init();
        let ln2 = LayerNormConfig::new(d_model).init(device);
        let ffn_w1 = LinearConfig::new(d_model, d_ff)
            .with_bias(true)
            .init(device);
        let ffn_w2 = LinearConfig::new(d_ff, d_model)
            .with_bias(true)
            .init(device);
        let dropout2 = DropoutConfig::new(dropout).init();

        Self {
            ln1,
            self_attn,
            dropout1,
            ln2,
            ffn_w1,
            ffn_w2,
            dropout2,
        }
    }

    /// post-LN transformer block 前向
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        // Self-attention sub-layer (post-LN)
        let attn_out = self
            .self_attn
            .forward(MhaInput::self_attn(x.clone()))
            .context;
        let x = self.ln1.forward(x + self.dropout1.forward(attn_out));

        // FFN sub-layer (post-LN)
        let ffn = self.ffn_w2.forward(relu(self.ffn_w1.forward(x.clone())));
        self.ln2.forward(x + self.dropout2.forward(ffn))
    }
}

// ============================================================
//  GomokuNetwork（Transformer Decoder）
// ============================================================

#[derive(Module, Debug)]
pub struct GomokuNetwork {
    pub board_size: usize,
    token_embed: Embedding,
    pos_encoding: Pos2DEmbed,
    blocks: Vec<TransformerBlock>,
    policy_head: Linear,
    value_fc1: Linear,
    value_fc2: Linear,
}

impl GomokuNetwork {
    pub fn new(device: &Device) -> Self {
        Self::with_config(&GomokuNetworkConfig::default(), device)
    }

    pub fn with_config(config: &GomokuNetworkConfig, device: &Device) -> Self {
        let d = config.d_model;

        let token_embed = EmbeddingConfig::new(NUM_TOKENS, d).init(device);
        let pos_encoding = Pos2DEmbedConfig::new(config.board_size, d).init(device);

        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(TransformerBlock::new(
                d,
                config.d_ff,
                config.n_heads,
                config.dropout,
                device,
            ));
        }

        // 策略头：每个位置 → 1 个 logit
        let policy_head = LinearConfig::new(d, 1).init(device);
        // 价值头：mean pool → d → d → 1 → tanh
        let value_fc1 = LinearConfig::new(d, d).init(device);
        let value_fc2 = LinearConfig::new(d, 1).init(device);

        Self {
            board_size: config.board_size,
            token_embed,
            pos_encoding,
            blocks,
            policy_head,
            value_fc1,
            value_fc2,
        }
    }

    /// 前向传播。
    ///
    /// 输入 [batch, BOARD_SIZE²] i32 → 策略 [batch, BOARD_SIZE²] + 价值 [batch, 1]
    pub fn forward(&self, input: Tensor<2, Int>) -> (Tensor<2>, Tensor<2>) {
        let batch = input.dims()[0];
        let seq = input.dims()[1] as usize;

        // ── Embedding ──
        let tok = self.token_embed.forward(input); // [batch, seq, d]
        let mut x = self
            .pos_encoding
            .forward(tok, self.board_size, self.board_size);

        // ── Transformer blocks ──
        for block in &self.blocks {
            x = block.forward(x);
        }

        // ── 策略头：per-position logit ──
        let p = self.policy_head.forward(x.clone()); // [batch, seq, 1]
        let policy_logits = p.reshape([batch, seq]); // [batch, seq]

        // ── 价值头：mean pool → MLP → Tanh ──
        let v = x.clone().mean_dim(1); // [batch, d]
        let v = self.value_fc1.forward(v);
        let v = relu(v);
        let v = self.value_fc2.forward(v);
        let value = tanh(v).squeeze_dim(2); // [batch, 1, 1] → [batch, 1]

        (policy_logits, value)
    }
}
