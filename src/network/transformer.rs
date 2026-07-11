//! Transformer Decoder 架构的五子棋网络
//!
//! 输入：[batch, board_size²]  i32 (0=空, 1=黑, 2=白)
//! 输出：
//!   - 策略 logits：[batch, board_size²]  各落子点未归一化分数
//!   - 局势价值：  [batch, 1]  范围 [-1, 1]（Tanh）
//!
//! 架构：
//!   ContentEmbedding(3 → d_model) + Pos2DEmbed
//!   → N × TransformerBlock (post-LN, self-attn with RoPE 2D on Q/K + FFN)
//!   → Policy head (per-position Linear) + Value head (mean pool + MLP → Tanh)

use crate::game::board::Board;
use crate::network::pos_embed::{Pos2DEmbed, Pos2DEmbedConfig};
use crate::network::rope::RoPE2D;

use burn::tensor::activation::{relu, softmax, tanh};
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
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
}

impl Default for GomokuNetworkConfig {
    fn default() -> Self {
        Self {
            board_size: Board::DEFAULT_BOARD_SIZE,
            d_model: 128,
            d_ff: 512,
            n_heads: 4,
            n_layers: 4,
        }
    }
}

// ============================================================
//  Transformer Block（post-LN, RoPE 2D self-attention）
// ============================================================

#[derive(Module, Debug)]
pub struct TransformerBlock {
    ln1: LayerNorm,
    qkv_proj: Linear,
    out_proj: Linear,
    ln2: LayerNorm,
    ffn_w1: Linear,
    ffn_w2: Linear,
    n_heads: usize,
    d_head: usize,
    d_model: usize,
}

impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize, n_heads: usize, device: &Device) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        );
        let d_head = d_model / n_heads;

        let ln1 = LayerNormConfig::new(d_model).init(device);
        let qkv_proj = LinearConfig::new(d_model, 3 * d_model)
            .with_bias(false)
            .init(device);
        let out_proj = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);
        let ln2 = LayerNormConfig::new(d_model).init(device);
        let ffn_w1 = LinearConfig::new(d_model, d_ff)
            .with_bias(true)
            .init(device);
        let ffn_w2 = LinearConfig::new(d_ff, d_model)
            .with_bias(true)
            .init(device);

        Self {
            ln1,
            qkv_proj,
            out_proj,
            ln2,
            ffn_w1,
            ffn_w2,
            n_heads,
            d_head,
            d_model,
        }
    }

    /// post-LN transformer block 前向，RoPE 2D 作用于 Q 和 K。
    ///
    /// - `x`: [batch, seq, d_model]
    /// - `rope`: RoPE 2D 实例
    /// - `board_size`: 棋盘尺寸（seq == board_size²）
    pub fn forward(&self, x: Tensor<3>, rope: &RoPE2D, board_size: usize) -> Tensor<3> {
        let [batch, seq, d] = x.dims();

        // ── 投影 QKV → reshape → chunk 为 [q, k, v] 各 [B, H, S, d_head] ──
        let qkv = self.qkv_proj.forward(x.clone()); // [B, S, 3D]
        let qkv = qkv
            .reshape([batch, seq, 3 * self.n_heads, self.d_head])
            .swap_dims(1, 2); // [B, 3H, S, d_head]
        let h = self.n_heads;
        let q = qkv.clone().narrow(1, 0, h);
        let k = qkv.clone().narrow(1, h, h);
        let v = qkv.narrow(1, 2 * h, h);

        // ── RoPE 2D ──
        let q = rope.apply(q, board_size);
        let k = rope.apply(k, board_size);

        // ── Scaled dot-product attention ──
        let scale = (self.d_head as f32).sqrt();
        let attn = q.matmul(k.swap_dims(2, 3)); // [B, H, S_q, S_k]
        let attn = softmax(attn.mul_scalar(1.0 / scale), 3);
        let out = attn.matmul(v); // [B, H, S, d]

        // ── reshape back: [B, H, S, d] → [B, S, D] ──
        let out = out.swap_dims(1, 2).reshape([batch, seq, d]);

        // ── 输出投影 ──
        let attn_out = self.out_proj.forward(out);

        // ── post-LN: attn sub-layer ──
        let x = self.ln1.forward(x + attn_out);

        // ── post-LN: FFN sub-layer ──
        let ffn = self.ffn_w2.forward(relu(self.ffn_w1.forward(x.clone())));
        self.ln2.forward(x + ffn)
    }
}

// ============================================================
//  GomokuNetwork（Transformer Decoder）
// ============================================================

#[derive(Module, Debug)]
pub struct GomokuNetwork {
    #[module(skip)]
    pub board_size: usize,
    rope: RoPE2D,
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
        let d_head = d / config.n_heads;

        let token_embed = EmbeddingConfig::new(NUM_TOKENS, d).init(device);
        let pos_encoding = Pos2DEmbedConfig::new(config.board_size, d).init(device);
        let rope = RoPE2D::new(config.board_size, d_head, device);

        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(TransformerBlock::new(
                d,
                config.d_ff,
                config.n_heads,
                device,
            ));
        }

        // 策略头：每个位置 → 1 个 logit
        let policy_head = LinearConfig::new(d, 1).with_bias(false).init(device);
        // 价值头：mean pool → d → d → 1 → tanh
        let value_fc1 = LinearConfig::new(d, d).with_bias(true).init(device);
        let value_fc2 = LinearConfig::new(d, 1).with_bias(true).init(device);

        Self {
            board_size: config.board_size,
            rope,
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
    /// 输入 [batch, board_size²] i32 → 策略 [batch, board_size²] + 价值 [batch, 1]
    pub fn forward(&self, input: Tensor<2, Int>) -> (Tensor<2>, Tensor<2>) {
        let batch = input.dims()[0];
        let seq = input.dims()[1] as usize;
        let board = self.board_size;

        // ── Embedding ──
        let tok = self.token_embed.forward(input); // [batch, seq, d]
        let mut x = self.pos_encoding.forward(tok, board, board);

        // ── Transformer blocks (with RoPE 2D on Q/K) ──
        for block in &self.blocks {
            x = block.forward(x, &self.rope, board);
        }

        // ── 策略头：per-position logit ──
        let p = self.policy_head.forward(x.clone()); // [batch, seq, 1]
        let policy_logits = p.reshape([batch, seq]); // [batch, seq]

        // ── 价值头：mean pool → MLP → Tanh ──
        let v = x.mean_dim(1); // [batch, d]
        let v = relu(self.value_fc1.forward(v));
        let v = self.value_fc2.forward(v);
        let value = tanh(v).squeeze_dim(2); // [batch, 1]

        (policy_logits, value)
    }
}
