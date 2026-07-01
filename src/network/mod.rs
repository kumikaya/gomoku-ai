//! Transformer Decoder 五子棋网络
//!
//! 输入：[batch, BOARD_SIZE²] i32  (0=空, 1=黑, 2=白)
//! 输出：
//!   - 策略 logits：[batch, BOARD_SIZE²] 各落子点未归一化分数
//!   - 局势价值：  [batch, 1]  范围 [-1, 1]（Tanh）
//!
//! 架构：
//!   ContentEmbedding(3 → d_model)
//!   + Pos2DEmbed (行/列解耦可学习位置编码)
//!   + N × TransformerBlock (pre-LN self-attention + FFN, ReLU)
//!   + Policy head (per-position Linear) + Value head (mean pool → MLP → Tanh)

pub mod pos_embed;
pub mod transformer;
