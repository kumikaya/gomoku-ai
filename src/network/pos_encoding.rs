//! 2D 可学习位置编码（行/列解耦）
//!
//! 将棋盘上每个坐标 (row, col) 编码为两个独立的嵌入向量：
//!   - 行位置嵌入：max_board_size 个 d_model 维向量
//!   - 列位置嵌入：max_board_size 个 d_model 维向量
//! 两者相加得到最终 2D 位置编码。
//!
//! forward 时直接从 embedding 权重切片 + 广播，无需构建索引张量。

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{Device, Tensor},
};

#[derive(Debug, Clone)]
pub struct Pos2DEmbedConfig {
    pub max_board_size: usize,
    pub d_model: usize,
}

impl Pos2DEmbedConfig {
    pub fn new(max_board_size: usize, d_model: usize) -> Self {
        Self {
            max_board_size,
            d_model,
        }
    }
}

/// 2D 可学习位置编码模块。
#[derive(Module, Debug)]
pub struct Pos2DEmbed {
    row_embed: Embedding,
    col_embed: Embedding,
}

impl Pos2DEmbedConfig {
    pub fn init(&self, device: &Device) -> Pos2DEmbed {
        Pos2DEmbed {
            row_embed: EmbeddingConfig::new(self.max_board_size, self.d_model).init(device),
            col_embed: EmbeddingConfig::new(self.max_board_size, self.d_model).init(device),
        }
    }
}

impl Pos2DEmbed {
    /// 前向传播：将输入嵌入加上 2D 位置编码后返回。
    ///
    /// - `x`: 输入张量 [batch, seq, d_model]
    /// - `h`: 棋盘高度（行数），必须 ≤ max_board_size
    /// - `w`: 棋盘宽度（列数），必须 ≤ max_board_size
    ///
    /// 返回 `x + pos_encoding` [batch, seq, d_model]。
    pub fn forward(&self, x: Tensor<3>, h: usize, w: usize) -> Tensor<3> {
        let d = x.dims()[2] as usize;

        // 行嵌入: [h, d_model] → [h, 1, d]
        let row = self
            .row_embed
            .weight
            .val()
            .narrow(0, 0, h)
            .reshape([h as i32, 1, d as i32]);

        // 列嵌入: [w, d_model] → [1, w, d]
        let col = self
            .col_embed
            .weight
            .val()
            .narrow(0, 0, w)
            .reshape([1, w as i32, d as i32]);

        // 广播: [h, 1, d] + [1, w, d] → [h, w, d] → [1, h*w, d]
        let pos = (row + col).reshape([1, (h * w) as i32, d as i32]);

        x + pos // [batch, seq, d] + [1, seq, d] 自动广播
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证输出形状 [batch, h*w, d_model]
    #[test]
    fn test_output_shape() {
        let device = Device::default();
        let d = 8;
        let max = 19;
        let pe = Pos2DEmbedConfig::new(max, d).init(&device);

        for (w, h) in [(5, 5), (9, 9), (13, 13)] {
            for batch in [1, 4] {
                let x = Tensor::zeros([batch, w * h, d], &device);
                let out = pe.forward(x, h, w);
                assert_eq!(
                    &*out.shape(),
                    [batch, w * h, d],
                    "w={w} h={h} batch={batch}"
                );
            }
        }
    }

    /// 同坐标在不同 batch 样本中编码相同
    #[test]
    fn test_batch_consistency() {
        let device = Device::default();
        let d = 8;
        let max = 9;
        let pe = Pos2DEmbedConfig::new(max, d).init(&device);

        let w = 3;
        let h = 3;
        let batch = 4;
        let x = Tensor::zeros([batch, w * h, d], &device);
        let out = pe.forward(x, h, w).to_data();

        // 取 batch 0 和 batch 2 的相同 seq 位置，应该完全相等
        let raw = out.as_slice::<f32>().unwrap();
        let stride = w * h * d;
        for seq_pos in 0..w * h {
            let base0 = seq_pos * d;
            let base2 = 2 * stride + seq_pos * d;
            for k in 0..d {
                assert_eq!(
                    raw[base0 + k],
                    raw[base2 + k],
                    "batch0 vs batch2, seq={seq_pos}, dim={k}"
                );
            }
        }
    }

    /// 不同行/列的正弦编码不同位置应产生不同值（随机初始化大概率不碰撞）
    #[test]
    fn test_positions_differ() {
        let device = Device::default();
        let d = 16;
        let max = 9;
        let pe = Pos2DEmbedConfig::new(max, d).init(&device);

        let w = 3;
        let h = 3;
        let x = Tensor::zeros([1, w * h, d], &device);
        let out = pe.forward(x, h, w).to_data();
        let raw = out.as_slice::<f32>().unwrap();

        // 不同位置 (0,0) 和 (1,2) 不应该完全相同
        let pos_a = 0 * d; // seq=0: row=0, col=0
        let pos_b = (1 * w + 2) * d; // seq=1*3+2=5: row=1, col=2
        let same = (0..d).all(|k| raw[pos_a + k] == raw[pos_b + k]);
        assert!(!same, "positions (0,0) and (1,2) should differ");
    }

    /// 较小的棋盘尺寸 wxh < max 仍可正常工作
    #[test]
    fn test_partial_board() {
        let device = Device::default();
        let d = 8;
        let max = 19;
        let pe = Pos2DEmbedConfig::new(max, d).init(&device);

        // 只用 max=19 里的 5x5 子区域
        let w = 5;
        let h = 5;
        let x = Tensor::zeros([1, w * h, d], &device);
        let out = pe.forward(x, h, w);
        assert_eq!(&*out.shape(), [1, 25, d]);
    }
}
