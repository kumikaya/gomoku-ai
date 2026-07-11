//! RoPE 2D 旋转位置编码（仅作用于 Q 和 K）— 高效版
//!
//! 使用 GPT-NeoX 风格的 rotate_half 替代 interleaved 风格：
//!   - 消除 5D reshape + squeeze/unsqueeze 开销
//!   - 预计算 [1,1,seq,d_head] cos/sin（含符号折叠），forward 零 gather/cat
//!   - 中间张量 ~15→3，kernel launch ~20→5

use burn::module::Module;
use burn::tensor::{Device, Tensor, TensorData};

/// RoPE 2D 旋转位置编码（仅作用于 Q 和 K）。
///
/// 作为 Burn Module, 预计算 cos/sin 张量会自动跟随模型后端切换
/// （如 fork 到 `Autodiff<Cuda>`）。
#[derive(Module, Debug)]
pub struct RoPE2D {
    /// [1, 1, seq, d_head] — 布局: [cos_row | cos_row | cos_col | cos_col]
    cos_full: Tensor<4>,
    /// [1, 1, seq, d_head] — 布局: [-sin_row | sin_row | -sin_col | sin_col]
    sin_full: Tensor<4>,
}

impl RoPE2D {
    pub fn new(board_size: usize, d_head: usize, device: &Device) -> Self {
        assert!(
            d_head >= 4 && d_head % 4 == 0,
            "d_head ({d_head}) must be >= 4 and divisible by 4"
        );
        let quarter = d_head / 4;
        let seq = board_size * board_size;

        // ── 基准频率表 [board_size, quarter] ──
        let mut base_cos = vec![0.0f32; board_size * quarter];
        let mut base_sin = vec![0.0f32; board_size * quarter];
        for pos in 0..board_size {
            for j in 0..quarter {
                let theta = 10_000.0f32.powf(-4.0 * j as f32 / d_head as f32);
                let angle = pos as f32 * theta;
                let idx = pos * quarter + j;
                base_cos[idx] = angle.cos();
                base_sin[idx] = angle.sin();
            }
        }

        // ── 一次性构建 cos_full / sin_full: [1, 1, seq, d_head] ──
        // 维度 4 等分:
        //   cos_full = [cos_row | cos_row | cos_col | cos_col]
        //   sin_full = [-sin_row | sin_row | -sin_col | sin_col]  ← 负号折叠
        let mut cos_data = vec![0.0f32; seq * d_head];
        let mut sin_data = vec![0.0f32; seq * d_head];
        for r in 0..board_size {
            for c in 0..board_size {
                let s = r * board_size + c;
                let base = s * d_head;
                for j in 0..quarter {
                    let (cr, sr) = (base_cos[r * quarter + j], base_sin[r * quarter + j]);
                    let (cc, sc) = (base_cos[c * quarter + j], base_sin[c * quarter + j]);
                    cos_data[base + j] = cr;
                    cos_data[base + quarter + j] = cr;
                    cos_data[base + 2 * quarter + j] = cc;
                    cos_data[base + 3 * quarter + j] = cc;
                    sin_data[base + j] = -sr;
                    sin_data[base + quarter + j] = sr;
                    sin_data[base + 2 * quarter + j] = -sc;
                    sin_data[base + 3 * quarter + j] = sc;
                }
            }
        }

        let cos_full = Tensor::from_data(TensorData::new(cos_data, [1, 1, seq, d_head]), device);
        let sin_full = Tensor::from_data(TensorData::new(sin_data, [1, 1, seq, d_head]), device);
        Self { cos_full, sin_full }
    }

    /// 对 Q 或 K 应用 RoPE 2D。
    ///
    /// - `x`: [batch, n_heads, seq, d_head]
    /// - `board_size`: 棋盘尺寸（seq == board_size²）
    pub fn apply(&self, x: Tensor<4>, board_size: usize) -> Tensor<4> {
        let dims = x.dims();
        let [b, h, s, d] = [dims[0], dims[1], dims[2], dims[3]];
        let quarter = d / 4;
        debug_assert_eq!(s, board_size * board_size);

        // rotate_half(x) via reshape + flip:
        //   x[B,H,S,D] → [B,H,S,2,2,Q] → flip(dim 4) → [B,H,S,D]
        //   等价于 cat([q1, q0, q3, q2])，对 row/col 各半独立做 rotate_half
        let rotated = x
            .clone()
            .reshape([b, h, s, 2, 2, quarter])
            .flip([4])
            .reshape([b, h, s, d]);

        // out = x ⊙ cos_full + rotate_half(x) ⊙ sin_full
        //   cos_full / sin_full 为 [1,1,S,D]，自动广播到 [B,H,S,D]
        x * self.cos_full.clone() + rotated * self.sin_full.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_shape() {
        let device = Device::default();
        let rope = RoPE2D::new(5, 64, &device);
        let x = Tensor::zeros([2, 4, 25, 64], &device);
        let out = rope.apply(x, 5);
        assert_eq!(out.dims(), [2, 4, 25, 64]);
    }

    #[test]
    fn test_modifies_values() {
        let device = Device::default();
        let bs = 3;
        let rope = RoPE2D::new(bs, 32, &device);
        let seq = bs * bs;
        let data: Vec<f32> = (0..(2 * seq * 32))
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let x = Tensor::<4>::from_data(TensorData::new(data, [1, 2, seq, 32]), &device);
        let out = rope.apply(x.clone(), bs);
        let changed = x
            .to_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip(out.to_data().as_slice::<f32>().unwrap().iter())
            .any(|(a, b)| a != b);
        assert!(changed, "RoPE should modify input values");
    }

    #[test]
    fn test_identity_at_origin() {
        // 位置 (0,0) 角度全为 0 → cos=1, sin=0 → 输出 = 输入
        let device = Device::default();
        let rope = RoPE2D::new(4, 16, &device);
        let data: Vec<f32> = (0..256).map(|i| i as f32 + 1.0).collect();
        let x = Tensor::<4>::from_data(TensorData::new(data, [1, 1, 16, 16]), &device);
        let out = rope.apply(x.clone(), 4);
        let in_s = x.to_data().to_vec::<f32>().unwrap();
        let out_s = out.to_data().to_vec::<f32>().unwrap();
        for j in 0..16 {
            assert!(
                (in_s[j] - out_s[j]).abs() < 1e-5,
                "Position (0,0) should be identity: in[{j}]={}, out[{j}]={}",
                in_s[j],
                out_s[j]
            );
        }
    }

    #[test]
    fn test_norm_preservation() {
        // RoPE 是正交变换，应保持每个 token 向量的 L2 范数
        let device = Device::default();
        let bs = 4;
        let d_head = 32;
        let rope = RoPE2D::new(bs, d_head, &device);
        let seq = bs * bs;

        let data: Vec<f32> = (0..(2 * 3 * seq * d_head))
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let x = Tensor::<4>::from_data(TensorData::new(data, [2, 3, seq, d_head]), &device);
        let out = rope.apply(x.clone(), bs);

        let in_s = x.to_data().to_vec::<f32>().unwrap();
        let out_s = out.to_data().to_vec::<f32>().unwrap();

        let total = 2 * 3 * seq;
        for t in 0..total {
            let in_norm: f32 = (0..d_head)
                .map(|j| in_s[t * d_head + j].powi(2))
                .sum::<f32>()
                .sqrt();
            let out_norm: f32 = (0..d_head)
                .map(|j| out_s[t * d_head + j].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                (in_norm - out_norm).abs() < 1e-3,
                "Norm not preserved at token {t}: in={in_norm:.5}, out={out_norm:.5}"
            );
        }
    }
}
