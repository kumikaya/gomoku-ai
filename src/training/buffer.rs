//! 经验回放缓冲区
//!
//! `RolloutBuffer` 封装了固定容量的 FIFO 缓冲区，支持：
//! - 批量写入（自动淘汰旧数据）
//! - Policy Surprise Weighting 加权随机采样（对齐 KataGo）

use crate::selfplay::PlayRecord;
use rand::RngExt;
use rand::distr::{Distribution, weighted::WeightedIndex};

/// 经验回放缓冲区
pub struct RolloutBuffer {
    buf: Vec<PlayRecord>,
    capacity: usize,
    /// 归一化后的 surprise 权重（与 buf 一一对应）
    surprise_weights: Vec<f32>,
}

impl RolloutBuffer {
    /// 创建容量为 `capacity` 的空缓冲区。
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "buffer capacity must be > 0");
        Self {
            buf: Vec::with_capacity(capacity),
            capacity,
            surprise_weights: Vec::with_capacity(capacity),
        }
    }

    /// 当前样本数。
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// 按索引读取（不可变引用）。
    pub fn get(&self, index: usize) -> &PlayRecord {
        &self.buf[index]
    }

    /// 批量追加记录，超出容量时自动淘汰最旧的数据。
    /// 返回实际新增的样本数。
    pub fn extend(&mut self, records: impl IntoIterator<Item = PlayRecord>) -> usize {
        let mut count = 0;
        for record in records {
            let sw = record.surprise_weight;
            if self.buf.len() >= self.capacity {
                self.buf.remove(0);
                self.surprise_weights.remove(0);
            }
            self.buf.push(record);
            self.surprise_weights.push(sw);
            count += 1;
        }
        count
    }

    /// Policy Surprise 加权随机采样：生成 `0..len` 的随机排列，但惊奇度高的样本
    /// 有更大的概率在前面出现。
    ///
    /// 对齐 KataGo：一半权重均匀，一半权重按 KL 比例分配。
    pub fn sample(&self, rng: &mut impl rand::Rng) -> Vec<usize> {
        let n = self.buf.len();
        if n == 0 {
            return Vec::new();
        }

        let num_to_sample = n;

        // 计算有效权重：uniform_base + surprise_scaled
        let uniform_base = 0.5 / n as f64;

        let max_kl = self.surprise_weights.iter().fold(0.0f32, |a, &b| a.max(b));
        let sum_kl: f64 = if max_kl > 1e-10 {
            self.surprise_weights
                .iter()
                .map(|&w| w as f64)
                .sum::<f64>()
                .max(1e-10)
        } else {
            1.0
        };

        let surprise_scaled: Vec<f64> = if max_kl > 1e-10 {
            self.surprise_weights
                .iter()
                .map(|&w| 0.5 * w as f64 / sum_kl)
                .collect()
        } else {
            vec![0.0; n]
        };

        let weights: Vec<f64> = (0..n).map(|i| uniform_base + surprise_scaled[i]).collect();

        // 加权无放回采样
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut result = Vec::with_capacity(num_to_sample);

        for _ in 0..num_to_sample {
            let w: Vec<f64> = remaining.iter().map(|&i| weights[i]).collect();
            let total: f64 = w.iter().sum();
            if total <= 0.0 {
                // fallback: uniform
                let idx = (rng.random::<f64>() * remaining.len() as f64) as usize;
                result.push(remaining.swap_remove(idx));
            } else {
                let dist = match WeightedIndex::new(&w) {
                    Ok(d) => d,
                    Err(_) => {
                        let idx = (rng.random::<f64>() * remaining.len() as f64) as usize;
                        result.push(remaining.swap_remove(idx));
                        continue;
                    }
                };
                let chosen = dist.sample(rng);
                result.push(remaining.swap_remove(chosen));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_record(value: f32, surprise: f32) -> PlayRecord {
        PlayRecord {
            state: vec![0i32; crate::game::board::ENCODE_LEN],
            policy: vec![0.0f32; crate::game::board::NUM_POSITIONS],
            value,
            surprise_weight: surprise,
        }
    }

    #[test]
    fn capacity_enforced() {
        let mut buf = RolloutBuffer::new(3);
        buf.extend((0..5).map(|i| dummy_record(i as f32, 0.0)));
        assert_eq!(buf.len(), 3);
        assert!((buf.get(0).value - 2.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_sampling_covers_all() {
        let mut buf = RolloutBuffer::new(10);
        buf.extend((0..10).map(|i| dummy_record(i as f32, i as f32)));
        let mut rng = rand::rng();
        let sample = buf.sample(&mut rng);
        let mut sorted = sample.clone();
        sorted.sort();
        assert_eq!(sorted.len(), 10);
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }
}
