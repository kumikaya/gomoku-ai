//! 经验回放缓冲区（KataGo 风格窗口化加权采样）
//!
//! `RolloutBuffer` 封装了固定容量的 FIFO 缓冲区，支持：
//! - 批量写入（自动淘汰旧数据）
//! - 均匀采样（shuffle）
//! - 窗口化加权采样（近期数据权重更高，加速适应新策略）

use crate::selfplay::PlayRecord;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::seq::SliceRandom;
use std::collections::VecDeque;

/// 经验回放缓冲区
pub struct RolloutBuffer {
    buf: VecDeque<PlayRecord>,
    capacity: usize,
}

impl RolloutBuffer {
    /// 创建容量为 `capacity` 的空缓冲区。
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "buffer capacity must be > 0");
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
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
            if self.buf.len() >= self.capacity {
                self.buf.pop_front();
            }
            self.buf.push_back(record);
            count += 1;
        }
        count
    }

    /// 均匀随机采样：生成 `0..len` 的随机排列。
    pub fn sample_uniform(&self, rng: &mut impl rand::Rng) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.buf.len()).collect();
        indices.shuffle(rng);
        indices
    }

    /// 窗口化加权采样：越新的数据（索引越大）权重越高。
    ///
    /// `recent_bonus` 控制新近度加成强度，0.0 = 均匀，推荐 0.5~1.0。
    ///
    /// 权重公式：`w[i] = 1.0 + recent_bonus * (i / n)`
    /// 即最旧样本权重=1，最新样本权重=1+recent_bonus。
    ///
    /// 返回 `len` 个索引的有放回采样序列（标准 on-policy RL 做法）。
    pub fn sample_weighted(&self, rng: &mut impl rand::Rng, recent_bonus: f32) -> Vec<usize> {
        let n = self.buf.len();
        if recent_bonus <= 0.0 || n <= 1 {
            return self.sample_uniform(rng);
        }
        let nf = n as f32;
        let weights: Vec<f32> = (0..n)
            .map(|i| 1.0 + recent_bonus * (i as f32 / nf))
            .collect();
        let dist = WeightedIndex::new(&weights)
            .expect("RolloutBuffer: failed to create WeightedIndex (weights must be >0)");
        (0..n).map(|_| dist.sample(rng)).collect()
    }

    /// 通用采样：`recent_bonus = 0` 时用均匀 shuffle，否则加权采样。
    pub fn sample(&self, rng: &mut impl rand::Rng, recent_bonus: f32) -> Vec<usize> {
        if recent_bonus <= 0.0 {
            self.sample_uniform(rng)
        } else {
            self.sample_weighted(rng, recent_bonus)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_record(value: f32) -> PlayRecord {
        PlayRecord {
            state: vec![0i32; crate::game::board::ENCODE_LEN],
            policy: vec![0.0f32; crate::game::board::NUM_POSITIONS],
            value,
        }
    }

    #[test]
    fn capacity_enforced() {
        let mut buf = RolloutBuffer::new(3);
        buf.extend((0..5).map(|i| dummy_record(i as f32)));
        assert_eq!(buf.len(), 3);
        // 最旧的 0,1 被淘汰，剩余 2,3,4
        assert!((buf.get(0).value - 2.0).abs() < 1e-6);
    }

    #[test]
    fn uniform_sampling_covers_all() {
        let mut buf = RolloutBuffer::new(10);
        buf.extend((0..10).map(|i| dummy_record(i as f32)));
        let mut rng = rand::rng();
        let sample = buf.sample_uniform(&mut rng);
        let mut sorted = sample.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn weighted_sampling() {
        let mut buf = RolloutBuffer::new(100);
        buf.extend((0..100).map(|i| dummy_record(i as f32)));
        let mut rng = rand::rng();
        let sample = buf.sample_weighted(&mut rng, 1.0);
        assert_eq!(sample.len(), 100);
        // 新近样本应该有更高出现频率（不 strict 但样本中有更大索引）
        let late_fraction = sample.iter().filter(|&&i| i >= 80).count() as f64 / 100.0;
        // 在 1.0 bonus 下后 20% 数据应有远超均匀的采样率
        assert!(late_fraction > 0.15);
    }
}
