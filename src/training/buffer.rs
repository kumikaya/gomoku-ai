//! 经验回放缓冲区
//!
//! `RolloutBuffer` 封装了固定容量的 FIFO 缓冲区，支持：
//! - 批量写入（自动淘汰旧数据）
//! - 均匀随机采样（shuffle）

use crate::selfplay::PlayRecord;
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
    pub fn sample(&self, rng: &mut impl rand::Rng) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.buf.len()).collect();
        indices.shuffle(rng);
        indices
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
        let sample = buf.sample(&mut rng);
        let mut sorted = sample.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }
}
