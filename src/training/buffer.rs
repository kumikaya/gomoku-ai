//! 经验回放缓冲区
//!
//! `RolloutBuffer` 封装了固定容量的 FIFO 缓冲区，支持：
//! - 批量写入（自动淘汰旧数据）
//! - Policy Surprise Weighting：高权重样本写入时自动复制多份（对齐 KataGo）
//! - 均匀随机采样（shuffle）

use crate::selfplay::PlayRecord;
use rand::RngExt;
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

    /// 遍历所有样本的不可变引用。
    pub fn iter(&self) -> impl Iterator<Item = &PlayRecord> {
        self.buf.iter()
    }

    /// 批量追加记录，超出容量时自动淘汰最旧的数据。
    ///
    /// Policy Surprise Weighting (KataGo):
    /// `sample_weight` 控制样本的写入份数。权重为 w 的样本会写入
    /// `floor(w)` 份，外加 `w - floor(w)` 概率额外写入一份。
    /// 返回实际新增的样本数。
    pub fn extend(&mut self, records: impl IntoIterator<Item = PlayRecord>) -> usize {
        let mut rng = rand::rng();
        let mut count = 0;
        for mut record in records {
            let w = record.sample_weight.max(1.0);
            let whole = w.trunc() as usize;
            let frac = w - whole as f32;

            // 写入整数份，每个副本的 sample_weight 归一化为 1.0
            record.sample_weight = 1.0;
            let copies = whole + (if rng.random::<f32>() < frac { 1 } else { 0 });
            for _ in 0..copies {
                if self.buf.len() >= self.capacity {
                    self.buf.pop_front();
                }
                self.buf.push_back(record.clone());
                count += 1;
            }
        }
        count
    }

    /// 均匀随机采样：生成 `0..len` 的随机排列。
    ///
    /// Policy Surprise 加权已在写入时通过复制实现，采样只需 shuffle。
    pub fn sample(&self, rng: &mut impl rand::Rng) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.buf.len()).collect();
        indices.shuffle(rng);
        indices
    }
}

#[cfg(test)]
mod tests {
    use crate::game::board::Board;

    use super::*;

    fn dummy_record(value: f32, weight: f32) -> PlayRecord {
        let npos = Board::DEFAULT_BOARD_SIZE * Board::DEFAULT_BOARD_SIZE;
        PlayRecord {
            state: vec![0i32; npos],
            policy: vec![0.0f32; npos],
            value,
            sample_weight: weight,
            ..Default::default()
        }
    }

    #[test]
    fn capacity_enforced() {
        let mut buf = RolloutBuffer::new(3);
        buf.extend((0..5).map(|i| dummy_record(i as f32, 1.0)));
        assert_eq!(buf.len(), 3);
        assert!((buf.get(0).value - 2.0).abs() < 1e-6);
    }

    #[test]
    fn uniform_sampling_covers_all() {
        let mut buf = RolloutBuffer::new(10);
        buf.extend((0..10).map(|i| dummy_record(i as f32, 1.0)));
        let mut rng = rand::rng();
        let sample = buf.sample(&mut rng);
        let mut sorted = sample.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn high_weight_duplication() {
        let mut buf = RolloutBuffer::new(10);
        // 权重 3.0 + 1.0 + 1.0 = 预期产生约 5 份
        buf.extend(vec![
            dummy_record(1.0, 3.0),
            dummy_record(2.0, 1.0),
            dummy_record(3.0, 1.0),
        ]);
        assert!(buf.len() >= 4);
        // 权重 3.0 的记录应该出现更多次
        let count_high = buf
            .buf
            .iter()
            .filter(|r| (r.value - 1.0).abs() < 1e-6)
            .count();
        assert!(
            count_high >= 3,
            "weight 3.0 got {} copies, expected >= 3",
            count_high
        );
    }
}
