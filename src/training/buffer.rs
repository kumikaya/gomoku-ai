//! 经验回放缓冲区
//!
//! `RolloutBuffer<T>` 封装了固定容量的 FIFO 缓冲区，支持：
//! - 批量写入（自动淘汰旧数据）
//! - 加权随机采样

use rand::distr::{Distribution, weighted::WeightedIndex};
use std::collections::VecDeque;

/// 经验回放缓冲区
pub struct RolloutBuffer<T> {
    buf: VecDeque<T>,
    capacity: usize,
}

impl<T> RolloutBuffer<T> {
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
    pub fn get(&self, index: usize) -> &T {
        &self.buf[index]
    }

    /// 遍历所有样本的不可变引用。
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buf.iter()
    }

    /// 批量追加记录，超出容量时自动淘汰最旧的数据。
    ///
    /// 返回实际新增的样本数。后续可通过
    /// [`sample_batch`] 进行带权采样。
    pub fn extend(&mut self, records: impl IntoIterator<Item = T>) -> usize {
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

    /// 带权随机采样：按 `weights` 有放回地采样 `batch_size` 个索引。
    pub fn sample_batch(
        &self,
        batch_size: usize,
        weights: &[f32],
        rng: &mut impl rand::Rng,
    ) -> impl Iterator<Item = &T> {
        let dist = WeightedIndex::new(weights).expect("all weights are zero");
        (0..batch_size).map(move |_| {
            let idx = dist.sample(rng);
            self.get(idx)
        })
    }
}
