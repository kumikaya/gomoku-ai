//! 学习率调度器
//!
//! Cosine decay：从 `1.0` 衰减到 `final_ratio`。
//! 最终 lr = factor * base_lr

pub struct LrSchedule {
    num_iterations: usize,
    final_ratio: f64,
}

impl LrSchedule {
    /// 创建调度器。
    ///
    /// - `num_iterations`：总迭代数
    /// - `final_ratio`：最终 lr 比例，0.0 表示衰减到 0。
    pub fn new(num_iterations: usize, final_ratio: f32) -> Self {
        Self {
            num_iterations,
            final_ratio: final_ratio as f64,
        }
    }

    /// 获取第 `iteration` 次迭代的 lr 衰减因子（∈ [final_ratio, 1.0]）。
    #[inline]
    pub fn factor(&self, iteration: usize) -> f64 {
        let progress = iteration as f64 / self.num_iterations as f64;
        let cos = (std::f64::consts::PI * progress).cos();
        self.final_ratio + (1.0 - self.final_ratio) * 0.5 * (1.0 + cos)
    }

    /// 获取实际 lr = factor * base_lr。
    #[inline]
    pub fn lr(&self, iteration: usize, base_lr: f64) -> f64 {
        self.factor(iteration) * base_lr
    }
}
