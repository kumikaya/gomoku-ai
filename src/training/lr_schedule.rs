//! 学习率调度器（KataGo 风格）
//!
//! 支持 warmup + cosine decay：
//! - 前 `warmup_ratio` 比例迭代：线性从 `final_ratio` warmup 到 `1.0`。
//! - 剩余迭代：cosine 衰减从 `1.0` 到 `final_ratio`。
//!
//! 最终 lr = factor * base_lr

/// 预计算整个训练过程的 lr 衰减因子，O(1) 查询。
pub struct LrSchedule {
    /// 预计算的因子数组，索引 = iteration
    factors: Vec<f64>,
}

impl LrSchedule {
    /// 创建调度器。
    ///
    /// - `num_iterations`：总迭代数
    /// - `warmup_ratio`：warmup 占比，0.05 表示前 5% 迭代线性 warmup。设为 0 跳过 warmup。
    /// - `final_ratio`：最终 lr 比例，0.0 表示衰减到 0。
    pub fn new(num_iterations: usize, warmup_ratio: f32, final_ratio: f32) -> Self {
        let factors: Vec<f64> = (0..num_iterations)
            .map(|it| {
                let progress = it as f32 / (num_iterations - 1).max(1) as f32;
                if warmup_ratio <= 0.0 || (warmup_ratio < 1.0 && progress >= warmup_ratio) {
                    // ── cosine decay 阶段 ──
                    let decay_progress = if warmup_ratio >= 1.0 {
                        1.0
                    } else {
                        ((progress - warmup_ratio) / (1.0 - warmup_ratio)).min(1.0)
                    };
                    let cos = (std::f32::consts::PI * decay_progress).cos();
                    (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + cos)) as f64
                } else {
                    // ── warmup 阶段 ──
                    let w = progress / warmup_ratio;
                    (final_ratio + (1.0 - final_ratio) * w) as f64
                }
            })
            .collect();
        Self { factors }
    }

    /// 获取第 `iteration` 次迭代的 lr 衰减因子（∈ [final_ratio, 1.0]）。
    #[inline]
    pub fn factor(&self, iteration: usize) -> f64 {
        self.factors[iteration.min(self.factors.len() - 1)]
    }

    /// 获取实际 lr = factor * base_lr。
    #[inline]
    pub fn lr(&self, iteration: usize, base_lr: f64) -> f64 {
        self.factor(iteration) * base_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_then_cosine() {
        let sched = LrSchedule::new(100, 0.05, 0.0);
        assert!(sched.factor(0) < 0.1); // warmup 开始，接近 0
        assert!(sched.factor(4) > 0.8); // warmup 结束，接近 1.0
        assert!(sched.factor(50) > 0.4 && sched.factor(50) < 0.6); // cosine 中间
        assert!(sched.factor(99) < 0.01); // 接近 0
    }

    #[test]
    fn no_warmup() {
        let sched = LrSchedule::new(100, 0.0, 0.0);
        assert!((sched.factor(0) - 1.0).abs() < 1e-6); // 起始=1.0
        assert!((sched.factor(99)).abs() < 0.01); // 衰减到接近 0
    }

    #[test]
    fn all_warmup() {
        let sched = LrSchedule::new(100, 1.0, 0.0);
        assert!((sched.factor(0) - 0.0).abs() < 1e-6);
        assert!((sched.factor(99) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn non_zero_final() {
        let sched = LrSchedule::new(100, 0.1, 0.1);
        assert!(sched.factor(99) > 0.09 && sched.factor(99) < 0.11);
    }
}
