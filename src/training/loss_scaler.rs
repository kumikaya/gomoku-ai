//! 混合精度训练的 Loss Scaler
//!
//! 前向传播使用 f16 半精度时，小梯度可能在反向传播中下溢为零。
//! Loss Scaler 将 loss 放大再反向传播，然后对梯度做 unscaling 恢复真实值。
//!
//! ## 机制
//!
//! ```text
//! loss × scale → backward(f16 grad) → grad ÷ scale → optim.step
//!                                          ↑
//!                                   检测 NaN → 缩减 scale, skip
//! ```
//!
//! - 连续 `grow_interval` 步无溢出后，scale 翻倍（上限 2^24）
//! - 检测到梯度中有 NaN，scale 减半并跳过本次更新

use burn::module::{AutodiffModule, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::Tensor;

/// Loss scaling 配置
#[derive(Debug, Clone)]
pub struct LossScaleConfig {
    /// 初始缩放因子
    pub init_scale: f32,
    /// 连续多少步无溢出后增大 scale
    pub grow_interval: u32,
    /// scale 增长因子（通常 2.0）
    pub growth_factor: f32,
    /// 缩减因子（通常 0.5）
    pub backoff_factor: f32,
    /// 最大缩放因子（f16 安全上限约 2^24）
    pub max_scale: f32,
}

impl Default for LossScaleConfig {
    fn default() -> Self {
        Self {
            init_scale: 128.0,
            grow_interval: 2000,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            max_scale: 16_777_216.0, // 2^24
        }
    }
}

/// Loss Scaler 状态机
pub struct LossScaler {
    scale: f32,
    good_steps: u32,
    config: LossScaleConfig,
}

impl LossScaler {
    pub fn new(config: LossScaleConfig) -> Self {
        Self {
            scale: config.init_scale.max(1.0),
            good_steps: 0,
            config,
        }
    }

    /// 当前 scale 值
    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// 对梯度做 unscaling（÷ scale），并检测溢出
    ///
    /// 返回 `Ok(())` 表示梯度可用，`Err(())` 表示检测到溢出应跳过本次更新。
    ///
    /// 使用 `ModuleVisitor` 遍历 param tree — 每个 param 自带正确的 `const D`，
    /// 避免 `GradientsParams::get/remove` 在 rank 不匹配时 panic。
    pub fn unscale_and_check<M: AutodiffModule>(
        &mut self,
        grads_params: &mut GradientsParams,
        model: &M,
    ) -> Result<(), ()> {
        // 第一步：遍历检查 NaN
        let mut overflow_visitor = OverflowCheckVisitor {
            has_overflow: false,
            grads: grads_params,
            _phantom: std::marker::PhantomData::<M>,
        };
        model.visit(&mut overflow_visitor);

        if overflow_visitor.has_overflow {
            self.scale = (self.scale * self.config.backoff_factor).max(1.0);
            self.good_steps = 0;
            println!(
                "    [AMP] gradient overflow, scale → {:.0}, skip step",
                self.scale
            );
            return Err(());
        }

        // 第二步：unscale（remove ÷ scale → register）
        let inv = 1.0 / self.scale;
        let mut unscale_visitor = UnscaleVisitor {
            inv,
            grads: grads_params,
            _phantom: std::marker::PhantomData::<M>,
        };
        model.visit(&mut unscale_visitor);

        // 动态增长 scale
        self.good_steps += 1;
        if self.good_steps >= self.config.grow_interval {
            self.scale = (self.scale * self.config.growth_factor).min(self.config.max_scale);
            self.good_steps = 0;
        }

        Ok(())
    }
}

// ── ModuleVisitor：检查梯度 NaN ──

struct OverflowCheckVisitor<'a, M: AutodiffModule> {
    has_overflow: bool,
    grads: &'a GradientsParams,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: AutodiffModule> ModuleVisitor for OverflowCheckVisitor<'_, M> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if self.has_overflow {
            return;
        }
        // get::<D> 的 D 与 param 的 rank 完全匹配，不会 panic
        if let Some(g) = self.grads.get::<D>(param.id) {
            if g.is_nan().any().into_scalar() {
                self.has_overflow = true;
            }
        }
    }
}

// ── ModuleVisitor：梯度 ÷ scale ──

struct UnscaleVisitor<'a, M: AutodiffModule> {
    inv: f32,
    grads: &'a mut GradientsParams,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: AutodiffModule> ModuleVisitor for UnscaleVisitor<'_, M> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        // remove::<D> 的 D 与 param 的 rank 完全匹配，不会 panic
        if let Some(g) = self.grads.remove::<D>(param.id) {
            self.grads.register::<D>(param.id, g * self.inv);
        }
    }
}
