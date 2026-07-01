//! 训练循环
//!
//! 自对弈 → 收集数据 → 采样训练时随机 D4 增强 →
//! （KL 早停 + 自适应学习率） → 更新参数 → 重复。
//!
//! 精度由 `Device::configure` 全局设定（BF16/F32），前向传播不感知精度。

pub mod buffer;
pub mod lr_schedule;
pub mod trainer;
