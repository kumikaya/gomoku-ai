//! 训练循环
//!
//! 自对弈 → 收集数据 → 采样训练时随机 D4 增强 →
//! （KL 早停 + 自适应学习率） → 更新参数 → 重复。
//!
//! ## 模块结构
//!
//! - `loss_scaler`: 混合精度训练 Loss Scaling
//! - `trainer`: 训练器（配置 + 训练循环）

pub mod loss_scaler;
pub mod trainer;
