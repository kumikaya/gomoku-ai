//! 五子棋 AI - 基于 AlphaZero 算法
//!
//! 模块结构：
//! - `game`:     棋盘逻辑、状态编码、胜负判定
//! - `network`:  神经网络模型（f16 半精度）
//! - `mcts`:     蒙特卡洛树搜索
//! - `selfplay`: 自对弈数据生成
//! - `training`: 训练循环（含 Loss Scaling）
//!   - `loss_scaler`: 混合精度 Loss Scaling
//!   - `trainer`:     训练器
//! - `eval`:     神经网络棋力评估（对抗对弈 + Elo 追踪）

pub mod eval;
pub mod game;
pub mod inference;
pub mod mcts;
pub mod network;
pub mod selfplay;
pub mod training;
