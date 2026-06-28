//! 五子棋 AI - 基于 AlphaZero 算法
//!
//! 模块结构：
//! - `game`:   棋盘逻辑、胜负判定
//! - `network`: 神经网络模型
//! - `mcts`:   蒙特卡洛树搜索
//! - `selfplay`: 自对弈数据生成
//! - `training`: 训练循环

pub mod game;
pub mod mcts;
pub mod network;
pub mod selfplay;
pub mod training;
