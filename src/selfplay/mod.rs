//! 自对弈数据生成
//!
//! 使用 MCTS 指导双方走棋，生成 (状态, 策略, 价值) 训练样本。

use crate::game::board::{Board, Color};
use crate::mcts::node::MCTS;
use crate::network::residual::GomokuNetwork;
use burn::tensor::Device;

#[derive(Clone, Debug)]
pub struct PlayRecord {
    pub state: Vec<f32>,
    pub policy: Vec<f32>,
    pub value: f32,
}

pub struct SelfPlayGame {
    pub records: Vec<PlayRecord>,
    pub winner: Option<Color>,
}

impl SelfPlayGame {
    pub fn num_steps(&self) -> usize {
        self.records.len()
    }
}

pub struct SelfPlayConfig {
    pub num_simulations: usize,
    pub temperature: f32,
    pub temperature_decay_steps: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_simulations: 400,
            temperature: 1.0,
            temperature_decay_steps: 30,
        }
    }
}

/// 运行一局自对弈，生成 (状态, 策略, 价值) 训练样本。
///
/// ## 自对弈流程
///
/// 1. 初始化空白棋盘和 MCTS 搜索器
/// 2. 每一步：
///    - 确定温度参数（前 N 步用高温探索，之后退火到确定性选择）
///    - 执行 MCTS 搜索，得到策略分布和最佳走法
///    - 记录当前状态和 MCTS 策略（价值暂时填 0，对局结束后回填）
///    - 执行选定的走法
/// 3. 对局结束后，根据胜负结果回填每个状态的价值标签
///
/// ## 温度退火策略
///
/// 前 `temperature_decay_steps` 步使用 temperature=1.0 做概率采样，
/// 之后用极低温度（≈1e-6）做确定性选择。这是 AlphaZero 的标准做法：
/// 前期鼓励探索产生多样化数据，后期确保高质量终局。
pub fn self_play(network: &GomokuNetwork, config: &SelfPlayConfig, device: Device) -> SelfPlayGame {
    let mut board = Board::new();
    let mut mcts = MCTS::new();
    let mut records = Vec::new();

    loop {
        let t = if board.step_count < config.temperature_decay_steps {
            config.temperature
        } else {
            1e-6
        };

        let result = mcts.search(&mut board, network, &device, config.num_simulations, t);

        records.push(PlayRecord {
            state: board.encode_state(),
            policy: result.policy,
            value: 0.0,
        });

        board.play_idx(result.best_move);

        if board.game_over {
            break;
        }
    }

    fill_values(&mut records, board.winner);

    SelfPlayGame {
        winner: board.winner,
        records,
    }
}

/// 根据对局结果回填每个状态的价值标签。
///
/// 价值标签规则：
/// - 如果该步的玩家最终获胜 → +1.0
/// - 如果该步的玩家最终失败 → -1.0
/// - 平局 → 0.0
///
/// 偶数步（index 0, 2, 4...）为黑方，奇数步为白方。
fn fill_values(records: &mut [PlayRecord], winner: Option<Color>) {
    for (i, record) in records.iter_mut().enumerate() {
        let player = if i % 2 == 0 {
            Color::Black
        } else {
            Color::White
        };
        record.value = match winner {
            Some(w) if w == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
    }
}
