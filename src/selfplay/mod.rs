//! 自对弈数据生成
//!
//! 使用 MCTS (Gumbel Zero) 指导双方走棋，生成 (状态, 策略, 价值) 训练样本。
//!
//! Gumbel Zero 的 Completed Q 策略已自带 softmax 分布，
//! 不需要 PUCT 风格的 temperature 退火。Gumbel 噪声提供了
//! 足够的探索多样性。

use crate::game::board::{Board, Color};
use crate::inference::Evaluator;
use crate::mcts::node::{GumbelConfig, MCTS};

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
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_simulations: 32,
        }
    }
}

/// 运行一局自对弈，生成 (状态, 策略, 价值) 训练样本。
///
/// `evaluator` 可以是 `InferenceServer` 或其他 `Evaluator` 实现。
/// 多局并发时可共享同一个 evaluator，GPU 线程自动跨对局攒批。
pub fn self_play<E: Evaluator>(evaluator: &E, config: &SelfPlayConfig) -> SelfPlayGame {
    let mut board = Board::new();
    let mut records = Vec::new();

    let search_config = GumbelConfig::pure_gumbel(config.num_simulations);

    loop {
        // 每步新建 MCTS
        let mut mcts = MCTS::new();
        let result = mcts.search(&mut board, evaluator, &search_config);

        // V 标签对齐 minizero `getMCTSValue()`：
        // 使用 MCTS visit-count 加权 Q 作为 bootstrap target，
        // 而非对局结果。方差更低，收敛更快。
        records.push(PlayRecord {
            state: board.encode_state(),
            policy: result.policy,
            value: result.root_value,
        });

        board.play_idx(result.best_move);

        if board.game_over {
            break;
        }
    }

    SelfPlayGame {
        winner: board.winner,
        records,
    }
}
