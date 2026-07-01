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
use rand::RngExt;

#[derive(Clone, Debug)]
pub struct PlayRecord {
    pub state: Vec<i32>,
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
    /// 动作选择 softmax 温度，minizero 风格随训练进度衰减：
    ///   0%–50% → 1.0  |  50%–75% → 0.5  |  75%–100% → 0.25
    pub select_temperature: f32,
    /// KataGo Playout Cap：启用在 `[min, max]` 内随机化每次搜索的模拟次数。
    /// 让网络同时从不同质量的数据中学习，提升泛化能力并节省计算量。
    pub playout_cap_enabled: bool,
    /// 随机化下界比例（相对于 `num_simulations`），例如 0.25 表示最少 25%。
    pub playout_cap_min_ratio: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_simulations: 32,
            select_temperature: 1.0,
            playout_cap_enabled: false,
            playout_cap_min_ratio: 0.25,
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
    let mut mcts = MCTS::new();
    let mut rng = rand::rng();

    loop {
        // ── Playout Cap: 每步随机化模拟次数 ──
        let sims = if config.playout_cap_enabled {
            let min =
                (config.num_simulations as f32 * config.playout_cap_min_ratio).max(1.0) as usize;
            let range = config.num_simulations.saturating_sub(min);
            if range > 0 {
                min + rng.random_range(0..=range)
            } else {
                min
            }
        } else {
            config.num_simulations
        };

        let mut search_config = GumbelConfig::pure_gumbel(sims);
        search_config.select_temperature = config.select_temperature;

        // 每步新建 MCTS
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
