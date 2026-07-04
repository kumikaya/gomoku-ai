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
    /// 训练权重（KataGo 风格）：
    ///   - 1.0 = 满模拟的 high-quality 样本
    ///   - 0.0 = 最少模拟的 low-quality 样本（仅训练价值头，不训练策略头）
    ///   中间值按模拟数比例插值
    pub target_weight: f32,
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
    /// KataGo Playout Cap：启用后每步在 `[min, max]` 内均匀随机模拟次数。
    /// `min = num_simulations * playout_cap_min_ratio`。
    pub playout_cap_enabled: bool,
    /// 下界比例（相对于 `num_simulations`），例如 0.25 表示最少 25%。
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
pub fn self_play<E: Evaluator>(
    evaluator: &E,
    config: &SelfPlayConfig,
    rng: &mut impl RngExt,
) -> SelfPlayGame {
    let mut board = Board::new();
    let mut records = Vec::new();
    let mut mcts = MCTS::new();

    loop {
        // ── Playout Cap: 每步均匀随机模拟次数 + target_weight ──
        let (sims, target_weight) = if config.playout_cap_enabled {
            let min_sims =
                (config.num_simulations as f32 * config.playout_cap_min_ratio).max(1.0) as usize;
            let max_sims = config.num_simulations;
            let range = max_sims.saturating_sub(min_sims);
            let sims = if range > 0 {
                min_sims + rng.random_range(0..=range)
            } else {
                min_sims
            };
            let weight = if max_sims > min_sims {
                (sims - min_sims) as f32 / (max_sims - min_sims) as f32
            } else {
                1.0
            };
            (sims, weight.max(0.1))
        } else {
            (config.num_simulations, 1.0f32)
        };

        let mut search_config = GumbelConfig::pure_gumbel(sims);
        search_config.select_temperature = config.select_temperature;

        let result = mcts.search(&mut board, evaluator, &search_config, rng);

        records.push(PlayRecord {
            state: board.encode_state(),
            policy: result.policy,
            value: result.root_value,
            target_weight,
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
