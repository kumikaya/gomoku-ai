//! 自对弈数据生成
//!
//! 使用 MCTS (Gumbel Zero) 指导双方走棋，生成 (状态, 策略, 价值) 训练样本。
//!
//! Playout Cap Randomization (KataGo): 每步随机决定做"完整搜索"还是"快速搜索"。
//! - 完整搜索: num_simulations 全量模拟 → 产生训练样本 + 选择走法
//! - 快速搜索: fast_sim_factor * num_simulations → 仅用于选择走法推进棋局，不产生样本
//!
//! 对齐 KataGo: cheapSearchTargetWeight=0 → 快速搜索的样本不写入训练数据。

use crate::game::board::{Board, Color, NUM_POSITIONS};
use crate::inference::Evaluator;
use crate::mcts::node::{GumbelConfig, MCTS};
use rand::RngExt;

#[derive(Clone, Debug, Default)]
pub struct PlayRecord {
    pub state: Vec<i32>,
    pub policy: Vec<f32>,
    pub value: f32,
    /// 训练样本权重 (Policy Surprise = KL(nn_prior || mcts_posterior)).
    /// 仅完整搜索产生样本，权重即 KL 值。写入时按权重复制多份 (KataGo frequency weighting).
    pub sample_weight: f32,
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
    pub select_temperature: f32,
    /// 完整搜索概率 (0.0-1.0)，其余步做快速搜索仅用于推进棋局不产生样本。
    pub full_search_prob: f32,
    /// 快速搜索的模拟数比例 (相对于 num_simulations).
    pub fast_sim_factor: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_simulations: 32,
            select_temperature: 1.0,
            full_search_prob: 0.4,
            fast_sim_factor: 0.25,
        }
    }
}

/// 运行一局自对弈。
///
/// 完整搜索 → 产生训练样本；快速搜索 → 仅推进棋局。
pub fn self_play<E: Evaluator>(
    evaluator: &E,
    config: &SelfPlayConfig,
    rng: &mut impl RngExt,
) -> SelfPlayGame {
    let mut board = Board::new();
    let mut records = Vec::new();
    let mut mcts = MCTS::new();

    loop {
        let is_full_search = rng.random::<f32>() < config.full_search_prob;
        let sims = if is_full_search {
            config.num_simulations
        } else {
            (config.num_simulations as f32 * config.fast_sim_factor).max(1.0) as usize
        };

        let mut search_config = GumbelConfig::pure_gumbel(sims);
        search_config.select_temperature = config.select_temperature;

        let result = mcts.search(&board, evaluator, &search_config, rng);

        // 仅完整搜索产生训练样本 (对齐 KataGo cheapSearchTargetWeight=0)
        if is_full_search {
            let kl = compute_kl(&result.root_nn_prior, &result.policy);
            records.push(PlayRecord {
                state: board.encode_state(),
                policy: result.policy,
                value: result.root_value,
                sample_weight: kl,
            });
        }

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

/// Policy Surprise: NN 先验 P 与 MCTS 后验 Q 的 KL 散度。
fn compute_kl(nn_prior: &[f32], mcts_policy: &[f32]) -> f32 {
    let epsilon = 1e-12f32;
    let mut kl = 0.0f32;
    for i in 0..NUM_POSITIONS {
        let p = nn_prior[i].max(epsilon);
        let q = mcts_policy[i].max(epsilon);
        if p > epsilon {
            kl += p * (p / q).ln();
        }
    }
    kl
}
