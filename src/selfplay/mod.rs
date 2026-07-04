//! 自对弈数据生成
//!
//! 使用 MCTS (Gumbel Zero) 指导双方走棋，生成 (状态, 策略, 价值) 训练样本。
//!
//! Gumbel Zero 的 Completed Q 策略已自带 softmax 分布，
//! 不需要 PUCT 风格的 temperature 退火。Gumbel 噪声提供了
//! 足够的探索多样性。

use crate::game::board::{Board, Color, NUM_POSITIONS};
use crate::inference::Evaluator;
use crate::mcts::node::{GumbelConfig, MCTS};
use rand::RngExt;

#[derive(Clone, Debug, Default)]
pub struct PlayRecord {
    pub state: Vec<i32>,
    pub policy: Vec<f32>,
    pub value: f32,
    /// 训练样本权重（KataGo Playout Cap + Policy Surprise）。
    /// 完整搜索 + 惊奇局面 → 高权重；快速搜索 → 低权重。
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
    /// 动作选择 softmax 温度
    pub select_temperature: f32,
    /// Playout Cap Randomization (KataGo): 完整搜索的概率 (0.0-1.0)
    /// 其余步用 `num_simulations * fast_sim_factor` 做快速搜索。
    pub full_search_prob: f32,
    /// 快速搜索的模拟数比例（相对于 num_simulations），默认 0.25
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

/// 运行一局自对弈，生成 (状态, 策略, 价值) 训练样本。
///
/// Playout Cap Randomization (KataGo): 每步随机决定做"完整搜索"还是"快速搜索"。
/// - 完整搜索：num_simulations 全量模拟 → 高质量 policy target
/// - 快速搜索：fast_sim_factor * num_simulations → 低成本覆盖更多局面
/// 快速搜索的样本标记更低的 surprise_weight，避免低质量目标主导训练。
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

        let result = mcts.search(&mut board, evaluator, &search_config, rng);

        let weight = compute_sample_weight(&result.root_nn_prior, &result.policy, is_full_search);

        records.push(PlayRecord {
            state: board.encode_state(),
            policy: result.policy,
            value: result.root_value,
            sample_weight: weight,
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

/// 计算训练样本权重 (KataGo Playout Cap + Policy Surprise Weighting)。
///
/// - `nn_prior`: 根节点 NN 干净先验（不含 noise/temperature）
/// - `mcts_policy`: MCTS 搜索后验（Completed Q policy）
/// - `is_full_search`: 是否为完整搜索（全量模拟）
///
/// 返回 KL(P || Q) × 搜索质量折扣。
fn compute_sample_weight(nn_prior: &[f32], mcts_policy: &[f32], is_full_search: bool) -> f32 {
    // Policy Surprise: KL(P || Q)
    let epsilon = 1e-12f32;
    let mut kl = 0.0f32;
    for i in 0..NUM_POSITIONS {
        let p = nn_prior[i].max(epsilon);
        let q = mcts_policy[i].max(epsilon);
        if p > epsilon {
            kl += p * (p / q).ln();
        }
    }
    // Playout Cap 折扣：快速搜索样本降权，避免低质量目标主导训练
    let weight_scale: f32 = if is_full_search { 1.0 } else { 0.2 };
    kl * weight_scale
}
