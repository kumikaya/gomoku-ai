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

#[derive(Clone, Debug)]
pub struct PlayRecord {
    pub state: Vec<i32>,
    pub policy: Vec<f32>,
    pub value: f32,
    /// Policy Surprise Weighting: KL(nn_prior || mcts_posterior) 归一化后的权重。
    /// 0 表示无 surprise 信息（均匀权重），>0 表示惊奇程度。
    pub surprise_weight: f32,
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
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_simulations: 32,
            select_temperature: 1.0,
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
        let sims = config.num_simulations;

        let mut search_config = GumbelConfig::pure_gumbel(sims);
        search_config.select_temperature = config.select_temperature;

        let result = mcts.search(&mut board, evaluator, &search_config, rng);

        let surprise = compute_surprise(&result.root_nn_prior, &result.policy);
        records.push(PlayRecord {
            state: board.encode_state(),
            policy: result.policy,
            value: result.root_value,
            surprise_weight: surprise,
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

/// Policy Surprise Weighting (KataGo):
/// 计算 NN 干净先验 P 与 MCTS 搜索后验 Q 之间的 KL 散度。
///
/// KL(P || Q) = Σ P(i) * ln(P(i) / Q(i))
///
/// 值越大表示 NN 越"惊到"于搜索结果。
/// 返回原始 KL 值（调用方在缓冲区中做归一化）。
fn compute_surprise(nn_prior: &[f32], mcts_policy: &[f32]) -> f32 {
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
