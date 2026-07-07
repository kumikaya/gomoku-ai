//! 自对弈数据生成
//!
//! 使用 MCTS (Gumbel Zero) 指导双方走棋，生成 (状态, 策略, 价值) 训练样本。

use crate::game::board::{Board, Color};
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
    /// 走这步棋的玩家（仅用于自对弈结束后用游戏结果修正 value，不参与训练）。
    pub player: Color,
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

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub num_simulations: usize,
    pub select_temperature: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_simulations: 64,
            select_temperature: 1.0,
        }
    }
}

/// 运行一局自对弈。每一步都产生训练样本。
pub async fn self_play<E: Evaluator>(
    evaluator: &E,
    config: &SelfPlayConfig,
    rng: &mut impl RngExt,
) -> SelfPlayGame {
    let mut board = Board::new();
    let mut records = Vec::new();
    let mut mcts = MCTS::new();

    loop {
        let mut search_config = GumbelConfig::pure_gumbel(config.num_simulations);
        search_config.select_temperature = config.select_temperature;

        let result = mcts.search(&board, evaluator, &search_config, rng).await;

        let kl = compute_kl(&result.root_nn_prior, &result.policy);
        records.push(PlayRecord {
            state: board.encode_state(),
            policy: result.policy,
            // 先写入 MCTS root value 作为占位，游戏结束后用最终结果修正
            value: result.root_value,
            sample_weight: kl,
            player: board.current_player,
        });

        board.play_idx(result.best_move);

        if board.game_over {
            break;
        }
    }

    // ── 对齐 minizero：用游戏最终结果修正 value 标签 ──
    // 从当前玩家视角：赢=+1.0, 输=-1.0, 平=0.0
    for record in &mut records {
        record.value = match board.winner {
            Some(w) if w == record.player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
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
    let npos = nn_prior.len().min(mcts_policy.len());
    for i in 0..npos {
        let p = nn_prior[i].max(epsilon);
        let q = mcts_policy[i].max(epsilon);
        if p > epsilon {
            kl += p * (p / q).ln();
        }
    }
    kl
}
