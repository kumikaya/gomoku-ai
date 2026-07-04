//! 神经网络棋力评估
//!
//! 通过对抗对弈评估模型实力：
//! - `MatchRunner`: 让两个模型对弈 N 局，统计胜率
//! - `EloTracker`: Elo 评分追踪（1500 基准）
//! - 在训练循环中周期性评估，胜率超过阈值则晋升模型

use crate::game::board::{Board, Color, NUM_POSITIONS};
use crate::inference::InferenceServer;
use crate::mcts::node::{GumbelConfig, MCTS};
use crate::network::transformer::GomokuNetwork;

use burn::module::{AutodiffModule, Module};
use burn::tensor::Device;
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

// ============================================================
//  Elo 评分系统
// ============================================================

/// Elo 评分追踪器。
///
/// 初始 Elo 1500，每轮评估后根据实际 vs 预期胜率更新。
/// 基线模型视为 Elo 固定（不更新），仅计算当前模型相对 baseline 的评分变化。
pub struct EloTracker {
    pub current_elo: f64,
    pub history: Vec<(usize, f64)>,
    k_factor: f64,
}

impl EloTracker {
    pub fn new() -> Self {
        Self {
            current_elo: 1500.0,
            history: Vec::new(),
            k_factor: 32.0,
        }
    }

    /// 根据对 baseline 的胜率更新 Elo。
    ///
    /// `iteration`: 当前训练轮次
    /// `win_rate`: 当前模型 vs baseline 的胜率 (0.0~1.0，平局计 0.5)
    ///
    /// 返回新的 Elo 分数。
    pub fn update(&mut self, iteration: usize, win_rate: f64) -> f64 {
        let baseline_elo = 1500.0;
        let clamped = win_rate.clamp(0.01, 0.99);
        let expected = 1.0 / (1.0 + 10f64.powf((baseline_elo - self.current_elo) / 400.0));
        self.current_elo += self.k_factor * (clamped - expected);
        self.history.push((iteration, self.current_elo));
        self.current_elo
    }

    pub fn print_history(&self) {
        println!("\n  Elo History:");
        for (iter, elo) in &self.history {
            let delta = elo - 1500.0;
            let sign = if delta >= 0.0 { "+" } else { "" };
            println!("    iter {:>4}: {:.1} ({}{:.1})", iter, elo, sign, delta);
        }
    }
}

// ============================================================
//  对弈结果
// ============================================================

/// 一局游戏的结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameOutcome {
    BlackWins,
    WhiteWins,
    Draw,
}

/// 多局对抗对弈的汇总结果
#[derive(Debug, Clone, Default)]
pub struct MatchResult {
    pub total_games: usize,
    /// 当前模型（挑战者）的胜场
    pub wins_current: usize,
    /// 基线模型（擂主）的胜场
    pub wins_baseline: usize,
    /// 平局数
    pub draws: usize,
    /// 全局黑方胜场（用于检测先手优势）
    pub black_wins: usize,
    /// 全局白方胜场
    pub white_wins: usize,
}

impl MatchResult {
    /// 当前模型的胜率（平局计 0.5 胜）
    pub fn win_rate_current(&self) -> f64 {
        if self.total_games == 0 {
            return 0.5;
        }
        (self.wins_current as f64 + self.draws as f64 * 0.5) / self.total_games as f64
    }

    pub fn print(&self) {
        let wr = self.win_rate_current() * 100.0;
        println!(
            "  Current vs Baseline: W:{}-L:{}-D:{} ({:.1}% win rate) | Black wins:{} White wins:{}",
            self.wins_current, self.wins_baseline, self.draws, wr, self.black_wins, self.white_wins,
        );
    }
}

// ============================================================
//  评估配置
// ============================================================

pub struct EvalConfig {
    /// 评估对弈局数
    pub num_games: usize,
    /// 每步 MCTS 模拟次数（评估时建议比训练时高，如 64~128）
    pub num_simulations: usize,
    /// 晋升阈值：当前模型胜率超过此值则替换 baseline
    /// 例如 0.55 表示胜率 > 55% 时晋升
    pub promotion_threshold: f64,
    /// 每隔多少轮评估一次（也用于决定是否晋升）
    pub eval_every: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            num_games: 100,
            num_simulations: 64,
            promotion_threshold: 0.55,
            eval_every: 5,
        }
    }
}

// ============================================================
//  MatchRunner: 两模型对抗对弈
// ============================================================

/// 对抗对弈引擎。
///
/// 让当前模型与基线模型对弈 N 局，各自交替先后手，
/// 使用 Gumbel Zero MCTS，低温度接近确定性下棋。
pub struct MatchRunner {
    config: EvalConfig,
}

impl MatchRunner {
    pub fn new(config: EvalConfig) -> Self {
        Self { config }
    }

    /// 运行对抗对弈。
    ///
    /// `current`: 当前训练中的模型（挑战者）
    /// `baseline`: 基线模型（擂主，如上次晋升的最佳模型）
    /// `device`: 推理设备
    ///
    /// 返回 `MatchResult`，其中 `wins_current` 是当前模型的胜场。
    pub fn run_match(
        &self,
        current: GomokuNetwork,
        baseline: GomokuNetwork,
        device: Device,
        rng_seed: u64,
    ) -> MatchResult {
        let num_games = self.config.num_games;
        let half = num_games / 2;

        // 为两个模型各创建一个 InferenceServer
        let server_current = Arc::new(InferenceServer::new(current, device.clone()));
        let server_baseline = Arc::new(InferenceServer::new(baseline, device.clone()));

        let pb = ProgressBar::new(num_games as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Eval match: {bar:40.yellow/blue} {pos}/{len} ({eta})")
                .unwrap(),
        );

        // 前半：当前模型执黑（先手），基线执白
        let outcomes_current_black: Vec<GameOutcome> = (0..half)
            .into_par_iter()
            .map(|game_i| {
                let seed = rng_seed.wrapping_add(game_i as u64);
                let result = play_eval_game(
                    server_current.as_ref(),
                    server_baseline.as_ref(),
                    self.config.num_simulations,
                    seed,
                );
                pb.inc(1);
                result
            })
            .collect();

        // 后半：基线执黑，当前模型执白
        let outcomes_current_white: Vec<GameOutcome> = (0..num_games - half)
            .into_par_iter()
            .map(|game_i| {
                let seed = rng_seed.wrapping_add((half + game_i) as u64);
                let result = play_eval_game(
                    server_baseline.as_ref(),
                    server_current.as_ref(),
                    self.config.num_simulations,
                    seed,
                );
                pb.inc(1);
                result
            })
            .collect();

        pb.finish_and_clear();

        let mut result = MatchResult::default();
        result.total_games = num_games;

        // 当前模型执黑 → BlackWins = 当前胜
        for outcome in &outcomes_current_black {
            match outcome {
                GameOutcome::BlackWins => {
                    result.wins_current += 1;
                    result.black_wins += 1;
                }
                GameOutcome::WhiteWins => {
                    result.wins_baseline += 1;
                    result.white_wins += 1;
                }
                GameOutcome::Draw => {
                    result.draws += 1;
                }
            }
        }

        // 当前模型执白 → WhiteWins = 当前胜
        for outcome in &outcomes_current_white {
            match outcome {
                GameOutcome::BlackWins => {
                    result.wins_baseline += 1;
                    result.black_wins += 1;
                }
                GameOutcome::WhiteWins => {
                    result.wins_current += 1;
                    result.white_wins += 1;
                }
                GameOutcome::Draw => {
                    result.draws += 1;
                }
            }
        }

        result
    }
}

// ============================================================
//  评估对弈逻辑
// ============================================================

/// 运行一局评估对弈。
///
/// 评估时使用确定性搜索：温度设很低（接近 argmax），无 Dirichlet 噪声。
fn play_eval_game(
    black_server: &InferenceServer,
    white_server: &InferenceServer,
    num_simulations: usize,
    seed: u64,
) -> GameOutcome {
    let mut board = Board::new();
    let mut black_mcts = MCTS::new();
    let mut white_mcts = MCTS::new();

    let mut config = GumbelConfig::pure_gumbel(num_simulations);
    config.select_temperature = 0.1;

    // 为每局生成独立的确定性 RNG
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let max_moves = NUM_POSITIONS * 2;

    for _ in 0..max_moves {
        let is_black = board.current_player == Color::Black;

        let eval: &InferenceServer = if is_black { black_server } else { white_server };
        let mcts = if is_black {
            &mut black_mcts
        } else {
            &mut white_mcts
        };

        let result = mcts.search(&mut board, eval, &config, &mut rng);

        if result.best_move >= NUM_POSITIONS || !board.play_idx(result.best_move) {
            break;
        }

        if board.game_over {
            break;
        }
    }

    match board.winner {
        Some(Color::Black) => GameOutcome::BlackWins,
        Some(Color::White) => GameOutcome::WhiteWins,
        None => GameOutcome::Draw,
    }
}

// ============================================================
//  基线模型管理
// ============================================================

/// 管理 baseline 模型文件的路径。
pub struct BaselineManager {
    model_dir: PathBuf,
}

impl BaselineManager {
    pub fn new(model_dir: PathBuf) -> Self {
        Self { model_dir }
    }

    fn baseline_path(&self) -> PathBuf {
        self.model_dir.join("gomoku_baseline")
    }

    /// 加载基线模型（若不存在则返回 None）
    pub fn load_baseline(&self, device: &Device) -> Option<GomokuNetwork> {
        let path = self.baseline_path();
        match burn::store::ModuleRecord::load(&path) {
            Ok(record) => {
                println!("  Loaded baseline model from {:?}", path);
                Some(GomokuNetwork::new(device).load_record(record))
            }
            Err(_) => None,
        }
    }

    /// 将当前模型保存为新的基线
    pub fn promote(&self, model: &GomokuNetwork) {
        let path = self.baseline_path();
        if let Err(e) = model.clone().valid().save_file(&path) {
            eprintln!("  Warning: failed to save baseline: {}", e);
        } else {
            println!("  Baseline promoted! Saved to {:?}", path);
        }
    }
}
