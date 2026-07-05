//! 神经网络棋力评估
//!
//! 通过对抗对弈评估模型实力：
//! - `MatchRunner`: 让两个模型对弈 N 局，统计胜率
//! - `EloTracker`: Elo 评分追踪（1500 基准）
//! - `BaselineServer`: 长期缓存 baseline 的 InferenceServer，避免重复创建 GPU 线程

use crate::game::board::{Board, Color};
use crate::inference::InferenceServer;
use crate::mcts::node::{GumbelConfig, MCTS};
use crate::network::transformer::GomokuNetwork;

use burn::module::{AutodiffModule, Module};
use burn::tensor::Device;
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rayon::prelude::*;
use std::path::PathBuf;

// ============================================================
//  Elo 评分系统
// ============================================================

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameOutcome {
    BlackWins,
    WhiteWins,
    Draw,
}

#[derive(Debug, Clone, Default)]
pub struct MatchResult {
    pub total_games: usize,
    pub wins_current: usize,
    pub wins_baseline: usize,
    pub draws: usize,
    pub black_wins: usize,
    pub white_wins: usize,
}

impl MatchResult {
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
    pub num_games: usize,
    pub num_simulations: usize,
    pub eval_every: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            num_games: 100,
            num_simulations: 64,
            eval_every: 5,
        }
    }
}

// ============================================================
//  BaselineServer: 长期缓存 baseline 的 GPU 推理线程
// ============================================================

/// 整个训练周期只创建一个 GPU 线程，晋升时通过 `update_model` 热更新权重，
/// 避免反复创建/销毁 GPU 线程导致显存碎片。
pub struct BaselineServer {
    server: InferenceServer,
    model_dir: PathBuf,
}

impl BaselineServer {
    pub fn new(initial_model: &GomokuNetwork, model_dir: PathBuf, device: &Device) -> Self {
        let baseline_path = model_dir.join("gomoku_baseline");

        let model = if let Ok(record) = burn::store::ModuleRecord::load(&baseline_path) {
            println!("  Loaded baseline model from {:?}", baseline_path);
            GomokuNetwork::new(device).load_record(record)
        } else {
            println!("  No baseline found, using current model as baseline.");
            initial_model.clone().valid()
        };

        Self {
            server: InferenceServer::new(model, device.clone()),
            model_dir,
        }
    }

    pub fn promote(&self, model: &GomokuNetwork) {
        let path = self.model_dir.join("gomoku_baseline");
        if let Err(e) = model.clone().valid().save_file(&path) {
            eprintln!("  Warning: failed to save baseline: {}", e);
            return;
        }
        println!("  Baseline promoted! Saved to {:?}", path);
        self.server.update_model(model.clone().valid());
    }

    pub fn server(&self) -> &InferenceServer {
        &self.server
    }
}

// ============================================================
//  MatchRunner: 两模型对抗对弈
// ============================================================

/// 不持有自己的 InferenceServer，接受外部引用：
/// - `current`: 训练循环中的 `inference_server`
/// - `baseline`: `BaselineServer` 中缓存的 server
pub struct MatchRunner {
    config: EvalConfig,
}

impl MatchRunner {
    pub fn new(config: EvalConfig) -> Self {
        Self { config }
    }

    pub fn run_match(
        &self,
        current_server: &InferenceServer,
        baseline_server: &InferenceServer,
        rng_seed: u64,
    ) -> MatchResult {
        let num_games = self.config.num_games;
        let half = num_games / 2;

        let pb = ProgressBar::new(num_games as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Eval match: {bar:40.yellow/blue} {pos}/{len} ({eta})")
                .unwrap(),
        );

        let outcomes_current_black: Vec<GameOutcome> = (0..half)
            .into_par_iter()
            .map(|game_i| {
                let seed = rng_seed.wrapping_add(game_i as u64);
                let result = play_eval_game(
                    current_server,
                    baseline_server,
                    self.config.num_simulations,
                    seed,
                );
                pb.inc(1);
                result
            })
            .collect();

        let outcomes_current_white: Vec<GameOutcome> = (0..num_games - half)
            .into_par_iter()
            .map(|game_i| {
                let seed = rng_seed.wrapping_add((half + game_i) as u64);
                let result = play_eval_game(
                    baseline_server,
                    current_server,
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

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let max_moves = board.num_positions() * 2;

    let npos = board.num_positions();

    for _ in 0..max_moves {
        let is_black = board.current_player == Color::Black;

        let eval: &InferenceServer = if is_black { black_server } else { white_server };
        let mcts = if is_black {
            &mut black_mcts
        } else {
            &mut white_mcts
        };

        let result = mcts.search(&mut board, eval, &config, &mut rng);

        if result.best_move >= npos || !board.play_idx(result.best_move) {
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
