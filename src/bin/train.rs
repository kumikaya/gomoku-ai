use std::path::PathBuf;

use clap::Parser;
use gomoku_ai::training::trainer::{TrainConfig, Trainer};

#[derive(Parser)]
#[command(name = "gomoku-train")]
#[command(about = "五子棋 AI — AlphaZero 训练")]
struct Cli {
    /// 训练轮数（默认 400）
    #[arg(short = 'i', long, default_value = "400")]
    iterations: usize,

    /// 每轮自对弈局数
    #[arg(short = 'g', long, default_value = "128")]
    games: usize,

    /// 每次 MCTS 模拟次数（Gumbel Zero 只需 16~64）
    #[arg(short = 's', long, default_value = "128")]
    simulations: usize,

    /// 学习率
    #[arg(short = 'l', long, default_value = "0.001")]
    learning_rate: f64,

    /// 批大小
    #[arg(short = 'b', long, default_value = "256")]
    batch_size: usize,

    /// 模型保存目录
    #[arg(short = 'd', long, default_value = "checkpoints")]
    model_dir: String,

    /// 从指定 checkpoint 恢复训练
    #[arg(short = 'c', long)]
    checkpoint: Option<String>,

    /// 每隔多少轮保存一次
    #[arg(long, default_value = "10")]
    save_every: usize,

    /// 禁用对抗评估
    #[arg(long)]
    no_eval: bool,

    /// 评估局数（默认 20）
    #[arg(long, default_value = "20")]
    eval_games: usize,

    /// 评估模拟次数（默认 64）
    #[arg(long, default_value = "64")]
    eval_simulations: usize,

    /// 晋升阈值（默认 0.55）
    #[arg(long, default_value = "0.55")]
    eval_threshold: f64,
}

fn main() {
    let cli = Cli::parse();

    println!("=== Gomoku AI (AlphaZero) - Training ===\n");

    let device = burn::tensor::Device::default();
    let config = TrainConfig {
        num_simulations: cli.simulations,
        games_per_iteration: cli.games,
        batch_size: cli.batch_size,
        num_iterations: cli.iterations,
        learning_rate: cli.learning_rate,
        save_every: cli.save_every,
        model_dir: cli.model_dir.into(),
        checkpoint: cli.checkpoint.map(PathBuf::from),
        eval_enabled: !cli.no_eval,
        eval_num_games: cli.eval_games,
        eval_num_simulations: cli.eval_simulations,
        eval_promotion_threshold: cli.eval_threshold,
        ..Default::default()
    };
    let mut trainer = Trainer::new(config, device);
    trainer.train();
}
