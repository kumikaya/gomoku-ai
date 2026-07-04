use std::path::PathBuf;

use burn::module::Module;
use burn::store::ModuleRecord;
use burn::tensor::Device;
use clap::{Parser, Subcommand};
use gomoku_ai::eval::{EvalConfig, MatchRunner};
use gomoku_ai::game::play::play_game;
use gomoku_ai::inference::InferenceServer;
use gomoku_ai::network::transformer::GomokuNetwork;
use gomoku_ai::training::trainer::{TrainConfig, Trainer};

#[derive(Parser)]
#[command(name = "gomoku-ai")]
#[command(about = "五子棋 AI — AlphaZero 训练与推理")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 训练模型
    Train {
        /// 训练轮数（默认 100）
        #[arg(short = 'i', long, default_value = "400")]
        iterations: usize,

        /// 每轮自对弈局数
        #[arg(short = 'g', long, default_value = "128")]
        games: usize,

        /// 每次 MCTS 模拟次数（Gumbel Zero 只需 16~64）
        #[arg(short = 's', long, default_value = "32")]
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
    },

    /// 人机对弈
    Play {
        /// MCTS 模拟次数（Gumbel Zero 只需 32~64）
        #[arg(short = 's', long, default_value = "128")]
        simulations: usize,

        /// 模型文件路径（.bpk）
        #[arg(short = 'm', long)]
        model_path: String,
    },

    /// 对抗评估：两个模型对弈
    Eval {
        /// 挑战者模型路径
        #[arg(short = 'c', long)]
        challenger: String,

        /// 基线模型路径
        #[arg(short = 'b', long)]
        baseline: String,

        /// 对弈局数
        #[arg(short = 'n', long, default_value = "100")]
        num_games: usize,

        /// MCTS 模拟次数
        #[arg(short = 's', long, default_value = "64")]
        simulations: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Train {
            iterations,
            games,
            simulations,
            learning_rate,
            batch_size,
            model_dir,
            checkpoint,
            save_every,
            no_eval,
            eval_games,
            eval_simulations,
            eval_threshold,
        } => {
            println!("=== Gomoku AI (AlphaZero) - Training ===\n");

            let device = create_device();
            let config = TrainConfig {
                num_simulations: simulations,
                games_per_iteration: games,
                batch_size,
                num_iterations: iterations,
                learning_rate,
                save_every,
                model_dir: model_dir.into(),
                checkpoint: checkpoint.map(PathBuf::from),
                eval_enabled: !no_eval,
                eval_num_games: eval_games,
                eval_num_simulations: eval_simulations,
                eval_promotion_threshold: eval_threshold,
                ..Default::default()
            };
            let mut trainer = Trainer::new(config, device);
            trainer.train();
        }

        Command::Play {
            simulations,
            model_path,
        } => {
            let device = create_device();
            let path = PathBuf::from(&model_path);

            let record = ModuleRecord::load(&path).unwrap_or_else(|e| {
                eprintln!("Error: failed to load model: {}", e);
                std::process::exit(1);
            });
            let model = GomokuNetwork::new(&device).load_record(record);

            let server = InferenceServer::new(model, device.clone());
            println!("Model loaded. Starting game...");
            play_game(&server, simulations);
        }

        Command::Eval {
            challenger,
            baseline,
            num_games,
            simulations,
        } => {
            println!("=== Gomoku AI - Tournament Evaluation ===\n");

            let device = create_device();
            let challenger_path = PathBuf::from(&challenger);
            let baseline_path = PathBuf::from(&baseline);

            let challenger_record = ModuleRecord::load(&challenger_path).unwrap_or_else(|e| {
                eprintln!("Error: failed to load challenger model: {}", e);
                std::process::exit(1);
            });
            let baseline_record = ModuleRecord::load(&baseline_path).unwrap_or_else(|e| {
                eprintln!("Error: failed to load baseline model: {}", e);
                std::process::exit(1);
            });

            let challenger_model = GomokuNetwork::new(&device).load_record(challenger_record);
            let baseline_model = GomokuNetwork::new(&device).load_record(baseline_record);

            println!(
                "Challenger: {:?}  vs  Baseline: {:?}",
                challenger_path, baseline_path
            );
            println!("Games: {}, Simulations: {}\n", num_games, simulations);

            let config = EvalConfig {
                num_games,
                num_simulations: simulations,
                ..Default::default()
            };
            let runner = MatchRunner::new(config);
            let server_c = InferenceServer::new(challenger_model, device.clone());
            let server_b = InferenceServer::new(baseline_model, device);
            let result = runner.run_match(&server_c, &server_b, 0);

            println!();
            result.print();
        }
    }
}

/// 创建 device 并设置默认精度。
fn create_device() -> Device {
    Device::default()
}
