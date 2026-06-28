use burn::module::Module;
use clap::{Parser, Subcommand};
use gomoku_ai::network::residual::GomokuNetwork;

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
        #[arg(short = 'i', long, default_value = "100")]
        iterations: usize,

        /// 每轮自对弈局数
        #[arg(short = 'g', long, default_value = "4")]
        games: usize,

        /// 每次 MCTS 模拟次数
        #[arg(short = 's', long, default_value = "200")]
        simulations: usize,

        /// 学习率
        #[arg(short = 'l', long, default_value = "0.001")]
        learning_rate: f64,

        /// 批大小
        #[arg(short = 'b', long, default_value = "64")]
        batch_size: usize,

        /// 模型保存目录
        #[arg(short = 'd', long, default_value = "checkpoints")]
        model_dir: String,

        /// 每隔多少轮保存一次
        #[arg(long, default_value = "10")]
        save_every: usize,
    },

    /// 人机对弈
    Play {
        /// MCTS 模拟次数
        #[arg(short = 's', long, default_value = "800")]
        simulations: usize,

        /// 模型文件路径（.bpk）
        #[arg(short = 'm', long, default_value = "checkpoints/gomoku_latest")]
        model_path: String,
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
            save_every,
        } => {
            println!("=== Gomoku AI (AlphaZero) - Training ===\n");

            let device = burn::tensor::Device::default();
            let config = gomoku_ai::training::TrainConfig {
                num_simulations: simulations,
                games_per_iteration: games,
                batch_size,
                train_steps: 50,
                num_iterations: iterations,
                learning_rate,
                value_loss_weight: 1.0,
                save_every,
                model_dir: model_dir.into(),
            };
            let mut trainer = gomoku_ai::training::Trainer::new(config, device);
            trainer.train();
        }

        Command::Play {
            simulations,
            model_path,
        } => {
            let device = burn::tensor::Device::default();
            let path = std::path::PathBuf::from(&model_path);

            if !path.exists() {
                eprintln!("Error: model file not found: {}", path.display());
                eprintln!("Run 'gomoku-ai train' first to create a model.");
                std::process::exit(1);
            }

            let record = burn::store::ModuleRecord::load(&path).unwrap_or_else(|e| {
                eprintln!("Error: failed to load model: {}", e);
                std::process::exit(1);
            });
            let model = GomokuNetwork::new(&device).load_record(record);

            println!("Model loaded. Starting game...");
            gomoku_ai::game::play::play_game(&model, &device, simulations);
        }
    }
}
