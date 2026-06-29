use burn::module::{AutodiffModule, Module};
use burn::tensor::{Device, FloatDType, IntDType};
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
        #[arg(short = 'b', long, default_value = "128")]
        batch_size: usize,

        /// 模型保存目录
        #[arg(short = 'd', long, default_value = "checkpoints")]
        model_dir: String,

        /// 每隔多少轮保存一次
        #[arg(long, default_value = "5")]
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

            let device = create_device();
            let config = gomoku_ai::training::trainer::TrainConfig {
                num_simulations: simulations,
                games_per_iteration: games,
                batch_size,
                num_iterations: iterations,
                learning_rate,
                save_every,
                model_dir: model_dir.into(),
                ..Default::default()
            };
            let mut trainer = gomoku_ai::training::trainer::Trainer::new(config, device);
            trainer.train();
        }

        Command::Play {
            simulations,
            model_path,
        } => {
            let device = create_device();
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

            let server = gomoku_ai::inference::InferenceServer::new(model.valid(), device.clone());
            println!("Model loaded. Starting game...");
            gomoku_ai::game::play::play_game(&server, simulations);
        }
    }
}

/// 创建 device 并设置默认精度。
///
/// 训练+推理统一使用 BF16（指数位与 F32 相同，避免梯度溢出）。
/// backend 不支持 BF16 时自动降级 F32。
fn create_device() -> Device {
    let mut device = Device::default();
    match device.configure((FloatDType::BF16, IntDType::I32)) {
        Ok(()) => println!("Device: float=BF16"),
        Err(e) => {
            eprintln!("BF16 not supported ({e}), falling back to F32");
            device.configure((FloatDType::F32, IntDType::I32)).ok();
        }
    }
    device
}
