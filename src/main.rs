use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "gobang-ai")]
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
            println!("=== Gobang AI (AlphaZero) - Training ===\n");

            let device = burn::tensor::Device::default();
            let config = gobang_ai::training::TrainConfig {
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
            let mut trainer = gobang_ai::training::Trainer::new(config, device);
            trainer.train();
        }
    }
}
