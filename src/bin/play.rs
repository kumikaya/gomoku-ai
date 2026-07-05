use std::path::PathBuf;

use burn::module::Module;
use burn::store::ModuleRecord;
use clap::Parser;
use gomoku_ai::game::play::play_game;
use gomoku_ai::inference::InferenceServer;
use gomoku_ai::network::transformer::GomokuNetwork;

#[derive(Parser)]
#[command(name = "gomoku-play")]
#[command(about = "五子棋 AI — 人机对弈")]
struct Cli {
    /// MCTS 模拟次数（Gumbel Zero 只需 32~64）
    #[arg(short = 's', long, default_value = "512")]
    simulations: usize,

    /// 模型文件路径（.bpk）
    #[arg(short = 'm', long)]
    model_path: String,
}

fn main() {
    let cli = Cli::parse();

    let device = burn::tensor::Device::default();
    let path = PathBuf::from(&cli.model_path);

    let record = ModuleRecord::load(&path).unwrap_or_else(|e| {
        eprintln!("Error: failed to load model: {}", e);
        std::process::exit(1);
    });
    let model = GomokuNetwork::new(&device).load_record(record);

    let server = InferenceServer::new(model, device.clone());
    println!("Model loaded. Starting game...");
    play_game(&server, cli.simulations);
}
