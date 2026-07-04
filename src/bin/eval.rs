use std::path::PathBuf;

use burn::module::Module;
use burn::store::ModuleRecord;
use clap::Parser;
use gomoku_ai::eval::{EvalConfig, MatchRunner};
use gomoku_ai::inference::InferenceServer;
use gomoku_ai::network::transformer::GomokuNetwork;

#[derive(Parser)]
#[command(name = "gomoku-eval")]
#[command(about = "五子棋 AI — 对抗评估：两个模型对弈")]
struct Cli {
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
}

fn main() {
    let cli = Cli::parse();

    println!("=== Gomoku AI - Tournament Evaluation ===\n");

    let device = burn::tensor::Device::default();
    let challenger_path = PathBuf::from(&cli.challenger);
    let baseline_path = PathBuf::from(&cli.baseline);

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
    println!(
        "Games: {}, Simulations: {}\n",
        cli.num_games, cli.simulations
    );

    let config = EvalConfig {
        num_games: cli.num_games,
        num_simulations: cli.simulations,
        ..Default::default()
    };
    let runner = MatchRunner::new(config);
    let server_c = InferenceServer::new(challenger_model, device.clone());
    let server_b = InferenceServer::new(baseline_model, device);
    let result = runner.run_match(&server_c, &server_b, 0);

    println!();
    result.print();
}
