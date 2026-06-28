fn main() {
    println!("=== Gobang AI (AlphaZero) ===\n");

    let device = burn::tensor::Device::default();
    let config = gobang_ai::training::TrainConfig::default();
    let mut trainer = gobang_ai::training::Trainer::new(config, device);

    trainer.train();
}
