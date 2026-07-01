//! 神经网络推理抽象层
//!
//! 将 GPU 推理从 MCTS 中解耦，通过 `Evaluator` trait + `InferenceServer`
//! 实现：单 CUDA 上下文，多 MCTS 实例共享，跨请求自动攒批。

use crate::network::transformer::{GomokuNetwork, policy_out_dim};
use burn::tensor::{Device, FloatDType, Int, Tensor};
use std::time::Duration;

/// 批量评估上限（GPU 线程内部攒批的最大请求数）
const SERVER_BATCH_CAP: usize = 256;

// ============================================================
//  Evaluator trait：MCTS 只依赖这个接口，不感知 GPU/CPU
// ============================================================

/// 批量评估接口。
///
/// 输入 `states`：每个元素是 `ENCODE_LEN` 长度的 i32 编码棋盘。
/// 返回 `(policies, values)`：
/// - `policies[i]`：长度 `POLICY_OUT` 的原始 logits
/// - `values[i]`：单个 f32 标量，范围 [-1, 1]
///
/// 返回结果与输入一一对应：`policies.len() == values.len() == states.len()`。
pub trait Evaluator: Send + Sync {
    fn evaluate_batch(&self, states: &[Vec<i32>]) -> (Vec<Vec<f32>>, Vec<f32>);
}

// ============================================================
//  InferenceServer：单 GPU 线程，跨 MCTS 攒批
// ============================================================

struct InferenceRequest {
    states: Vec<Vec<i32>>,
    response_tx: crossbeam_channel::Sender<(Vec<Vec<f32>>, Vec<f32>)>,
}

/// GPU 线程支持两类命令：
/// - `Evaluate`: 推理请求
/// - `UpdateModel`: 热更新模型权重（训练后）
enum GpuCommand {
    Evaluate(InferenceRequest),
    UpdateModel {
        model: GomokuNetwork,
        // 发回确认，确保调用方在模型更新完成前不进行下一轮自对弈
        done_tx: crossbeam_channel::Sender<()>,
    },
}

struct InferenceServerInner {
    cmd_tx: crossbeam_channel::Sender<GpuCommand>,
    gpu_handle: std::thread::JoinHandle<()>,
}

/// GPU 推理服务。
///
/// 内部维护一个专用 GPU 线程（唯一持有 `Device` 和 `GomokuNetwork`），
/// 所有并发的 `evaluate_batch` 调用汇入同一个 channel，
/// GPU 线程自动将多个请求的状态合并为更大的 batch 后做一次 `forward`，
/// 再将结果按请求拆分返回。
///
/// ### 模型热更新
///
/// 训练循环中模型权重持续更新。调用 `update_model()` 可在线替换
/// GPU 线程持有的模型，无需销毁重建 → 避免 cubecl 为每个线程分配
/// 独立的 CUDA workspace pool 导致显存线性增长。
///
/// ### 生命周期
///
/// 析构时关闭请求通道 → 等待 GPU 线程完成最后一批推理后退出。
/// 如果在退出前有未完成的请求，`Drop` 会 join 线程确保安全。
pub struct InferenceServer {
    inner: Option<InferenceServerInner>,
}

impl InferenceServer {
    /// 创建一个新的推理服务。
    ///
    /// `model` 所有权移入服务内部的 GPU 线程（非 autodiff）。
    /// `device` 必须与 `model` 的 device 一致。
    pub fn new(model: GomokuNetwork, device: Device) -> Self {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<GpuCommand>();

        let gpu_handle = std::thread::spawn(move || {
            Self::gpu_loop(cmd_rx, model, device);
        });

        Self {
            inner: Some(InferenceServerInner { cmd_tx, gpu_handle }),
        }
    }

    /// 热更新 GPU 线程持有的模型权重。
    ///
    /// 阻塞直到更新完成。调用方应在训练更新权重后调用此方法，
    /// 使下一轮自对弈使用最新模型。
    pub fn update_model(&self, model: GomokuNetwork) {
        let inner = self
            .inner
            .as_ref()
            .expect("InferenceServer already shut down");
        let (done_tx, done_rx) = crossbeam_channel::bounded(0);
        inner
            .cmd_tx
            .send(GpuCommand::UpdateModel { model, done_tx })
            .expect("GPU inference thread died");
        done_rx.recv().expect("GPU inference thread died");
    }

    /// 对当前 batch 做一次 forward 并分发结果。
    fn forward_batch(model: &GomokuNetwork, device: &Device, batch: &mut Vec<InferenceRequest>) {
        if batch.is_empty() {
            return;
        }

        // 计算总状态数
        let total_states: usize = batch.iter().map(|r| r.states.len()).sum();
        let state_size = batch[0].states[0].len(); // ENCODE_LEN

        // 扁平化所有状态
        let mut flat = Vec::with_capacity(total_states * state_size);
        for req in batch.iter() {
            for state in &req.states {
                flat.extend_from_slice(state);
            }
        }

        // GPU 批量前向：i32 输入 → [batch, ENCODE_LEN]
        let state_tensor = Tensor::<2, Int>::from_data(
            burn::tensor::TensorData::new(flat, [total_states as i32, state_size as i32]),
            device,
        );
        let policy_out = policy_out_dim(model.board_size);
        let (logits, values) = model.forward(state_tensor);
        // 若 device 配置为 f16，cast 回 f32 才能 to_vec::<f32>()
        let policy_flat: Vec<f32> = logits
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let values_flat: Vec<f32> = values
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // 按请求拆分结果
        let mut pol_offset = 0;
        let mut val_offset = 0;
        for req in batch.drain(..) {
            let n = req.states.len();
            let policies: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    let start = pol_offset + i * policy_out;
                    policy_flat[start..start + policy_out].to_vec()
                })
                .collect();
            let values: Vec<f32> = values_flat[val_offset..val_offset + n].to_vec();

            pol_offset += n * policy_out;
            val_offset += n;

            let _ = req.response_tx.send((policies, values));
        }
    }

    fn gpu_loop(
        rx: crossbeam_channel::Receiver<GpuCommand>,
        mut model: GomokuNetwork,
        device: Device,
    ) {
        let mut batch: Vec<InferenceRequest> = Vec::with_capacity(SERVER_BATCH_CAP);

        'outer: loop {
            // 阻塞等待第一个命令
            let first = match rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break,
            };

            match first {
                GpuCommand::Evaluate(req) => {
                    batch.push(req);
                }
                GpuCommand::UpdateModel {
                    model: new_model,
                    done_tx,
                } => {
                    model = new_model;
                    let _ = done_tx.send(());
                    continue;
                }
            }

            // 短超时攒更多推理请求
            loop {
                if batch.len() >= SERVER_BATCH_CAP {
                    break;
                }
                match rx.recv_timeout(Duration::from_micros(200)) {
                    Ok(GpuCommand::Evaluate(req)) => batch.push(req),
                    Ok(GpuCommand::UpdateModel {
                        model: new_model,
                        done_tx,
                    }) => {
                        // 先处理当前 batch，再更新模型
                        Self::forward_batch(&model, &device, &mut batch);
                        model = new_model;
                        let _ = done_tx.send(());
                        continue 'outer;
                    }
                    Err(_) => break, // 超时或 channel 关闭
                }
            }

            Self::forward_batch(&model, &device, &mut batch);
        }
    }
}

impl Evaluator for InferenceServer {
    fn evaluate_batch(&self, states: &[Vec<i32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let inner = self
            .inner
            .as_ref()
            .expect("InferenceServer already shut down");
        let (response_tx, response_rx) = crossbeam_channel::bounded(1);
        inner
            .cmd_tx
            .send(GpuCommand::Evaluate(InferenceRequest {
                states: states.to_vec(),
                response_tx,
            }))
            .expect("GPU inference thread died");
        response_rx.recv().expect("GPU inference thread died")
    }
}

impl Drop for InferenceServer {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            // 关闭请求通道，通知 GPU 线程退出
            drop(inner.cmd_tx);
            // 等待 GPU 线程处理完最后一批
            let _ = inner.gpu_handle.join();
        }
    }
}
