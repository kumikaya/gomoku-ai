//! 神经网络推理抽象层
//!
//! 将 GPU 推理从 MCTS 中解耦，通过 `Evaluator` trait + `InferenceServer`
//! 实现：单 CUDA 上下文，多 MCTS 实例共享，跨请求自动攒批。

use crate::network::residual::{BOARD_SIZE, GomokuNetwork, INPUT_CHANNELS, POLICY_OUT};
use burn::tensor::{Device, Tensor};
use std::time::Duration;

/// 批量评估上限（GPU 线程内部攒批的最大请求数）
const SERVER_BATCH_CAP: usize = 256;

// ============================================================
//  Evaluator trait：MCTS 只依赖这个接口，不感知 GPU/CPU
// ============================================================

/// 批量评估接口。
///
/// 输入 `states`：每个元素是 `ENCODE_CHANNELS * NUM_POSITIONS` 长度的编码棋盘。
/// 返回 `(policies, values)`：
/// - `policies[i]`：长度 `POLICY_OUT`(225) 的原始 logits
/// - `values[i]`：单个 f32 标量，范围 [-1, 1]
///
/// 返回结果与输入一一对应：`policies.len() == values.len() == states.len()`。
pub trait Evaluator: Send + Sync {
    fn evaluate_batch(&self, states: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>);
}

// ============================================================
//  InferenceServer：单 GPU 线程，跨 MCTS 攒批
// ============================================================

struct InferenceRequest {
    states: Vec<Vec<f32>>,
    response_tx: crossbeam_channel::Sender<(Vec<Vec<f32>>, Vec<f32>)>,
}

struct InferenceServerInner {
    request_tx: crossbeam_channel::Sender<InferenceRequest>,
    gpu_handle: std::thread::JoinHandle<()>,
}

/// GPU 推理服务。
///
/// 内部维护一个专用 GPU 线程（唯一持有 `Device` 和 `GomokuNetwork`），
/// 所有并发的 `evaluate_batch` 调用汇入同一个 channel，
/// GPU 线程自动将多个请求的状态合并为更大的 batch 后做一次 `forward`，
/// 再将结果按请求拆分返回。
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
        let (request_tx, request_rx) = crossbeam_channel::unbounded::<InferenceRequest>();

        let gpu_handle = std::thread::spawn(move || {
            Self::gpu_loop(request_rx, model, device);
        });

        Self {
            inner: Some(InferenceServerInner {
                request_tx,
                gpu_handle,
            }),
        }
    }

    fn gpu_loop(
        rx: crossbeam_channel::Receiver<InferenceRequest>,
        model: GomokuNetwork,
        device: Device,
    ) {
        let mut batch: Vec<InferenceRequest> = Vec::with_capacity(SERVER_BATCH_CAP);

        loop {
            // 阻塞等待第一个请求
            match rx.recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }

            // 短超时攒更多请求（跨 MCTS 实例聚合）
            while batch.len() < SERVER_BATCH_CAP {
                match rx.recv_timeout(Duration::from_micros(200)) {
                    Ok(req) => batch.push(req),
                    Err(_) => break,
                }
            }

            let _n_requests = batch.len();

            // 计算总状态数
            let total_states: usize = batch.iter().map(|r| r.states.len()).sum();
            let state_size = batch[0].states[0].len(); // ENCODE_CHANNELS * NUM_POSITIONS

            // 扁平化所有状态
            let mut flat = Vec::with_capacity(total_states * state_size);
            for req in &batch {
                for state in &req.states {
                    flat.extend_from_slice(state);
                }
            }

            // GPU 批量前向
            let state_tensor = Tensor::<1>::from_floats(flat.as_slice(), &device).reshape([
                total_states as i32,
                INPUT_CHANNELS as i32,
                BOARD_SIZE as i32,
                BOARD_SIZE as i32,
            ]);
            let (logits, values) = model.forward(state_tensor);
            let policy_flat: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap();
            let values_flat: Vec<f32> = values.into_data().to_vec::<f32>().unwrap();

            // 按请求拆分结果
            let mut pol_offset = 0;
            let mut val_offset = 0;
            for req in batch.drain(..) {
                let n = req.states.len();
                let policies: Vec<Vec<f32>> = (0..n)
                    .map(|i| {
                        let start = pol_offset + i * POLICY_OUT;
                        policy_flat[start..start + POLICY_OUT].to_vec()
                    })
                    .collect();
                let values: Vec<f32> = values_flat[val_offset..val_offset + n].to_vec();

                pol_offset += n * POLICY_OUT;
                val_offset += n;

                let _ = req.response_tx.send((policies, values));
            }
        }
    }
}

impl Evaluator for InferenceServer {
    fn evaluate_batch(&self, states: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let inner = self
            .inner
            .as_ref()
            .expect("InferenceServer already shut down");
        let (response_tx, response_rx) = crossbeam_channel::bounded(1);
        inner
            .request_tx
            .send(InferenceRequest {
                states: states.to_vec(),
                response_tx,
            })
            .expect("GPU inference thread died");
        response_rx.recv().expect("GPU inference thread died")
    }
}

impl Drop for InferenceServer {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            // 关闭请求通道，通知 GPU 线程退出
            drop(inner.request_tx);
            // 等待 GPU 线程处理完最后一批
            let _ = inner.gpu_handle.join();
        }
    }
}
