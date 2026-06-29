//! 蒙特卡洛树搜索 (MCTS)
//!
//! 使用 PUCT 公式选择节点，神经网络评估叶节点。
//!
//! ## 并行化策略（Leaf Parallelization + Batched Evaluation）
//!
//! 共享一棵树，K 个 worker 同时做 PUCT 走树；到达叶子时把状态丢进一个
//! batch 队列，evaluator 线程攒满一个 batch 后做一次 NN forward，
//! 再分发结果给 worker 各自回传。相比原来的 Root Parallelization：
//!
//! - 不再 clone 整棵树 → 树字段原子化，`Arc` 共享
//! - 虚拟损失跨线程可见（原子操作）
//! - 网络评估由专门的 evaluator 协调 batch，提升 GPU 利用率
//!
//! ## 节点统计量语义
//!
//! 每个节点的 `visit_count` 和 `total_value` 存储的是**父节点视角**下
//! 选择该子节点的模拟统计：
//! - `visit_count`: 有多少次模拟选择了该子节点（即父→子的边被穿越的次数）
//! - `total_value`: 这些模拟从父节点视角累积的价值和
//! - `Q = total_value / visit_count`：该边的平均价值

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::game::board::{Board, ENCODE_CHANNELS, NUM_POSITIONS};
use crate::network::residual::{BOARD_SIZE, GomokuNetwork, INPUT_CHANNELS, POLICY_OUT};
use burn::tensor::{Device, Tensor};
use rayon::prelude::*;

// ============================================================
//  常量
// ============================================================

/// 虚拟损失：模拟期间施加的惩罚值，防止并行线程重复选择同一路径。
const VIRTUAL_LOSS: f32 = 3.0;

/// PUCT 探索常数
const C_PUCT: f32 = 1.0;

/// Dirichlet 噪声的集中参数
const DIRICHLET_ALPHA: f32 = 0.3;

/// Dirichlet 噪声在混合先验中的权重
const DIRICHLET_EPSILON: f32 = 0.25;

/// 获胜模拟的价值
const WIN_VALUE: f32 = 1.0;
/// 平局模拟的价值
const DRAW_VALUE: f32 = 0.0;

/// 批量评估上限
const BATCH_CAP: usize = 256;

// ============================================================
//  f32 ↔ AtomicU32 转换
// ============================================================

#[inline]
fn f2u(x: f32) -> u32 {
    x.to_bits()
}
#[inline]
fn u2f(x: u32) -> f32 {
    f32::from_bits(x)
}

// ============================================================
//  Node：原子化的树节点
// ============================================================

pub struct Node {
    pub visit_count: AtomicU32,
    pub total_value: AtomicU32,
    pub prior: f32,
    pub virtual_loss: AtomicU32,
    pub children: parking_lot::RwLock<Vec<Option<usize>>>,
    pub expanded: AtomicBool,
}

impl Node {
    pub fn new(prior: f32) -> Self {
        Self {
            visit_count: AtomicU32::new(0),
            total_value: AtomicU32::new(f2u(0.0)),
            prior,
            virtual_loss: AtomicU32::new(0),
            children: parking_lot::RwLock::new(vec![None; NUM_POSITIONS]),
            expanded: AtomicBool::new(false),
        }
    }

    #[inline]
    pub fn visit_count_f32(&self) -> f32 {
        self.visit_count.load(Ordering::Relaxed) as f32
    }

    #[inline]
    pub fn total_value_f32(&self) -> f32 {
        u2f(self.total_value.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn q(&self) -> f32 {
        let n = self.visit_count_f32();
        if n > 0.0 {
            self.total_value_f32() / n
        } else {
            0.0
        }
    }

    #[inline]
    pub fn effective_n(&self) -> f32 {
        self.visit_count_f32() + self.virtual_loss.load(Ordering::Relaxed) as f32
    }

    #[inline]
    pub fn effective_q(&self) -> f32 {
        let n = self.effective_n();
        if n > 0.0 {
            self.total_value_f32() / n
        } else {
            0.0
        }
    }

    #[inline]
    pub fn add_virtual_loss(&self) {
        self.virtual_loss.fetch_add(1, Ordering::Relaxed);
        let mut cur = self.total_value.load(Ordering::Relaxed);
        loop {
            let new = f2u(u2f(cur) - VIRTUAL_LOSS);
            match self.total_value.compare_exchange_weak(
                cur,
                new,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    }

    #[inline]
    pub fn remove_virtual_loss(&self) {
        self.virtual_loss.fetch_sub(1, Ordering::Relaxed);
        let mut cur = self.total_value.load(Ordering::Relaxed);
        loop {
            let new = f2u(u2f(cur) + VIRTUAL_LOSS);
            match self.total_value.compare_exchange_weak(
                cur,
                new,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    }

    #[inline]
    pub fn add_visit(&self, value: f32) {
        self.visit_count.fetch_add(1, Ordering::Relaxed);
        let mut cur = self.total_value.load(Ordering::Relaxed);
        loop {
            let new = f2u(u2f(cur) + value);
            match self.total_value.compare_exchange_weak(
                cur,
                new,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    }
}

// ============================================================
//  MCTS 搜索器
// ============================================================

/// MCTS 搜索器。
///
/// 节点存储在 `boxcar::Vec<Node>` 中：`push()` 返回 `usize` 索引，
/// `get(idx)` 无锁返回 `&Node`。分块链表保证 push 后元素位置永不变。
///
/// arena 用 `Mutex` 包裹以支持 `reset()`（需要 `&mut` 以调用 `clear()`），
/// 但 `node()` 每次锁一次读取——`parking_lot::Mutex` 在此场景下足够轻量。
pub struct MCTS {
    arena: boxcar::Vec<Node>,
}

/// 搜索返回结果
pub struct SearchResult {
    pub best_move: usize,
    pub policy: Vec<f32>,
    pub root_value: f32,
}

// ── Evaluator / Worker 通信 ──

struct LeafRequest {
    path: Vec<(usize, usize, usize)>,
    encoding: Vec<f32>,
    legal_moves_at_leaf: Vec<(usize, usize)>,
    reply_tx: crossbeam_channel::Sender<LeafReply>,
}

struct LeafReply {
    value: f32,
}

impl MCTS {
    pub fn new() -> Self {
        Self {
            arena: boxcar::vec![Node::new(0.0)],
        }
    }

    #[inline]
    fn root_idx(&self) -> usize {
        0
    }

    // ── 节点访问 ──
    #[inline]
    fn root(&self) -> &Node {
        &self.arena[0]
    }

    /// 无锁读取节点。调用方需保证 idx 有效。
    #[inline]
    fn node(&self, idx: usize) -> &Node {
        &self.arena[idx]
    }

    /// 展开节点：为所有合法走法创建子节点，填入 parent.children。
    ///
    /// expand_node 只在 evaluator 单线程和 search() 的 root 展开阶段调用，
    /// 不存在并发写冲突。
    fn expand_node(
        &self,
        parent: usize,
        legal: &[(usize, usize)],
        probs: &[f32],
        noise: Option<(&[f32], f32)>,
    ) {
        let mut child_ids = Vec::with_capacity(legal.len());
        for (i, &(r, c)) in legal.iter().enumerate() {
            let idx = Board::pos_to_idx(r, c);
            let prior = match noise {
                Some((dir, eps)) => (1.0 - eps) * probs[idx] + eps * dir[i],
                None => probs[idx],
            };
            child_ids.push(self.arena.push(Node::new(prior)));
        }

        let parent_node = self.node(parent);
        let mut children = parent_node.children.write();
        for (i, &(r, c)) in legal.iter().enumerate() {
            let idx = Board::pos_to_idx(r, c);
            children[idx] = Some(child_ids[i]);
        }
    }

    // ── 公共搜索入口 ──

    pub fn search(
        &self,
        board: &mut Board,
        network: &GomokuNetwork,
        device: &Device,
        num_simulations: usize,
        temperature: f32,
    ) -> SearchResult {
        let root_idx = self.root_idx();
        let legal_moves = board.legal_moves();
        if legal_moves.is_empty() {
            return SearchResult {
                best_move: NUM_POSITIONS,
                policy: vec![0.0; NUM_POSITIONS],
                root_value: 0.0,
            };
        }

        // ── 根节点评估 ──
        let (policy_logits, value) = {
            let mut buf = vec![0.0f32; ENCODE_CHANNELS * NUM_POSITIONS];
            board.encode_into(&mut buf);
            let state = Tensor::<1>::from_floats(buf.as_slice(), device).reshape([
                1,
                INPUT_CHANNELS as i32,
                BOARD_SIZE as i32,
                BOARD_SIZE as i32,
            ]);
            network.forward(state)
        };
        let policy_probs = {
            let policy_bytes = policy_logits.into_data().to_vec::<f32>().unwrap();
            Self::softmax_legal(&policy_bytes, &legal_moves)
        };
        let root_value: f32 = value.reshape([1]).into_scalar();

        let dirichlet = Self::dirichlet_noise(legal_moves.len(), DIRICHLET_ALPHA);
        self.expand_node(
            root_idx,
            &legal_moves,
            &policy_probs,
            Some((&dirichlet, DIRICHLET_EPSILON)),
        );
        self.root().expanded.store(true, Ordering::Release);

        // ── 并行模拟阶段 ──
        let (leaf_tx, leaf_rx) = crossbeam_channel::bounded::<LeafRequest>(BATCH_CAP * 4);

        // 使用 scoped thread：evaluator 在独立线程运行，workers 通过 rayon 并发
        // 所有 channel sender 的生命周期由 move closure 管理
        std::thread::scope(move |s| {
            // Evaluator 线程：拥有 leaf_rx，攒 batch 做 forward
            s.spawn(move || {
                Self::evaluator_run(self, leaf_rx, network, device);
            });

            // Worker 池（rayon，阻塞直到全部完成）
            let num_workers = rayon::current_num_threads().min(num_simulations).max(1);
            let sims_per_worker = num_simulations / num_workers;
            let remainder = num_simulations % num_workers;

            (0..num_workers).into_par_iter().for_each(|tid| {
                let sims = if tid < remainder {
                    sims_per_worker + 1
                } else {
                    sims_per_worker
                };
                let mut sim_board = board.clone();
                let mut legal = Vec::with_capacity(NUM_POSITIONS);
                self.worker_loop(sims, &mut sim_board, &mut legal, &leaf_tx);
            });

            // Workers 结束，显式丢弃最后一个 sender 以关闭 channel
            drop(leaf_tx);
        });

        // ── 构建策略分布 ──
        let root = self.node(root_idx);
        let children = root.children.read();
        let sum_n: f32 = children
            .iter()
            .filter_map(|c| c.map(|ci| self.node(ci).visit_count_f32()))
            .sum();
        let mut policy = vec![0.0f32; NUM_POSITIONS];
        if sum_n > 0.0 {
            for (idx, child_opt) in children.iter().enumerate() {
                if let &Some(child_idx) = child_opt {
                    policy[idx] = self.node(child_idx).visit_count_f32() / sum_n;
                }
            }
        }
        drop(children);

        let best_move = if temperature > 0.0 {
            Self::sample_with_temperature(&policy, temperature)
        } else {
            policy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        SearchResult {
            best_move,
            policy,
            root_value,
        }
    }

    // ── Worker：走树 + 提交叶子 + 等待回传 ──

    fn worker_loop(
        &self,
        num_sims: usize,
        board: &mut Board,
        legal: &mut Vec<(usize, usize)>,
        leaf_tx: &crossbeam_channel::Sender<LeafRequest>,
    ) {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded::<LeafReply>(1);

        for _ in 0..num_sims {
            let mut path: Vec<(usize, usize, usize)> = Vec::new();
            let mut sim_board = board.clone();
            let mut node_idx = 0usize; // root

            loop {
                if sim_board.game_over {
                    let v = match sim_board.winner {
                        Some(_) => WIN_VALUE,
                        None => DRAW_VALUE,
                    };
                    self.backprop_path(&path, v);
                    break;
                }

                sim_board.fill_legal_moves(legal);
                if legal.is_empty() {
                    self.backprop_path(&path, DRAW_VALUE);
                    break;
                }

                let node = self.node(node_idx);
                if !node.expanded.load(Ordering::Acquire) {
                    // 叶子：提交评估请求
                    let mut encoding = vec![0.0f32; ENCODE_CHANNELS * NUM_POSITIONS];
                    sim_board.encode_into(&mut encoding);
                    leaf_tx
                        .send(LeafRequest {
                            path: path.clone(),
                            encoding,
                            legal_moves_at_leaf: legal.clone(),
                            reply_tx: reply_tx.clone(),
                        })
                        .expect("evaluator died");

                    let reply = reply_rx.recv().expect("evaluator dropped reply");
                    self.backprop_path(&path, -reply.value);
                    break;
                }

                // PUCT 选择
                let parent_sqrt_n = node.effective_n().sqrt();
                let children = node.children.read();
                let mut best: Option<(usize, usize, f32)> = None;
                for &(r, c) in legal.iter() {
                    let idx = Board::pos_to_idx(r, c);
                    if let Some(ci) = children[idx] {
                        let child = self.node(ci);
                        let score = child.effective_q()
                            + C_PUCT * child.prior * parent_sqrt_n / (1.0 + child.effective_n());
                        if best.map_or(true, |(_, _, s)| score > s) {
                            best = Some((idx, ci, score));
                        }
                    }
                }
                drop(children);

                let (move_idx, child_idx) = match best {
                    Some((mi, ci, _)) => (mi, ci),
                    None => {
                        let idx = Board::pos_to_idx(legal[0].0, legal[0].1);
                        (idx, node.children.read()[idx].unwrap())
                    }
                };

                // 虚拟损失（原子）
                self.node(child_idx).add_virtual_loss();
                node.add_virtual_loss();

                path.push((node_idx, move_idx, child_idx));
                sim_board.play_idx(move_idx);
                node_idx = child_idx;
            }
        }
    }

    // ── Evaluator：批量收集 + forward + 展开节点 ──

    fn evaluator_run(
        mcts: &MCTS,
        leaf_rx: crossbeam_channel::Receiver<LeafRequest>,
        network: &GomokuNetwork,
        device: &Device,
    ) {
        let mut batch: Vec<LeafRequest> = Vec::with_capacity(BATCH_CAP);

        loop {
            // 阻塞等待第一个请求
            match leaf_rx.recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }

            // 在 BATCH_CAP 内尽量多收
            while batch.len() < BATCH_CAP {
                match leaf_rx.recv_timeout(std::time::Duration::from_micros(200)) {
                    Ok(req) => batch.push(req),
                    Err(_) => break,
                }
            }

            let n = batch.len();
            let mut buf = vec![0.0f32; n * ENCODE_CHANNELS * NUM_POSITIONS];
            for (i, req) in batch.iter().enumerate() {
                let off = i * ENCODE_CHANNELS * NUM_POSITIONS;
                buf[off..off + ENCODE_CHANNELS * NUM_POSITIONS].copy_from_slice(&req.encoding);
            }

            let state = Tensor::<1>::from_floats(buf.as_slice(), device).reshape([
                n as i32,
                INPUT_CHANNELS as i32,
                BOARD_SIZE as i32,
                BOARD_SIZE as i32,
            ]);
            let (logits, values) = network.forward(state);
            let policy_flat: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap();
            let values_flat: Vec<f32> = values.into_data().to_vec::<f32>().unwrap();

            // 展开节点 + 分发结果
            for (i, req) in batch.drain(..).enumerate() {
                let leaf_node_idx = req.path.last().map(|&(_, _, c)| c).unwrap_or(0);

                let probs = Self::softmax_legal(
                    &policy_flat[i * POLICY_OUT..(i + 1) * POLICY_OUT],
                    &req.legal_moves_at_leaf,
                );

                mcts.expand_node(leaf_node_idx, &req.legal_moves_at_leaf, &probs, None);
                mcts.node(leaf_node_idx)
                    .expanded
                    .store(true, Ordering::Release);

                let _ = req.reply_tx.send(LeafReply {
                    value: values_flat[i],
                });
            }
        }
    }

    // ── 反向传播 ──

    fn backprop_path(&self, path: &[(usize, usize, usize)], mut value: f32) {
        for &(node_idx, _move_idx, child_idx) in path.iter().rev() {
            let child = self.node(child_idx);
            let parent = self.node(node_idx);
            child.remove_virtual_loss();
            parent.remove_virtual_loss();
            child.add_visit(value);
            value = -value;
        }
    }

    // ── 工具方法 ──

    #[inline]
    fn softmax_legal(logits: &[f32], legal: &[(usize, usize)]) -> Vec<f32> {
        let max_logit = legal
            .iter()
            .map(|&(r, c)| logits[Board::pos_to_idx(r, c)])
            .fold(f32::NEG_INFINITY, f32::max);

        let mut probs = vec![0.0f32; NUM_POSITIONS];
        let mut sum = 0.0f32;
        for &(r, c) in legal {
            let idx = Board::pos_to_idx(r, c);
            let exp = (logits[idx] - max_logit).exp();
            probs[idx] = exp;
            sum += exp;
        }

        if sum > 0.0 {
            for &(r, c) in legal {
                let idx = Board::pos_to_idx(r, c);
                probs[idx] /= sum;
            }
        } else {
            let count = legal.len() as f32;
            for &(r, c) in legal {
                probs[Board::pos_to_idx(r, c)] = 1.0 / count;
            }
        }
        probs
    }

    fn dirichlet_noise(n: usize, alpha: f32) -> Vec<f32> {
        use rand::distr::Distribution;
        let gamma = rand_distr::Gamma::new(alpha, 1.0).unwrap();
        let mut rng = rand::rng();
        let mut samples: Vec<f32> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
        let sum: f32 = samples.iter().sum();
        for s in &mut samples {
            *s /= sum;
        }
        samples
    }

    fn sample_with_temperature(probs: &[f32], temperature: f32) -> usize {
        use rand::distr::{Distribution, weighted::WeightedIndex};
        if temperature < 1e-5 {
            return probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
        let scaled: Vec<f64> = probs
            .iter()
            .map(|&p| (p as f64).powf(1.0 / temperature as f64))
            .collect();
        let dist = WeightedIndex::new(&scaled).unwrap();
        dist.sample(&mut rand::rng())
    }
}

// ============================================================
//  测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::Board;

    // ── Node 单元测试 ──

    #[test]
    fn test_node_new() {
        let n = Node::new(0.5);
        assert_eq!(n.visit_count_f32(), 0.0);
        assert_eq!(n.total_value_f32(), 0.0);
        assert_eq!(n.prior, 0.5);
        assert_eq!(n.virtual_loss.load(Ordering::Relaxed), 0);
        assert_eq!(n.children.read().len(), NUM_POSITIONS);
        assert!(!n.expanded.load(Ordering::Relaxed));
        assert!(n.children.read().iter().all(|c| c.is_none()));
    }

    #[test]
    fn test_node_q_zero_visits() {
        let n = Node::new(0.3);
        assert_eq!(n.q(), 0.0);
        assert_eq!(n.effective_q(), 0.0);
    }

    #[test]
    fn test_node_q_basic() {
        let n = Node::new(0.3);
        n.visit_count.store(10, Ordering::Relaxed);
        n.total_value.store(f2u(5.0), Ordering::Relaxed);
        assert_eq!(n.q(), 0.5);
        assert_eq!(n.effective_q(), 0.5);
    }

    #[test]
    fn test_virtual_loss_apply_remove() {
        let n = Node::new(0.3);
        n.visit_count.store(10, Ordering::Relaxed);
        n.total_value.store(f2u(5.0), Ordering::Relaxed);

        n.add_virtual_loss();
        assert_eq!(n.virtual_loss.load(Ordering::Relaxed), 1);
        let expected_val = 5.0 - VIRTUAL_LOSS;
        assert!((n.total_value_f32() - expected_val).abs() < 1e-6);
        assert!((n.effective_n() - 11.0).abs() < 1e-6);
        assert!((n.effective_q() - expected_val / 11.0).abs() < 1e-6);

        n.remove_virtual_loss();
        assert_eq!(n.virtual_loss.load(Ordering::Relaxed), 0);
        assert!((n.total_value_f32() - 5.0).abs() < 1e-6);
        assert!((n.effective_q() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_virtual_loss_multiple_threads() {
        let n = Node::new(0.3);
        n.visit_count.store(20, Ordering::Relaxed);
        n.total_value.store(f2u(10.0), Ordering::Relaxed);

        for _ in 0..3 {
            n.add_virtual_loss();
        }
        assert_eq!(n.virtual_loss.load(Ordering::Relaxed), 3);
        let expected_val = 10.0 - 3.0 * VIRTUAL_LOSS;
        assert!((n.total_value_f32() - expected_val).abs() < 1e-5);
        assert!((n.effective_q() - expected_val / 23.0).abs() < 1e-5);

        for _ in 0..3 {
            n.remove_virtual_loss();
        }
        assert_eq!(n.virtual_loss.load(Ordering::Relaxed), 0);
        assert!((n.total_value_f32() - 10.0).abs() < 1e-5);
        assert!((n.effective_q() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_add_visit_atomic() {
        let n = Node::new(0.3);
        n.add_visit(1.0);
        assert_eq!(n.visit_count_f32(), 1.0);
        assert!((n.total_value_f32() - 1.0).abs() < 1e-5);

        n.add_visit(-0.5);
        assert_eq!(n.visit_count_f32(), 2.0);
        assert!((n.total_value_f32() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_concurrent_visits_atomicity() {
        use std::sync::Arc;
        let node = Arc::new(Node::new(0.3));
        let num_threads = 8;
        let per_thread = 1000;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let n = Arc::clone(&node);
                std::thread::spawn(move || {
                    for _ in 0..per_thread {
                        n.add_visit(1.0);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let expected = (num_threads * per_thread) as f32;
        assert_eq!(node.visit_count_f32(), expected);
        assert!((node.total_value_f32() - expected).abs() < 1e-5);
    }

    // ── MCTS 方法测试 ──

    #[test]
    fn test_dirichlet_noise() {
        let noise = MCTS::dirichlet_noise(10, 0.3);
        assert_eq!(noise.len(), 10);
        let sum: f32 = noise.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        assert!(noise.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_dirichlet_noise_single() {
        let noise = MCTS::dirichlet_noise(1, 0.3);
        assert_eq!(noise.len(), 1);
        assert!((noise[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_with_temperature_deterministic() {
        let probs = vec![0.1, 0.5, 0.3, 0.1];
        let idx = MCTS::sample_with_temperature(&probs, 0.0);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_sample_with_temperature_uniform() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let idx = MCTS::sample_with_temperature(&probs, 100.0);
        assert!(idx < 4);
    }

    #[test]
    fn test_masked_softmax_all_legal() {
        let board = Board::new();
        let legal = board.legal_moves();
        let data = vec![0.0f32; NUM_POSITIONS];
        let probs = MCTS::softmax_legal(&data, &legal);
        assert_eq!(probs.len(), NUM_POSITIONS);
        let expected = 1.0 / NUM_POSITIONS as f32;
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-5,
                "idx {i}: expected {expected}, got {p}"
            );
        }
    }

    #[test]
    fn test_masked_softmax_partial_legal() {
        let mut board = Board::new();
        board.play(7, 7);
        board.play(7, 8);
        let legal = board.legal_moves();
        assert_eq!(legal.len(), NUM_POSITIONS - 2);

        let data = vec![1.0f32; NUM_POSITIONS];
        let probs = MCTS::softmax_legal(&data, &legal);
        let expected = 1.0 / legal.len() as f32;

        for (i, &p) in probs.iter().enumerate() {
            let pos = Board::idx_to_pos(i);
            if board.is_empty(pos.0, pos.1) {
                assert!(
                    (p - expected).abs() < 1e-5,
                    "idx {i}: expected {expected}, got {p}"
                );
            } else {
                assert_eq!(p, 0.0, "idx {i} should be masked");
            }
        }
    }

    #[test]
    fn test_expand_node() {
        let mcts = MCTS::new();
        let root = mcts.root_idx();
        let board = Board::new();
        let legal = board.legal_moves();
        let probs = vec![1.0 / NUM_POSITIONS as f32; NUM_POSITIONS];

        mcts.expand_node(root, &legal, &probs, None);
        let root_node = mcts.node(root);

        assert!(
            root_node
                .children
                .read()
                .iter()
                .filter(|c| c.is_some())
                .count()
                == NUM_POSITIONS
        );
        if let Some(ci) = root_node.children.read()[Board::pos_to_idx(7, 7)] {
            let child = mcts.node(ci);
            assert!((child.prior - 1.0 / NUM_POSITIONS as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_expand_node_with_noise() {
        let mcts = MCTS::new();
        let root = mcts.root_idx();
        let board = Board::new();
        let legal = board.legal_moves();
        let probs = vec![1.0 / NUM_POSITIONS as f32; NUM_POSITIONS];
        let noise = MCTS::dirichlet_noise(legal.len(), DIRICHLET_ALPHA);

        mcts.expand_node(root, &legal, &probs, Some((&noise, DIRICHLET_EPSILON)));
        let root_node = mcts.node(root);
        if let Some(ci) = root_node.children.read()[Board::pos_to_idx(7, 7)] {
            let child = mcts.node(ci);
            assert!(child.prior > 0.0 && child.prior < 1.0);
        }
    }

    #[test]
    fn test_mcts_reset() {
        let mcts = MCTS::new();
        assert_eq!(mcts.arena.count(), 1); // 根节点已预创建
        assert_eq!(mcts.root_idx(), 0);
        assert_eq!(mcts.node(0).visit_count_f32(), 0.0);
    }

    #[test]
    fn test_simulate_game_over() {
        let mcts = MCTS::new();
        let root = mcts.root_idx();
        assert_eq!(mcts.node(root).visit_count_f32(), 0.0);
    }

    #[test]
    fn test_backprop_visit_count_consistency() {
        let mcts = MCTS::new();
        let root = mcts.root_idx();
        let board = Board::new();
        let legal = board.legal_moves();
        let probs = vec![1.0 / NUM_POSITIONS as f32; NUM_POSITIONS];
        mcts.expand_node(root, &legal, &probs, None);

        let children: Vec<usize> = mcts
            .node(root)
            .children
            .read()
            .iter()
            .take(3)
            .filter_map(|&c| c)
            .collect();

        for &ci in &children {
            let path = vec![(0, 0, ci)];
            mcts.backprop_path(&path, 1.0);
        }

        let sum_n: f32 = mcts
            .node(root)
            .children
            .read()
            .iter()
            .filter_map(|&c| c.map(|ci| mcts.node(ci).visit_count_f32()))
            .sum();
        assert_eq!(sum_n, 3.0, "children visit_count should sum to num_sims");

        for &ci in &children {
            assert_eq!(mcts.node(ci).visit_count_f32(), 1.0);
        }

        for &ci in &children {
            let policy = mcts.node(ci).visit_count_f32() / sum_n;
            assert!((policy - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_backprop_multilevel_consistency() {
        let mcts = MCTS::new();
        let root = mcts.root_idx();
        let board = Board::new();
        let legal = board.legal_moves();
        let probs = vec![1.0 / NUM_POSITIONS as f32; NUM_POSITIONS];

        mcts.expand_node(root, &legal, &probs, None);
        let first_legal_idx = Board::pos_to_idx(legal[0].0, legal[0].1);
        let child_a = mcts.node(root).children.read()[first_legal_idx].unwrap();

        let mut board_a = board.clone();
        board_a.play_idx(first_legal_idx);
        let legal_a = board_a.legal_moves();
        mcts.node(child_a).expanded.store(true, Ordering::Release);
        mcts.expand_node(child_a, &legal_a, &probs, None);
        let a_first_legal_idx = Board::pos_to_idx(legal_a[0].0, legal_a[0].1);
        let gc = mcts.node(child_a).children.read()[a_first_legal_idx].unwrap();

        let path = vec![
            (0, first_legal_idx, child_a),
            (child_a, a_first_legal_idx, gc),
        ];
        mcts.backprop_path(&path, 1.0);

        assert_eq!(mcts.node(child_a).visit_count_f32(), 1.0);
        assert_eq!(mcts.node(gc).visit_count_f32(), 1.0);

        let sum_n: f32 = mcts
            .node(root)
            .children
            .read()
            .iter()
            .filter_map(|&c| c.map(|ci| mcts.node(ci).visit_count_f32()))
            .sum();
        assert_eq!(sum_n, 1.0);
    }
}
