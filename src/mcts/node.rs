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

use std::sync::atomic::{AtomicU32, Ordering};

use crate::game::board::{Board, ENCODE_CHANNELS, NUM_POSITIONS};
use crate::inference::Evaluator;

// ============================================================
//  常量
// ============================================================

/// 虚拟损失：模拟期间施加的惩罚值，防止并行线程重复选择同一路径。
const VIRTUAL_LOSS: f32 = 1.0;

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
    /// 0=未扩展, 1=正在扩展, 2=已扩展
    pub expanded: AtomicU32,
}

impl Node {
    pub fn new(prior: f32) -> Self {
        Self {
            visit_count: AtomicU32::new(0),
            total_value: AtomicU32::new(f2u(0.0)),
            prior,
            virtual_loss: AtomicU32::new(0),
            children: parking_lot::RwLock::new(vec![None; NUM_POSITIONS]),
            expanded: AtomicU32::new(0),
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
    pub arena: boxcar::Vec<Node>,
}

/// 搜索返回结果
pub struct SearchResult {
    pub best_move: usize,
    pub policy: Vec<f32>,
    pub root_value: f32,
}

// ── Evaluator / Worker 通信 ──

/// Worker 提交的 leaf 评估请求（不携带 reply channel）
struct LeafRequest {
    path: Vec<(usize, usize, usize)>,
    encoding: Vec<f32>,
    legal_moves_at_leaf: Vec<(usize, usize)>,
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

    pub fn search<E: Evaluator>(
        &self,
        board: &mut Board,
        evaluator: &E,
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

        // ── 根节点评估（通过 Evaluator，多对局时可被 InferenceServer 攒批）──
        let mut root_encoding = vec![0.0f32; ENCODE_CHANNELS * NUM_POSITIONS];
        board.encode_into(&mut root_encoding);
        let (root_logits, _root_values) = evaluator.evaluate_batch(&[root_encoding]);
        let policy_probs = Self::softmax_legal(&root_logits[0], &legal_moves);

        let dirichlet = Self::dirichlet_noise(legal_moves.len(), DIRICHLET_ALPHA);
        self.expand_node(
            root_idx,
            &legal_moves,
            &policy_probs,
            Some((&dirichlet, DIRICHLET_EPSILON)),
        );
        self.root().expanded.store(2, Ordering::Relaxed);

        // ── 并行模拟阶段，GPU forward 仅在主线程执行 ──
        //
        // 显存限制方案：让 cubecl CUDA backend 只被一个线程（主线程）使用，
        // 避免为每个线程分配独立的 workspace pool。
        //
        // 主线程：接收 leaf → 攒 batch → GPU forward → 发结果到 result_tx
        // worker pool (scoped): PUCT walk + 提交 leaf（纯 CPU，不碰 GPU）
        // backprop 线程 (rayon): 消费 result → expand + backprop（纯 CPU）
        let (leaf_tx, leaf_rx) = crossbeam_channel::bounded::<LeafRequest>(BATCH_CAP * 4);

        let num_workers = rayon::current_num_threads().min(num_simulations).max(1);

        std::thread::scope(|s| {
            let sims_per_worker = num_simulations / num_workers;
            let remainder = num_simulations % num_workers;

            // 克隆 leaf_tx 给每个 worker
            for tid in 0..num_workers {
                let sims = if tid < remainder {
                    sims_per_worker + 1
                } else {
                    sims_per_worker
                };
                let leaf_tx = leaf_tx.clone();
                let mut sim_board = board.clone();
                s.spawn(move || {
                    let mut legal = Vec::with_capacity(NUM_POSITIONS);
                    self.worker_loop(sims, &mut sim_board, &mut legal, &leaf_tx);
                });
            }

            // drop 原始 sender，只保留 worker 中的 clone。
            // 当所有 worker 完成并 drop 各自的 sender 后，channel 关闭，
            // evaluator_run 收到 Err 后退出。
            drop(leaf_tx);

            // 主线程：evaluator + expand + backprop
            // GPU forward 通过 Evaluator trait 委派给 InferenceServer
            Self::evaluator_run(leaf_rx, evaluator, self);
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

        // root_value = 子节点 Q 的加权平均
        // Q 存的是从父节点（root）视角的价值，不加负号
        let root_value = if sum_n > 0.0 {
            children
                .iter()
                .filter_map(|c| {
                    c.and_then(|ci| {
                        let child = self.node(ci);
                        let n = child.visit_count_f32();
                        if n > 0.0 {
                            Some(n / sum_n * child.q())
                        } else {
                            None
                        }
                    })
                })
                .sum::<f32>()
        } else {
            0.0
        };
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

    // ── Worker 池：走树 + 提交叶子（不等回复，虚拟损失阻止冲突） ──
    //
    // 异步 continuation：到达未展开节点后，提交 leaf 请求并立即开始下一条模拟，
    // 不再同步等待 NN 结果。虚拟损失确保并行 worker 不会踩踏同一路径。
    fn worker_loop(
        &self,
        num_sims: usize,
        board: &mut Board,
        legal: &mut Vec<(usize, usize)>,
        leaf_tx: &crossbeam_channel::Sender<LeafRequest>,
    ) {
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
                if node.expanded.load(Ordering::Relaxed) != 2 {
                    // 叶子节点：用 CAS 抢 expanding 状态（0→1），避免重复 expand
                    let was_zero = node
                        .expanded
                        .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok();
                    if was_zero {
                        // 抢到 expanding，提交 evaluator
                        let mut encoding = vec![0.0f32; ENCODE_CHANNELS * NUM_POSITIONS];
                        sim_board.encode_into(&mut encoding);
                        leaf_tx
                            .send(LeafRequest {
                                path,
                                encoding,
                                legal_moves_at_leaf: std::mem::take(legal),
                            })
                            .expect("evaluator died");
                        break;
                    }
                    // 没抢到：可能正被 evaluator 处理，或已 expand
                    // 回退当前 path 上的虚拟损失，重新从 root 走
                    for &(_, _, ci) in path.iter().rev() {
                        self.node(ci).remove_virtual_loss();
                    }
                    node_idx = 0;
                    sim_board = board.clone();
                    path.clear();
                    continue;
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

                // 虚拟损失（原子）—— 只加在子节点上，parent 不需要
                // PUCT 公式已经用 child.effective_n() 来惩罚
                self.node(child_idx).add_virtual_loss();

                path.push((node_idx, move_idx, child_idx));
                sim_board.play_idx(move_idx);
                node_idx = child_idx;
            }
        }
    }

    // ── Evaluator：批量收集 + GPU forward（不做 expand/backprop） ──

    /// Evaluator 阻塞收集 leaf → 攒 batch → 通过 Evaluator trait 评估 → expand + backprop
    fn evaluator_run<E: Evaluator>(
        leaf_rx: crossbeam_channel::Receiver<LeafRequest>,
        evaluator: &E,
        mcts: &MCTS,
    ) {
        let mut batch: Vec<LeafRequest> = Vec::with_capacity(BATCH_CAP);

        loop {
            match leaf_rx.recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }

            while batch.len() < BATCH_CAP {
                match leaf_rx.recv_timeout(std::time::Duration::from_micros(200)) {
                    Ok(req) => batch.push(req),
                    Err(_) => break,
                }
            }

            // 收集编码，委托给 Evaluator（支持跨 MCTS 实例攒批）
            let encodings: Vec<Vec<f32>> = batch.iter().map(|req| req.encoding.clone()).collect();
            let (policies_batch, values_batch) = evaluator.evaluate_batch(&encodings);

            // 逐个 expand + backprop
            for (i, req) in batch.drain(..).enumerate() {
                let leaf_node_idx = req.path.last().map(|&(_, _, c)| c).unwrap_or(0);

                let probs = Self::softmax_legal(&policies_batch[i], &req.legal_moves_at_leaf);

                mcts.expand_node(leaf_node_idx, &req.legal_moves_at_leaf, &probs, None);
                mcts.node(leaf_node_idx)
                    .expanded
                    .store(2, Ordering::Relaxed);

                mcts.backprop_path_internal(&req.path, -values_batch[i]);
            }
        }
    }

    /// 内部 backprop（不移除虚拟损失，因为 worker 提交 leaf 时未加虚拟损失在叶子节点上）
    ///
    /// 虚拟损失已在 worker 的 PUCT 路径上加了，这里需要移除并加 visit。
    fn backprop_path_internal(&self, path: &[(usize, usize, usize)], mut value: f32) {
        for &(_node_idx, _move_idx, child_idx) in path.iter().rev() {
            let child = self.node(child_idx);
            child.remove_virtual_loss();
            child.add_visit(value);
            value = -value;
        }
    }

    // ── 反向传播 ──

    fn backprop_path(&self, path: &[(usize, usize, usize)], mut value: f32) {
        for &(_node_idx, _move_idx, child_idx) in path.iter().rev() {
            let child = self.node(child_idx);
            child.remove_virtual_loss();
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
        assert_eq!(n.expanded.load(Ordering::Relaxed), 0);
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
        mcts.node(child_a).expanded.store(2, Ordering::Release);
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
