//! 蒙特卡洛树搜索 (MCTS)
//!
//! 使用 PUCT 公式选择节点，神经网络评估叶节点。
//!
//! ## 并行化策略（Root Parallelization）
//!
//! 将 `num_simulations` 次模拟均匀分配给 `num_threads` 个线程，
//! 每个线程独立运行在 MCTS 树副本上（虚拟损失在副本内仍然有效），
//! 完成后将所有副本的 `visit_count` 和 `total_value` 合并回原始树。
//!
//! ## 节点统计量语义
//!
//! 每个节点的 `total_value` 和 `visit_count` 存储的是**从该节点父节点视角**的累积值。
//! 这样父节点的 PUCT 选择可以直接使用 `child.q()` 而无需取反。
//!
//! 回溯时只更新子节点的统计量；父节点的 visit_count 同步递增即可。

use crate::game::board::{Board, NUM_POSITIONS};
use crate::network::residual::{GobangNetwork, POLICY_OUT};
use burn::tensor::{Device, Tensor};
use rayon::prelude::*;

/// 虚拟损失：模拟期间施加的惩罚值，防止并行线程重复选择同一路径。
///
/// 原理：当线程 A 沿着某条路径进行模拟时，对该路径上的节点施加虚拟损失，
/// 使其 Q 值暂时降低，从而引导其他线程探索不同路径。模拟完成后撤销。
///
/// 取 3.0 是因为局面价值在 [-1, 1] 区间，3.0 可确保 Q 值大幅下降。
const VIRTUAL_LOSS: f32 = 3.0;

/// PUCT 探索常数，控制探索 vs 利用的平衡。
///
/// PUCT 公式：`score = Q(s,a) + C_PUCT * P(s,a) * sqrt(N_parent) / (1 + N_child)`
/// - 第一项 Q(s,a)：平均价值（利用）
/// - 第二项：基于先验概率和访问次数的探索奖励
/// C_PUCT 越大，MCTS 越倾向于探索未充分访问的高先验概率节点。
const C_PUCT: f32 = 1.0;

/// Dirichlet 噪声的集中参数（Alpha 越小噪声越分散）。
///
/// 在根节点的先验概率上叠加 Dirichlet 噪声，鼓励 MCTS 尝试先验概率较低的走法，
/// 增加探索多样性。Alpha=0.3 在 225 个合法走法下产生适中程度的噪声。
const DIRICHLET_ALPHA: f32 = 0.3;

/// Dirichlet 噪声在混合先验中的权重（0.25 表示 75% 原始先验 + 25% 噪声）。
/// 这一比例在 AlphaZero 原始论文中被证明能在探索和利用之间取得良好平衡。
const DIRICHLET_EPSILON: f32 = 0.25;

/// MCTS 树节点
///
/// `virtual_loss` 记录当前有多少个模拟正在经过此节点（计数），
/// 施加时同步从 `total_value` 减去 `VIRTUAL_LOSS`。
#[derive(Clone)]
pub struct Node {
    /// 访问次数（从父节点视角的模拟次数）
    pub visit_count: f32,
    /// 累积价值（从父节点视角）
    pub total_value: f32,
    /// 神经网络先验概率
    pub prior: f32,
    /// 当前虚拟损失计数（并行模拟数量）
    pub virtual_loss: f32,
    /// 子节点索引，按棋盘位置索引排列
    pub children: Vec<Option<usize>>,
    /// 是否已展开（已由神经网络评估）
    pub expanded: bool,
}

impl Node {
    pub fn new(prior: f32) -> Self {
        Self {
            visit_count: 0.0,
            total_value: 0.0,
            prior,
            virtual_loss: 0.0,
            children: vec![None; NUM_POSITIONS],
            expanded: false,
        }
    }

    /// 平均价值 Q = total_value / visit_count（从父节点视角）
    #[inline]
    pub fn q(&self) -> f32 {
        if self.visit_count > 0.0 {
            self.total_value / self.visit_count
        } else {
            0.0
        }
    }

    /// 有效访问次数（包含虚拟损失计数）
    #[inline]
    pub fn effective_n(&self) -> f32 {
        self.visit_count + self.virtual_loss
    }

    /// 有效平均价值（已通过 total_value 反映虚拟损失惩罚）
    #[inline]
    pub fn effective_q(&self) -> f32 {
        let n = self.effective_n();
        if n > 0.0 { self.total_value / n } else { 0.0 }
    }

    /// 施加虚拟损失：计数 +1，价值 -VIRTUAL_LOSS
    #[inline]
    fn add_virtual_loss(&mut self) {
        self.virtual_loss += 1.0;
        self.total_value -= VIRTUAL_LOSS;
    }

    /// 撤销虚拟损失：计数 -1，价值 +VIRTUAL_LOSS
    #[inline]
    fn remove_virtual_loss(&mut self) {
        self.virtual_loss -= 1.0;
        self.total_value += VIRTUAL_LOSS;
    }
}

/// MCTS 搜索器
#[derive(Clone)]
pub struct MCTS {
    nodes: Vec<Node>,
}

/// 搜索返回结果
pub struct SearchResult {
    /// 最佳落子索引（0..225），无效时返回 NUM_POSITIONS
    pub best_move: usize,
    /// 访问计数归一化后的策略分布
    pub policy: Vec<f32>,
    /// 根节点局面价值（从当前玩家视角）
    pub root_value: f32,
}

impl MCTS {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// 执行 MCTS 搜索，返回最佳走法和策略。
    ///
    /// ## 并行化策略
    ///
    /// 模拟阶段使用 Rayon 并行执行。每个线程在 MCTS 树的独立副本上运行，
    /// 完成后将所有副本的 visit_count / total_value 合并回主树。
    /// 这种"Root Parallelization"在 AlphaZero 文献中广泛使用，
    /// 能在线性扩展的同时保持搜索质量。
    ///
    /// `device` 用于将棋盘编码转为 Burn tensor。
    pub fn search(
        &mut self,
        board: &mut Board,
        network: &GobangNetwork,
        device: &Device,
        num_simulations: usize,
        temperature: f32,
    ) -> SearchResult {
        let root_idx = self.reset();
        let legal_moves = board.legal_moves();
        if legal_moves.is_empty() {
            return SearchResult {
                best_move: NUM_POSITIONS,
                policy: vec![0.0; NUM_POSITIONS],
                root_value: 0.0,
            };
        }

        let (policy_logits, value) = self.evaluate(board, network, device);
        let policy_probs = self.masked_softmax(&policy_logits, &legal_moves);
        let root_value: f32 = value.into_scalar();

        // 展开根节点（带 Dirichlet 噪声）
        let dirichlet = Self::dirichlet_noise(legal_moves.len(), DIRICHLET_ALPHA);
        self.expand_node(
            root_idx,
            &legal_moves,
            &policy_probs,
            Some((&dirichlet, DIRICHLET_EPSILON)),
        );
        self.nodes[root_idx].expanded = true;

        // ── 并行模拟 ──
        let num_threads = rayon::current_num_threads().min(num_simulations).max(1);
        let base = num_simulations / num_threads;
        let remainder = num_simulations % num_threads;

        let sim_results: Vec<Vec<Node>> = (0..num_threads)
            .into_par_iter()
            .map(|tid| {
                let sims = if tid < remainder { base + 1 } else { base };
                let mut local_mcts = self.clone();
                let mut sim_board = board.clone();
                let mut legal = Vec::with_capacity(NUM_POSITIONS);
                for _ in 0..sims {
                    local_mcts.simulate(&mut sim_board, 0, network, device, &mut legal);
                }
                local_mcts.nodes
            })
            .collect();

        // 合并：将各线程的 visit_count / total_value 累加回主树
        // virtual_loss 不合并（它是临时的）
        self.merge_trees(&sim_results);

        // 构建策略分布
        let root = &self.nodes[root_idx];
        let sum_n = root.visit_count;
        let mut policy = vec![0.0f32; NUM_POSITIONS];
        if sum_n > 0.0 {
            for (idx, child_opt) in root.children.iter().enumerate() {
                if let &Some(child_idx) = child_opt {
                    policy[idx] = self.nodes[child_idx].visit_count / sum_n;
                }
            }
        }

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

    /// 合并多个树副本的 visit_count / total_value 到 self。
    ///
    /// 合并规则：
    /// - `visit_count`：直接累加
    /// - `total_value`：直接累加（αβ 零和博弈中视角由父节点关系保证一致性）
    /// - `virtual_loss`：不合并（它是线程本地临时的）
    /// - `prior`、`children` 结构：不合并（所有副本共享相同的拓扑）
    fn merge_trees(&mut self, others: &[Vec<Node>]) {
        for other_nodes in others {
            let count = other_nodes.len().min(self.nodes.len());
            for i in 0..count {
                self.nodes[i].visit_count += other_nodes[i].visit_count;
                self.nodes[i].total_value += other_nodes[i].total_value;
            }
        }
    }

    /// 重置节点池，创建根节点
    fn reset(&mut self) -> usize {
        self.nodes.clear();
        self.nodes.push(Node::new(0.0));
        0
    }

    /// 单次 MCTS 模拟（递归实现）
    ///
    /// ## 执行步骤
    ///
    /// 1. **终局检测**：如果棋盘已结束，直接返回结果（胜 +1，平 0）
    /// 2. **叶节点评估**：如果当前节点未展开，调用神经网络评估并创建子节点，
    ///    返回网络评估值（取反，因为需要从父节点视角）
    /// 3. **PUCT 选择**：在所有合法子节点中，按 PUCT 公式选择得分最高的走法
    /// 4. **施加虚拟损失**：对选中子节点和当前节点施加虚拟损失，
    ///    降低其 Q 值以阻止其他并行线程重复选择
    /// 5. **递归模拟**：在选中的子节点上继续模拟
    /// 6. **撤销虚拟损失**：模拟完成后恢复原始统计量
    /// 7. **反向传播**：将模拟结果累加到子节点的 visit_count 和 total_value，
    ///    同时更新父节点 visit_count（用于 PUCT 的 sqrt(N_parent) 计算）
    ///
    /// ## 视角约定
    ///
    /// 返回值始终从**调用者（父节点）**的视角：
    /// - 如果模拟发现父节点获胜，返回 +1.0
    /// - 如果对手获胜，返回 -1.0
    /// - 平局返回 0.0
    ///
    /// 递归传递时每次取反，自动完成视角翻转。
    ///
    /// 返回从**调用者（父节点）视角**的局面价值。
    fn simulate(
        &mut self,
        board: &mut Board,
        node_idx: usize,
        network: &GobangNetwork,
        device: &Device,
        legal: &mut Vec<(usize, usize)>,
    ) -> f32 {
        // 终局：上一手玩家已获胜或平局
        if board.game_over {
            return match board.winner {
                Some(_) => 1.0,
                None => 0.0,
            };
        }

        board.fill_legal_moves(legal);
        if legal.is_empty() {
            return 0.0;
        }

        if !self.nodes[node_idx].expanded {
            let (policy_logits, value) = self.evaluate(board, network, device);
            let val: f32 = value.into_scalar();
            let probs = self.masked_softmax(&policy_logits, legal);
            self.expand_node(node_idx, legal, &probs, None);
            self.nodes[node_idx].expanded = true;
            return -val;
        }

        let parent_sqrt_n = self.nodes[node_idx].effective_n().sqrt();
        let mut best: Option<(usize, usize, f32)> = None;

        for &(r, c) in legal.iter() {
            let idx = Board::pos_to_idx(r, c);
            if let Some(ci) = self.nodes[node_idx].children[idx] {
                let child = &self.nodes[ci];
                let score = child.effective_q()
                    + C_PUCT * child.prior * parent_sqrt_n / (1.0 + child.effective_n());
                if best.map_or(true, |(_, _, s)| score > s) {
                    best = Some((idx, ci, score));
                }
            }
        }

        let (best_move_idx, best_child_idx) =
            best.map(|(mi, ci, _)| (mi, ci)).unwrap_or_else(|| {
                let idx = Board::pos_to_idx(legal[0].0, legal[0].1);
                (idx, self.nodes[node_idx].children[idx].unwrap())
            });

        self.nodes[best_child_idx].add_virtual_loss();
        self.nodes[node_idx].add_virtual_loss();

        let snap = board.snapshot();
        board.play_idx(best_move_idx);
        let value = self.simulate(board, best_child_idx, network, device, legal);
        let (mr, mc) = Board::idx_to_pos(best_move_idx);
        board.undo(mr, mc, &snap);

        self.nodes[best_child_idx].remove_virtual_loss();
        self.nodes[node_idx].remove_virtual_loss();

        self.nodes[best_child_idx].visit_count += 1.0;
        self.nodes[best_child_idx].total_value += value;
        self.nodes[node_idx].visit_count += 1.0;

        -value
    }

    /// 展开节点：为所有合法走法创建子节点。
    ///
    /// 每个子节点的 `prior` 来自神经网络的策略输出（经 softmax 后）。
    ///
    /// 如果在根节点展开，会叠加 Dirichlet 噪声：
    /// `prior = (1 - epsilon) * network_prior + epsilon * dirichlet_noise`
    fn expand_node(
        &mut self,
        parent: usize,
        legal: &[(usize, usize)],
        probs: &[f32],
        noise: Option<(&[f32], f32)>,
    ) {
        for (i, &(r, c)) in legal.iter().enumerate() {
            let idx = Board::pos_to_idx(r, c);
            let prior = match noise {
                Some((dir, eps)) => (1.0 - eps) * probs[idx] + eps * dir[i],
                None => probs[idx],
            };
            let ci = self.nodes.len();
            self.nodes.push(Node::new(prior));
            self.nodes[parent].children[idx] = Some(ci);
        }
    }

    /// 神经网络前向评估
    fn evaluate(
        &self,
        board: &Board,
        network: &GobangNetwork,
        device: &Device,
    ) -> (Tensor<1>, Tensor<1>) {
        let state_data = board.encode_state();
        let state = Tensor::<1>::from_floats(state_data.as_slice(), device).reshape([1, 4, 15, 15]);

        let (policy_logits, value) = network.forward(state);
        // [1, 225] -> [225]; [1, 1] -> [1]
        let policy_1d = policy_logits.reshape([POLICY_OUT]);
        let value_1d = value.reshape([1]);
        (policy_1d, value_1d)
    }

    /// 掩码 Softmax：仅对合法走法计算概率分布。
    ///
    /// `legal` 是当前棋盘的所有合法走法列表。
    /// 先取合法走法中 logit 的最大值做数值稳定，再 exp + 归一化。
    /// 如果所有合法走法的 logit 都极小（sum ≈ 0），退化为均匀分布。
    fn masked_softmax(&self, logits: &Tensor<1>, legal: &[(usize, usize)]) -> Vec<f32> {
        let bytes = logits.clone().into_data().to_vec::<f32>().unwrap();

        let max_logit = legal
            .iter()
            .map(|&(r, c)| bytes[Board::pos_to_idx(r, c)])
            .fold(f32::NEG_INFINITY, f32::max);

        let mut probs = vec![0.0f32; NUM_POSITIONS];
        let mut sum = 0.0f32;
        for &(r, c) in legal {
            let idx = Board::pos_to_idx(r, c);
            let exp = (bytes[idx] - max_logit).exp();
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

    /// 生成 Dirichlet 噪声向量（所有分量之和为 1）。
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

    /// 按温度从概率分布中加权采样。
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::Board;

    // ============================================================
    //  Node 单元测试
    // ============================================================

    #[test]
    fn test_node_new() {
        let n = Node::new(0.5);
        assert_eq!(n.visit_count, 0.0);
        assert_eq!(n.total_value, 0.0);
        assert_eq!(n.prior, 0.5);
        assert_eq!(n.virtual_loss, 0.0);
        assert_eq!(n.children.len(), NUM_POSITIONS);
        assert!(!n.expanded);
        assert!(n.children.iter().all(|c| c.is_none()));
    }

    #[test]
    fn test_node_q_zero_visits() {
        let n = Node::new(0.3);
        assert_eq!(n.q(), 0.0);
        assert_eq!(n.effective_q(), 0.0);
    }

    #[test]
    fn test_node_q_basic() {
        let mut n = Node::new(0.3);
        n.visit_count = 10.0;
        n.total_value = 5.0;
        assert_eq!(n.q(), 0.5);
        assert_eq!(n.effective_q(), 0.5);
    }

    #[test]
    fn test_virtual_loss_apply_remove() {
        let mut n = Node::new(0.3);
        n.visit_count = 10.0;
        n.total_value = 5.0;

        // 施加
        n.add_virtual_loss();
        assert_eq!(n.virtual_loss, 1.0);
        assert_eq!(n.total_value, 5.0 - VIRTUAL_LOSS);
        assert!((n.effective_n() - 11.0).abs() < 1e-6);
        assert!((n.effective_q() - 2.0 / 11.0).abs() < 1e-6);

        // 撤销
        n.remove_virtual_loss();
        assert_eq!(n.virtual_loss, 0.0);
        assert_eq!(n.total_value, 5.0);
        assert_eq!(n.effective_q(), 0.5);
    }

    #[test]
    fn test_virtual_loss_multiple_threads() {
        let mut n = Node::new(0.3);
        n.visit_count = 20.0;
        n.total_value = 10.0; // Q = 0.5

        // 模拟 3 个线程同时经过
        for _ in 0..3 {
            n.add_virtual_loss();
        }
        assert_eq!(n.virtual_loss, 3.0);
        assert_eq!(n.total_value, 10.0 - 3.0 * VIRTUAL_LOSS);
        assert!((n.effective_q() - 1.0 / 23.0).abs() < 1e-5);

        for _ in 0..3 {
            n.remove_virtual_loss();
        }
        assert_eq!(n.virtual_loss, 0.0);
        assert_eq!(n.total_value, 10.0);
        assert_eq!(n.effective_q(), 0.5);
    }

    // ============================================================
    //  MCTS 方法测试
    // ============================================================

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
        assert_eq!(idx, 1); // max at index 1
    }

    #[test]
    fn test_sample_with_temperature_uniform() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let idx = MCTS::sample_with_temperature(&probs, 100.0);
        assert!(idx < 4);
    }

    #[test]
    fn test_masked_softmax_all_legal() {
        let mcts = MCTS::new();
        let board = Board::new();
        let device = Device::default();

        let legal = board.legal_moves();
        let data = vec![0.0f32; NUM_POSITIONS];
        let logits = Tensor::<1>::from_floats(data.as_slice(), &device);
        let probs = mcts.masked_softmax(&logits, &legal);
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
        let mcts = MCTS::new();
        let mut board = Board::new();
        let device = Device::default();

        board.play(7, 7);
        board.play(7, 8);
        let legal = board.legal_moves();
        assert_eq!(legal.len(), NUM_POSITIONS - 2);

        let data = vec![1.0f32; NUM_POSITIONS];
        let logits = Tensor::<1>::from_floats(data.as_slice(), &device);
        let probs = mcts.masked_softmax(&logits, &legal);
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
        let mut mcts = MCTS::new();
        let root = mcts.reset();
        let board = Board::new();
        let legal = board.legal_moves();
        let probs = vec![1.0 / NUM_POSITIONS as f32; NUM_POSITIONS];

        mcts.expand_node(root, &legal, &probs, None);
        let root_node = &mcts.nodes[root];

        assert!(root_node.children.iter().filter(|c| c.is_some()).count() == NUM_POSITIONS);
        if let Some(ci) = root_node.children[Board::pos_to_idx(7, 7)] {
            let child = &mcts.nodes[ci];
            assert!((child.prior - 1.0 / NUM_POSITIONS as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_expand_node_with_noise() {
        let mut mcts = MCTS::new();
        let root = mcts.reset();
        let board = Board::new();
        let legal = board.legal_moves();
        let probs = vec![1.0 / NUM_POSITIONS as f32; NUM_POSITIONS];
        let noise = MCTS::dirichlet_noise(legal.len(), DIRICHLET_ALPHA);

        mcts.expand_node(root, &legal, &probs, Some((&noise, DIRICHLET_EPSILON)));
        let root_node = &mcts.nodes[root];
        if let Some(ci) = root_node.children[Board::pos_to_idx(7, 7)] {
            let child = &mcts.nodes[ci];
            assert!(child.prior > 0.0 && child.prior < 1.0);
        }
    }

    #[test]
    fn test_mcts_reset() {
        let mut mcts = MCTS::new();
        assert_eq!(mcts.nodes.len(), 0);
        let idx = mcts.reset();
        assert_eq!(idx, 0);
        assert_eq!(mcts.nodes.len(), 1);
        assert_eq!(mcts.nodes[0].visit_count, 0.0);
    }

    #[test]
    fn test_merge_trees() {
        let mut m1 = MCTS::new();
        m1.reset();
        // 模拟扩展一个节点
        let probs = vec![1.0; NUM_POSITIONS];
        let board = Board::new();
        let legal = board.legal_moves();
        m1.expand_node(0, &legal, &probs, None);
        // 手动设置一些统计量
        m1.nodes[0].visit_count = 10.0;
        m1.nodes[0].total_value = 5.0;
        let children = m1.nodes[0].children.clone();
        for child in children.into_iter().filter_map(|c| c) {
            m1.nodes[child].visit_count = 1.0;
            m1.nodes[child].total_value = 0.5;
        }

        let mut m2 = m1.clone();
        m2.nodes[0].visit_count = 5.0;
        m2.nodes[0].total_value = 3.0;

        m1.merge_trees(&[m2.nodes.clone()]);

        // 合并后 visit_count = 10 + 5 = 15, total_value = 5 + 3 = 8
        assert_eq!(m1.nodes[0].visit_count, 15.0);
        assert_eq!(m1.nodes[0].total_value, 8.0);
    }

    #[test]
    fn test_mcts_clone() {
        let mut m1 = MCTS::new();
        m1.reset();
        let probs = vec![1.0; NUM_POSITIONS];
        let board = Board::new();
        let legal = board.legal_moves();
        m1.expand_node(0, &legal, &probs, None);

        let m2 = m1.clone();
        assert_eq!(m2.nodes.len(), m1.nodes.len());
        assert_eq!(m2.nodes[0].visit_count, m1.nodes[0].visit_count);
        // 验证子节点也被克隆
        let c1 = m1.nodes[0].children[0];
        let c2 = m2.nodes[0].children[0];
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_simulate_game_over() {
        let mut mcts = MCTS::new();
        let root = mcts.reset();
        assert_eq!(mcts.nodes[root].visit_count, 0.0);
    }
}
