//! 蒙特卡洛树搜索 (MCTS) — Gumbel AlphaZero
//!
//! 算法对齐 LightZero `ctree_gumbel_alphazero`（C++ 参考实现）：
//! - 根节点选择：Gumbel 噪声 + Sequential Halving (`_select_root_child`)
//! - 内部节点选择：`priors + completed_Q - visit_penalty` (`_select_interior_child`)
//! - Completed Q：`_qtransform_completed_by_mix_value` → rescale → visit_scale * value_scale
//! - 反向传播沿 parent 链自行翻转视角，不再依赖 PUCT walk 构建 path
//!
//! ## 游戏解耦
//!
//! MCTS 通过 `Game` trait 与具体棋盘解耦。所有游戏操作（合法动作、落子、
//! 编码、终局检测）均通过 trait 调用。
//!
//! ## 噪声模式
//!
//! - Gumbel 噪声始终用于根节点选择（`_score_considered`）。
//! - Dirichlet 噪声可选叠加到根子节点的 `prior` 上（由 `add_dirichlet_noise` 控制）。

use super::game::{ActionId, Game};
use crate::inference::Evaluator;
use ndarray::Array2;
use rand::RngExt;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use std::collections::HashMap;
use std::marker::PhantomData;

// ============================================================
//  GumbelConfig
// ============================================================

#[derive(Debug, Clone)]
pub struct GumbelConfig {
    /// 模拟次数
    pub num_simulations: usize,
    /// Sequential Halving 初始考虑的动作数（对齐 LightZero `max_num_considered_actions`）
    pub max_num_considered_actions: usize,
    /// 访问数基础偏移（对齐 `maxvisit_init`，用于 visit_scale）
    pub max_visit_init: usize,
    /// Q 值缩放系数（对齐 `value_scale`）
    pub value_scale: f32,
    /// Gumbel 噪声幅度（对齐 `gumbel_scale`）
    pub gumbel_scale: f32,
    /// 是否对根子节点叠加 Dirichlet 探索噪声
    pub add_dirichlet_noise: bool,
    /// 最终动作选择的 softmax 温度
    pub select_temperature: f32,
    /// Dirichlet 噪声 alpha
    pub dirichlet_alpha: f32,
    /// Dirichlet 噪声混合权重
    pub dirichlet_epsilon: f32,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            num_simulations: 64,
            max_num_considered_actions: 16,
            max_visit_init: 50,
            value_scale: 0.1,
            gumbel_scale: 1.0,
            add_dirichlet_noise: true,
            select_temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }
}

impl GumbelConfig {
    /// 纯 Gumbel 模式：不使用 Dirichlet 噪声。
    pub fn pure_gumbel(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            add_dirichlet_noise: false,
            ..Default::default()
        }
    }

    /// 推理模式：纯 Gumbel + 低温度（更贪心的选择），用于人机对弈和分析。
    pub fn inference(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            add_dirichlet_noise: false,
            select_temperature: 0.1,
            ..Default::default()
        }
    }

    /// 混合噪声模式：叠加 Dirichlet 噪声到根子节点 prior。
    pub fn mixed(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            add_dirichlet_noise: true,
            ..Default::default()
        }
    }
}

// ============================================================
//  Node
// ============================================================

/// MCTS 树节点索引。
pub type NodeId = usize;

/// MCTS 树节点（对齐 LightZero `node_gumbel_alphazero.h`）。
#[derive(Clone)]
pub struct Node {
    pub visit_count: u32,
    /// Σ(value)：每次反向传播累加的值（子节点自身视角）
    pub total_value: f32,
    /// 先验概率（来自 NN softmax 输出，用于 `compute_mixed_value`、Dirichlet 噪声）
    pub prior: f32,
    /// NN 原始 logit（softmax 之前的值，用于 Gumbel 得分计算）
    /// 对齐 MiniZero `MCTSNode::policy_logit_`
    pub policy_logit: f32,
    /// NN 对该节点局面的原始估值（expand 时填入，用于 `_compute_mixed_value`）
    pub raw_value: Option<f32>,
    /// 父节点索引，根节点为 None
    pub parent: Option<NodeId>,
    /// 子节点表（action → node_id），与 LightZero `std::map<int, Node*>` 对齐
    pub children: HashMap<ActionId, NodeId>,
}

impl Node {
    pub fn new(prior: f32, policy_logit: f32, parent: Option<NodeId>) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            prior,
            policy_logit,
            raw_value: None,
            parent,
            children: HashMap::new(),
        }
    }

    /// 节点自身视角的均值。
    #[inline]
    pub fn q(&self) -> f32 {
        if self.visit_count > 0 {
            self.total_value / self.visit_count as f32
        } else {
            0.0
        }
    }

    /// 是否为叶子节点（未展开或展开后无子节点）。
    /// 对齐 LightZero：未展开的节点应当被视作叶子，停止 selection 并展开。
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.raw_value.is_none() || self.children.is_empty()
    }

    #[inline]
    pub fn add_visit(&mut self, value: f32) {
        self.visit_count += 1;
        self.total_value += value;
    }
}

// ============================================================
//  MCTS
// ============================================================

pub struct MCTS<G: Game> {
    pub arena: Vec<Node>,
    /// 当前逻辑根节点在 arena 中的索引。子树复用后可能非 0。
    pub root_id: NodeId,
    _phantom: PhantomData<G>,
}

/// 单次搜索返回结果
#[derive(Debug)]
pub struct SearchResult {
    /// 选中的最佳动作（平铺索引），若为 action_shape 表示无合法动作
    pub best_move: ActionId,
    /// Improved policy（completed Q policy，对齐 LightZero `_get_improved_policy`）
    pub policy: Vec<f32>,
    /// MCTS 根节点价值（子节点 Q 按访问次数加权平均）
    pub root_value: f32,
    /// NN 原始先验概率（softmax 后的概率值，用于 KL 散度和分析界面）
    pub root_nn_prior: Vec<f32>,
    /// NN 对根节点的估值（未经 MCTS 修正）
    pub root_nn_value: f32,
    /// 每个动作的子节点 Q 值（未访问动作为 0）
    pub children_q: Vec<f32>,
    /// 每个动作的子节点访问次数
    pub children_visits: Vec<u32>,
}

// ─── 内部辅助常量 ────────────────────────────────────────────

/// `_score_considered` 的 logit 下限
const LOW_LOGIT: f32 = -1e9;

/// 初始根节点索引（arena 的第一个元素）。
const INITIAL_ROOT: NodeId = 0;

/// 返回切片中最大值的索引（平局时返回第一个）。
#[inline]
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

impl<G: Game> MCTS<G> {
    pub fn new() -> Self {
        Self {
            arena: vec![Node::new(0.0, 0.0, None)],
            root_id: INITIAL_ROOT,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn root(&self) -> &Node {
        &self.arena[self.root_id]
    }

    #[inline]
    pub fn root_mut(&mut self) -> &mut Node {
        &mut self.arena[self.root_id]
    }

    #[inline]
    fn push_node(&mut self, node: Node) -> NodeId {
        self.arena.push(node);
        self.arena.len() - 1
    }

    pub fn reset(&mut self) {
        self.arena.clear();
        self.root_id = INITIAL_ROOT;
        self.arena.push(Node::new(0.0, 0.0, None));
    }

    // ================================================================
    //  Search tree reuse（自对弈加速）
    // ================================================================

    /// 将 `action` 对应的子树提升为新根（O(1) 指针操作）。
    pub fn reuse_subtree(&mut self, action: ActionId) {
        let child_id = self.root().children[&action];
        self.root_id = child_id;
    }

    /// 收集某个节点的所有子节点 `(action, child_id)`。
    fn children_vec(&self, node_idx: NodeId) -> Vec<(ActionId, NodeId)> {
        self.arena[node_idx]
            .children
            .iter()
            .map(|(&a, &id)| (a, id))
            .collect()
    }

    // ================================================================
    //  Sequential Halving 表
    // ================================================================

    /// 生成 single visit sequence（对齐 `get_sequence_of_considered_visits`）。
    fn sequential_halving_seq(max_num_considered: usize, num_simulations: usize) -> Vec<usize> {
        if max_num_considered <= 1 {
            return (0..num_simulations).collect();
        }
        let log2max = (max_num_considered as f32).log2().ceil() as usize;
        let mut visits = vec![0usize; max_num_considered];
        let mut num_considered = max_num_considered;
        let mut seq = Vec::with_capacity(num_simulations);

        while seq.len() < num_simulations {
            let num_extra = 1.max(num_simulations / (log2max * num_considered));
            for _ in 0..num_extra {
                if seq.len() >= num_simulations {
                    break;
                }
                seq.extend_from_slice(&visits[..num_considered.min(visits.len())]);
                // visits[..num_considered] 每个元素 +1
                for j in 0..num_considered.min(visits.len()) {
                    visits[j] += 1;
                }
            }
            num_considered = 2.max(num_considered / 2);
        }
        seq.truncate(num_simulations);
        seq
    }

    /// 预计算 halving 表（对齐 `get_table_of_considered_visits`）。
    /// shape: (max_num_considered + 1) × num_simulations
    fn halving_table(max_num_considered: usize, num_simulations: usize) -> Array2<usize> {
        let rows = max_num_considered + 1;
        let cols = num_simulations;
        let mut table = Array2::from_elem((rows, cols), 0usize);
        for m in 0..=max_num_considered {
            let seq = Self::sequential_halving_seq(m, num_simulations);
            for (j, &v) in seq.iter().enumerate() {
                table[(m, j)] = v;
            }
        }
        table
    }

    // ================================================================
    //  辅助
    // ================================================================

    /// 计算未访问子节点的混合 Q 值（对齐 `_compute_mixed_value`）。
    ///
    /// ```text
    /// weighted_q = Σ(q_i * prior_i / sum_prior)   for visited children
    /// mixed = (raw_value + sum_visits * weighted_q) / (1 + sum_visits)
    /// ```
    fn compute_mixed_value(
        raw_value: f32,
        qvalues: &[f32],
        visit_counts: &[usize],
        priors: &[f32],
    ) -> f32 {
        let sum_visits: usize = visit_counts.iter().sum();
        if sum_visits == 0 {
            return raw_value;
        }

        // sum_prior over visited children
        let sum_prior: f32 = priors
            .iter()
            .zip(visit_counts.iter())
            .filter(|(_, v)| **v > 0)
            .map(|(p, _)| *p)
            .sum();

        if sum_prior <= 1e-12 {
            return raw_value;
        }

        let weighted_q: f32 = priors
            .iter()
            .zip(visit_counts.iter())
            .zip(qvalues.iter())
            .filter(|((_, v), _)| **v > 0)
            .map(|((p, _), q)| q * p / sum_prior)
            .sum();

        (raw_value + sum_visits as f32 * weighted_q) / (1.0 + sum_visits as f32)
    }

    /// 计算 completed Q 值（对齐 `_qtransform_completed_by_mix_value`）。
    ///
    /// 返回 `rescaled_q * (maxvisit_init + max_visit_count) * value_scale`。
    fn completed_q_values(
        &self,
        parent_idx: NodeId,
        children: &[(ActionId, NodeId)],
        config: &GumbelConfig,
    ) -> Vec<f32> {
        let parent = &self.arena[parent_idx];
        let n = children.len();
        let mut qvalues = Vec::with_capacity(n);
        let mut priors = Vec::with_capacity(n);
        let mut visit_counts = Vec::with_capacity(n);

        for &(_, id) in children {
            let child = &self.arena[id];
            // child.q() 是子节点自身视角，翻转为父节点视角
            qvalues.push(G::flip_perspective(child.q()));
            priors.push(child.prior);
            visit_counts.push(child.visit_count as usize);
        }

        let raw_value = parent.raw_value.unwrap();
        let mixed = Self::compute_mixed_value(raw_value, &qvalues, &visit_counts, &priors);

        // completed: visited → raw Q, unvisited → mixed
        let mut completed: Vec<f32> = visit_counts
            .iter()
            .zip(qvalues.iter())
            .map(|(&vc, &q)| if vc > 0 { q } else { mixed })
            .collect();

        let max_q = completed.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_q = completed.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let gap = (max_q - min_q).max(1e-8);
        let max_visit = visit_counts.iter().copied().max().unwrap_or(0);
        let visit_scale = (config.max_visit_init + max_visit) as f32;
        for q in &mut completed {
            *q = visit_scale * config.value_scale * (*q - min_q) / gap;
        }
        completed
    }

    // ================================================================
    //  节点选择
    // ================================================================

    /// 根节点选择（对齐 `_score_considered` + `_select_root_child`）。
    ///
    /// `sim_index`: 本轮内当前模拟的序号（0-based），用于 halving table 索引。
    /// 利用 Sequential Halving 表确定当前 step 应该考虑哪些 visit_count 的孩子，
    /// 然后在这些孩子中选 `max(gumbel + policy_logit + completed_q)`。
    fn select_root_child(
        &self,
        halving_table: &Array2<usize>,
        config: &GumbelConfig,
        gumbel_noises: &[f32],
        sim_index: usize,
    ) -> (ActionId, NodeId) {
        let children = self.children_vec(self.root_id);

        if children.is_empty() {
            return (0, 0);
        }

        // completed Q
        let completed_q = self.completed_q_values(self.root_id, &children, config);

        // simulation index: 使用本轮模拟序号而非累计 visit_count
        // (树复用时累计 visit_count 会超出 num_simulations，导致 halving table 越界)
        let simulation_index = sim_index.min(halving_table.ncols().saturating_sub(1));

        let num_considered = config
            .max_num_considered_actions
            .min(config.num_simulations)
            .min(halving_table.nrows().saturating_sub(1));
        let considered_visit = halving_table[(num_considered, simulation_index)];

        // score_considered
        let scores =
            self.score_considered(considered_visit, &children, &completed_q, gumbel_noises);

        let best_idx = argmax(&scores);
        let (action, child_id) = children[best_idx];
        (action, child_id)
    }

    /// 计算 `_score_considered`：
    ///
    /// ```text
    /// penalty = 0 (if visit_count == considered_visit) else -inf
    /// score = max(LOW_LOGIT, gumbel[i] + policy_logit_i + penalty + completed_q[i])
    /// ```
    /// 对齐 MiniZero：score 基于 logit 空间而非概率空间。
    fn score_considered(
        &self,
        considered_visit: usize,
        children: &[(ActionId, NodeId)],
        completed_q: &[f32],
        gumbel_noises: &[f32],
    ) -> Vec<f32> {
        children
            .iter()
            .enumerate()
            .map(|(i, &(_, id))| {
                let penalty = if self.arena[id].visit_count as usize == considered_visit {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
                let noise = gumbel_noises.get(i).copied().unwrap_or(0.0);
                let raw = noise + self.arena[id].policy_logit + penalty + completed_q[i];
                raw.max(LOW_LOGIT)
            })
            .collect()
    }

    /// 内部节点选择（对齐 `_select_interior_child`）：
    ///
    /// ```text
    /// probs = priors + completed_q
    /// to_argmax = probs - visit_count / (1 + sum_visits)
    /// choose max(to_argmax)
    /// ```
    fn select_interior_child(&self, node_idx: NodeId, config: &GumbelConfig) -> (ActionId, NodeId) {
        let children = self.children_vec(node_idx);

        if children.is_empty() {
            return (0, node_idx); // 回退
        }

        let completed_q = self.completed_q_values(node_idx, &children, config);

        let visit_counts: Vec<usize> = children
            .iter()
            .map(|&(_, id)| self.arena[id].visit_count as usize)
            .collect();

        let sum_visits: usize = visit_counts.iter().sum();

        // probs = policy_logit + completed_q（对齐 MiniZero logit 空间得分）
        let probs: Vec<f32> = children
            .iter()
            .enumerate()
            .map(|(i, &(_, id))| self.arena[id].policy_logit + completed_q[i])
            .collect();

        // to_argmax = probs - visit_count / (1 + sum_visits)
        let to_argmax: Vec<f32> = children
            .iter()
            .enumerate()
            .map(|(i, _)| probs[i] - visit_counts[i] as f32 / (1 + sum_visits) as f32)
            .collect();

        let best_idx = argmax(&to_argmax);
        let (action, child_id) = children[best_idx];
        (action, child_id)
    }

    // ================================================================
    //  Expand / Simulate / Backprop
    // ================================================================

    /// 展开节点：调用 NN，为每个合法动作创建子节点（对齐 `_expand_leaf_node`）。
    ///
    /// 返回叶子值（当前玩家视角），同时把 raw_value 写入节点。
    async fn expand_node<E: Evaluator>(
        &mut self,
        node_idx: NodeId,
        game: &G,
        evaluator: &E,
    ) -> f32 {
        let legal = game.legal_actions();

        if legal.is_empty() {
            return 0.0;
        }

        let encoding = game.encode();
        let (policies_batch, values_batch) = evaluator.evaluate_batch(&[encoding]).await;
        let leaf_value = values_batch[0];
        let raw_logits = &policies_batch[0];
        let probs = softmax_legal(raw_logits, &legal);

        // 记录 raw_value
        self.arena[node_idx].raw_value = Some(leaf_value);

        // 创建子节点
        for &action in &legal {
            let child = Node::new(probs[action], raw_logits[action], Some(node_idx));
            let child_id = self.push_node(child);
            self.arena[node_idx].children.insert(action, child_id);
        }

        leaf_value
    }

    /// 沿 parent 链反向传播（对齐 `update_recursive` self_play 模式）。
    ///
    /// 初始 value 是叶子节点视角。每向上一层通过 `G::flip_perspective` 翻转。
    fn backprop_from(&mut self, leaf_idx: NodeId, mut value: f32) {
        let mut idx = leaf_idx;
        loop {
            self.arena[idx].add_visit(value);
            if let Some(parent) = self.arena[idx].parent {
                idx = parent;
                value = G::flip_perspective(value);
            } else {
                break;
            }
        }
    }

    /// 单次模拟（对齐 `_simulate`）。
    ///
    /// `sim_index`: 本轮内当前模拟的序号（0-based），用于 halving table 索引。
    /// 从根出发，根用 `select_root_child`，内部用 `select_interior_child`，
    /// 到达叶节点后展开并反向传播。
    async fn simulate<E: Evaluator>(
        &mut self,
        game: &mut G,
        evaluator: &E,
        config: &GumbelConfig,
        halving_table: &Array2<usize>,
        gumbel_noises: &[f32],
        sim_index: usize,
    ) {
        let mut node_idx: NodeId = self.root_id;

        // ── selection: 向下走到叶节点 ──
        loop {
            let node = &self.arena[node_idx];
            if node.is_leaf() {
                break;
            }

            let (action, child_idx) = if node_idx == self.root_id {
                self.select_root_child(halving_table, config, gumbel_noises, sim_index)
            } else {
                self.select_interior_child(node_idx, config)
            };

            game.play(action);
            node_idx = child_idx;
        }

        // ── evaluation ──
        let leaf_value = if let Some(terminal_value) = game.terminal_value() {
            terminal_value
        } else {
            self.expand_node(node_idx, game, evaluator).await
        };

        // ── backprop ──
        // leaf_value 已是当前玩家（叶节点）视角，直接传入 backprop_from。
        // backprop_from 内部会在每往上一层时 flip_perspective。
        self.backprop_from(node_idx, leaf_value);
    }

    // ================================================================
    //  Search（对齐 `get_next_action`）
    // ================================================================

    pub async fn search<E: Evaluator>(
        &mut self,
        game: &G,
        evaluator: &E,
        config: &GumbelConfig,
        rng: &mut impl RngExt,
    ) -> SearchResult {
        let action_shape = game.action_shape();
        let legal_moves = game.legal_actions();

        // ── 空终止态快速返回 ──
        if legal_moves.is_empty() {
            return SearchResult {
                best_move: action_shape,
                policy: vec![0.0; action_shape],
                root_value: 0.0,
                root_nn_prior: vec![0.0; action_shape],
                root_nn_value: 0.0,
                children_q: vec![0.0; action_shape],
                children_visits: vec![0; action_shape],
            };
        }

        // ── Phase 1: 根节点初始化 ──
        let root_nn_value: f32;
        let raw_policy_probs: Vec<f32>;
        if let Some(raw_value) = self.root().raw_value {
            // 复用模式：NN 不变，从已有子节点恢复 prior 分布
            root_nn_value = raw_value;
            let mut probs = vec![0.0f32; action_shape];
            for (&action, &child_id) in self.root().children.iter() {
                probs[action] = self.arena[child_id].prior;
            }
            raw_policy_probs = probs;
        } else {
            let root_encoding = game.encode();
            let (root_logits, root_values) = evaluator.evaluate_batch(&[root_encoding]).await;
            root_nn_value = root_values[0];
            let raw_logit_slice = &root_logits[0];
            raw_policy_probs = softmax_legal(raw_logit_slice, &legal_moves);

            self.root_mut().raw_value = Some(root_nn_value);
            for &action in &legal_moves {
                let child = Node::new(
                    raw_policy_probs[action],
                    raw_logit_slice[action],
                    Some(self.root_id),
                );
                let child_id = self.push_node(child);
                self.root_mut().children.insert(action, child_id);
            }
        }

        // ── Phase 2: 可选 Dirichlet 噪声 ──
        if config.add_dirichlet_noise {
            self.add_dirichlet_noise(config, rng);
        }

        // ── Phase 3: 预计算 halving 表 & Gumbel 噪声 ──
        let halving_table =
            Self::halving_table(config.max_num_considered_actions, config.num_simulations);
        let num_legal = legal_moves.len();
        let gumbel_noises = Self::gumbel_noise(num_legal, config.gumbel_scale, rng);

        // ── Phase 4: 模拟循环 ──
        for sim_i in 0..config.num_simulations {
            let mut sim_game = game.clone();
            self.simulate(
                &mut sim_game,
                evaluator,
                config,
                &halving_table,
                &gumbel_noises,
                sim_i,
            )
            .await;
        }

        // ── Phase 5: 构建 improved policy ──
        let improved_policy = self.improved_policy(config, action_shape);

        // ── Phase 6: 动作选择 ──
        let action_probs = self.visit_count_distribution(config.select_temperature, action_shape);
        let best_move = sample_action(&action_probs, rng);

        // ── 收集子节点信息 ──
        let mut children_q = vec![0.0f32; action_shape];
        let mut children_visits = vec![0u32; action_shape];
        for (&action, &child_id) in self.root().children.iter() {
            let child = &self.arena[child_id];
            children_q[action] = G::flip_perspective(child.q());
            children_visits[action] = child.visit_count;
        }

        let total_visits: u32 = self.root().visit_count;
        let root_value = if total_visits > 0 {
            children_q
                .iter()
                .enumerate()
                .map(|(i, &q)| children_visits[i] as f32 / total_visits as f32 * q)
                .sum()
        } else {
            root_nn_value
        };

        SearchResult {
            best_move,
            policy: improved_policy,
            root_value,
            root_nn_prior: raw_policy_probs,
            root_nn_value,
            children_q,
            children_visits,
        }
    }

    // ================================================================
    //  Improved policy（对齐 `_get_improved_policy`）
    // ================================================================

    /// 构建 improved policy：`softmax(policy_logit + completed_q)`。
    /// 对齐 MiniZero `getMCTSPolicy`：使用 logit 空间的得分经 temperature-scaled softmax。
    fn improved_policy(&self, config: &GumbelConfig, action_shape: usize) -> Vec<f32> {
        let children_vec = self.children_vec(self.root_id);

        if children_vec.is_empty() {
            return vec![0.0; action_shape];
        }

        let completed_q = self.completed_q_values(self.root_id, &children_vec, config);

        let mut probs = vec![f32::NEG_INFINITY; action_shape];
        for (i, &(action, id)) in children_vec.iter().enumerate() {
            probs[action] = self.arena[id].policy_logit + completed_q[i];
        }

        softmax_full(&probs, 1.0)
    }

    // ================================================================
    //  Visit count → action distribution / 动作采样
    // ================================================================

    /// 基于访问次数的动作分布（对齐 `visit_count_to_action_distribution`）。
    fn visit_count_distribution(&self, temperature: f32, action_shape: usize) -> Vec<f64> {
        let root = self.root();
        let visits: Vec<f64> = (0..action_shape)
            .map(|a| {
                root.children
                    .get(&a)
                    .map(|&id| self.arena[id].visit_count as f64)
                    .unwrap_or(0.0)
            })
            .collect();

        let sum: f64 = visits.iter().sum();
        if sum == 0.0 || temperature == 0.0 {
            let n = action_shape as f64;
            return vec![1.0 / n; action_shape];
        }

        // 先除以温度再归一化
        let scaled: Vec<f64> = visits.iter().map(|&v| v / temperature as f64).collect();
        let scaled_sum: f64 = scaled.iter().sum();
        scaled.iter().map(|&v| v / scaled_sum).collect()
    }

    // ================================================================
    //  Dirichlet / Gumbel 噪声
    // ================================================================

    /// 对根节点的直接子节点叠加 Dirichlet 噪声（对齐 `_add_exploration_noise`）。
    fn add_dirichlet_noise(&mut self, config: &GumbelConfig, rng: &mut impl RngExt) {
        let children: Vec<_> = self.root().children.values().copied().collect();

        let n = children.len();
        if n == 0 {
            return;
        }

        let noise = dirichlet_noise(n, config.dirichlet_alpha, rng);
        let frac = config.dirichlet_epsilon;

        for (i, child_id) in children.into_iter().enumerate() {
            self.arena[child_id].prior =
                self.arena[child_id].prior * (1.0 - frac) + noise[i] * frac;
        }
    }

    /// 生成 Gumbel 噪声（对齐 `_generate_gumbel`，带 scale）。
    fn gumbel_noise(n: usize, scale: f32, rng: &mut impl RngExt) -> Vec<f32> {
        let gumbel = rand_distr::Gumbel::new(0.0f32, 1.0).unwrap();
        (0..n)
            .map(|_| scale * gumbel.sample(rng).clamp(-12.0, 12.0))
            .collect()
    }
}

// ============================================================
//  工具函数
// ============================================================

/// 仅对合法动作做 softmax。
#[inline]
fn softmax_legal(logits: &[f32], legal: &[ActionId]) -> Vec<f32> {
    let max_logit = legal
        .iter()
        .map(|&idx| logits[idx])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for &idx in legal {
        let exp = (logits[idx] - max_logit).exp();
        probs[idx] = exp;
        sum += exp;
    }
    if sum > 0.0 {
        for &idx in legal {
            probs[idx] /= sum;
        }
    } else {
        let count = legal.len() as f32;
        for &idx in legal {
            probs[idx] = 1.0 / count;
        }
    }
    probs
}

/// 全量 softmax（用于 improved policy，含 -inf 的未展开动作）。
fn softmax_full(values: &[f32], temperature: f32) -> Vec<f32> {
    if values.is_empty() || temperature == 0.0 {
        let n = values.len();
        return vec![1.0 / n as f32; n];
    }
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exps: Vec<f32> = values
        .iter()
        .map(|&v| ((v - max_val) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for v in &mut exps {
            *v /= sum;
        }
    }
    exps
}

/// Dirichlet 噪声（Gamma 采样后归一化）。
fn dirichlet_noise(n: usize, alpha: f32, rng: &mut impl RngExt) -> Vec<f32> {
    let gamma = rand_distr::Gamma::new(alpha, 1.0).unwrap();
    let mut v: Vec<f32> = (0..n).map(|_| gamma.sample(rng)).collect();
    let s: f32 = v.iter().sum();
    for x in &mut v {
        *x /= s;
    }
    v
}

/// 按概率分布采样动作索引。
fn sample_action(probs: &[f64], rng: &mut impl RngExt) -> ActionId {
    if probs.is_empty() {
        return 0;
    }
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        // 回退：选最大概率索引
        return probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
    let dist = WeightedIndex::new(probs).unwrap();
    dist.sample(rng)
}

// ============================================================
//  测试
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::{Board, Color};

    /// 模拟神经网络评估器。
    struct MockEvaluator {
        value_map: std::collections::HashMap<String, f32>,
        policy_map: std::collections::HashMap<String, Vec<f32>>,
    }

    impl MockEvaluator {
        fn new() -> Self {
            Self {
                value_map: std::collections::HashMap::new(),
                policy_map: std::collections::HashMap::new(),
            }
        }

        fn with_value(mut self, encoded: Vec<i32>, value: f32) -> Self {
            let key = Self::encode_key(&encoded);
            self.value_map.insert(key, value);
            self
        }

        fn with_policy(mut self, encoded: Vec<i32>, policy: Vec<f32>) -> Self {
            let key = Self::encode_key(&encoded);
            self.policy_map.insert(key, policy);
            self
        }

        fn encode_key(encoded: &[i32]) -> String {
            encoded
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        }
    }

    impl Evaluator for MockEvaluator {
        async fn evaluate_batch(&self, states: &[Vec<i32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
            let action_dim = states.first().map_or(0, |s| s.len());
            let policies: Vec<Vec<f32>> = states
                .iter()
                .map(|s| {
                    let key = Self::encode_key(s);
                    self.policy_map
                        .get(&key)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0f32; action_dim])
                })
                .collect();
            let values: Vec<f32> = states
                .iter()
                .map(|s| {
                    let key = Self::encode_key(s);
                    self.value_map.get(&key).copied().unwrap_or(0.0)
                })
                .collect();
            (policies, values)
        }
    }

    fn make_board() -> Board {
        Board::with_size(3, 3)
    }

    // ── 基本测试 ──

    #[test]
    fn test_empty_board_search_returns_legal_move() {
        let board = make_board();
        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(64);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));

        assert!(result.best_move < 9);
        assert_eq!(result.policy.len(), 9);
        assert_eq!(result.root_nn_prior.len(), 9);

        let policy_sum: f32 = result.policy.iter().sum();
        assert!(
            (policy_sum - 1.0).abs() < 0.02,
            "policy sum should be ~1.0, got {policy_sum}"
        );

        assert!(result.root_value >= -1.0 && result.root_value <= 1.0);
    }

    #[test]
    fn test_terminal_board_returns_empty() {
        let mut board = make_board();
        board.play(0, 0);
        board.play(1, 0);
        board.play(0, 1);
        board.play(1, 1);
        board.play(0, 2);
        assert!(board.game_over);

        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(32);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));

        assert_eq!(result.best_move, 9);
        assert_eq!(result.policy, vec![0.0f32; 9]);
        assert_eq!(result.root_value, 0.0);
    }

    #[test]
    fn test_single_legal_move_is_chosen() {
        let mut board = make_board();
        let moves = [
            (0, 1),
            (1, 0),
            (0, 2),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ];
        for &(r, c) in &moves {
            board.play(r, c);
        }
        assert!(!board.game_over);
        assert_eq!(board.legal_actions().len(), 1);
        let only_move = board.pos_to_idx(0, 0);

        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(64);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));

        assert_eq!(result.best_move, only_move);
    }

    #[test]
    fn test_mcts_prefers_winning_move() {
        let mut board = make_board();
        board.play(0, 0);
        board.play(1, 0);
        board.play(0, 1);
        board.play(1, 1);
        assert!(!board.game_over);

        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(256);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));

        let win_move = board.pos_to_idx(0, 2);
        assert!(result.best_move < 9);
        assert!(
            result.policy[win_move] > 0.0,
            "winning move should have non-zero policy"
        );
    }

    #[test]
    fn test_mcts_blocks_opponent_win() {
        let mut board = make_board();
        board.play(0, 0);
        board.play(1, 0);
        board.play(0, 1);
        assert_eq!(board.current_player, Color::White);
        assert!(!board.game_over);

        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(256);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));

        let block_move = board.pos_to_idx(0, 2);
        assert!(result.best_move < 9);
        assert!(
            result.policy[block_move] > 0.0,
            "blocking move should have non-zero policy"
        );
    }

    #[test]
    fn test_best_move_reflects_policy() {
        let board = make_board();
        let center_idx = board.pos_to_idx(1, 1);

        let mut policy = vec![-10.0f32; 9];
        policy[center_idx] = 10.0;
        let evaluator = MockEvaluator::new().with_policy(board.encode(), policy);

        let config = GumbelConfig::pure_gumbel(64);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));

        assert!(
            result.root_nn_prior[center_idx] > 0.5,
            "center NN prior should be > 0.5, got {}",
            result.root_nn_prior[center_idx]
        );
    }

    #[test]
    fn test_pure_gumbel_config() {
        let config = GumbelConfig::pure_gumbel(128);
        assert!(!config.add_dirichlet_noise);
        assert_eq!(config.num_simulations, 128);
    }

    #[test]
    fn test_mixed_config() {
        let config = GumbelConfig::mixed(128);
        assert!(config.add_dirichlet_noise);
        assert_eq!(config.num_simulations, 128);
        assert!(config.dirichlet_epsilon > 0.0);
    }

    #[test]
    fn test_mixed_mode_search_on_empty_board() {
        let board = make_board();
        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::mixed(64);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));
        assert!(result.best_move < 9);
        assert!((result.policy.iter().sum::<f32>() - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_win_vs_delay() {
        let mut board = make_board();
        board.play(0, 0);
        board.play(1, 0);
        board.play(0, 1);
        board.play(1, 1);

        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(256);
        let win_move = board.pos_to_idx(0, 2);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));

        println!("policy: {:?}", result.policy);
        println!("children_q: {:?}", result.children_q);
        println!("children_visits: {:?}", result.children_visits);
        println!("root_value: {}", result.root_value);
        println!("root_nn_value: {}", result.root_nn_value);

        let max_idx = result
            .policy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, win_move, "winning move should have highest policy");
    }

    #[test]
    fn test_game_ends_correctly() {
        let mut board = make_board();
        assert!(board.play(0, 0));
        assert!(board.play(1, 0));
        assert!(board.play(0, 1));
        assert!(board.play(1, 1));
        assert!(board.play(0, 2));
        assert!(board.game_over);
        assert_eq!(board.winner, Some(Color::Black));
    }

    #[test]
    fn test_full_game_completion() {
        let mut board = make_board();
        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(128);
        let mut mcts: MCTS<Board> = MCTS::new();

        for _ in 0..9 {
            if board.game_over {
                break;
            }
            mcts.reset(); // 每步独立，不复用子树
            let result = futures_executor::block_on(mcts.search(
                &board,
                &evaluator,
                &config,
                &mut rand::rng(),
            ));
            if result.best_move >= 9 {
                break;
            }
            board.play_idx(result.best_move);
        }

        assert!(board.game_over);
        assert!(board.step_count <= 9);
    }

    #[test]
    fn test_draw_game() {
        let mut board = make_board();
        let moves = [
            (0, 0),
            (1, 1),
            (2, 2),
            (2, 0),
            (0, 2),
            (0, 1),
            (2, 1),
            (1, 2),
            (1, 0),
        ];
        for &(r, c) in &moves {
            assert!(board.play(r, c), "move ({},{}) should be legal", r, c);
        }
        assert!(board.game_over);
        assert_eq!(board.winner, None);
    }

    #[test]
    fn test_nn_value_influences_root_value() {
        let board = make_board();
        let evaluator = MockEvaluator::new().with_value(board.encode(), 1.0);
        let config = GumbelConfig::pure_gumbel(1024);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));

        assert!(
            result.root_value > -1.0,
            "positive NN value should not produce extreme negative root value"
        );
    }

    // ── 算法一致性测试 ──

    #[test]
    fn test_sequential_halving_sequence_monotonic() {
        let seq = MCTS::<Board>::sequential_halving_seq(4, 16);
        assert_eq!(seq.len(), 16);
        // 序列应非递减
        for w in seq.windows(2) {
            assert!(w[0] <= w[1], "halving seq should be non-decreasing");
        }
    }

    #[test]
    fn test_compute_mixed_value_no_visits_returns_raw() {
        let raw = 0.7;
        let qvalues: [f32; 0] = [];
        let visit_counts: [usize; 0] = [];
        let priors: [f32; 0] = [];
        let result = MCTS::<Board>::compute_mixed_value(raw, &qvalues, &visit_counts, &priors);
        assert!((result - raw).abs() < 1e-6);
    }

    #[test]
    fn test_compute_mixed_value_with_visits() {
        // 一个已访问子节点：Q=0.5, visits=3, prior=0.6
        let raw = 0.3;
        let qvalues = vec![0.5];
        let visit_counts = vec![3];
        let priors = vec![0.6];
        let result = MCTS::<Board>::compute_mixed_value(raw, &qvalues, &visit_counts, &priors);
        // weighted_q = 0.5 * 0.6 / 0.6 = 0.5
        // mixed = (0.3 + 3 * 0.5) / (1 + 3) = 1.8 / 4 = 0.45
        assert!((result - 0.45).abs() < 1e-5, "expected 0.45, got {result}");
    }

    #[test]
    fn test_improved_policy_sums_to_one() {
        let board = make_board();
        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(64);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));
        let sum: f32 = result.policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.02,
            "improved policy sum should be ~1, got {sum}"
        );
    }

    #[test]
    fn test_halving_table_correct_size() {
        let table = MCTS::<Board>::halving_table(4, 16);
        assert_eq!(table.nrows(), 5); // 0..=4
        assert_eq!(table.ncols(), 16);
    }

    #[test]
    fn test_root_and_interior_selection_reach_leaves() {
        // 验证在充足模拟下，所有合法动作都有非零访问
        let board = make_board();
        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(256);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result =
            futures_executor::block_on(mcts.search(&board, &evaluator, &config, &mut rand::rng()));

        // 所有 9 个合法动作都应该有非零访问
        let nonzero_visits = result.children_visits.iter().filter(|&&v| v > 0).count();
        assert!(
            nonzero_visits >= 5,
            "at least 5 actions should have visits, got {}",
            nonzero_visits
        );
    }

    #[test]
    fn test_select_interior_child_prefers_visited_high_q() {
        // 构建一个简单的人工树来验证 interior 选择偏好对父节点有利的 Q
        // child.q() 是子节点自身视角；父视角 = flip_perspective(child.q())
        // 父节点应选择父视角下 Q 更高的子节点（即子视角 Q 更低的）
        let mut mcts: MCTS<Board> = MCTS::new();
        let config = GumbelConfig::default();

        // 根节点
        mcts.arena.push(Node::new(0.0, 0.0, None));
        mcts.arena[0].raw_value = Some(0.0); // completed_q 需要 raw_value
        // 两个子节点
        let child0 = mcts.push_node(Node::new(0.5, 0.5, Some(0)));
        let child1 = mcts.push_node(Node::new(0.5, 0.5, Some(0)));
        mcts.arena[0].children.insert(0, child0);
        mcts.arena[0].children.insert(1, child1);

        // child0: 子视角高 Q（对父不利，父视角 = -0.8）
        for _ in 0..10 {
            mcts.arena[child0].add_visit(0.8);
        }
        // child1: 子视角低 Q（对父有利，父视角 = +0.5）
        for _ in 0..10 {
            mcts.arena[child1].add_visit(-0.5);
        }

        // 两个孙子节点挂在 child0 下（使 child0 不是叶子）
        let gc = mcts.push_node(Node::new(0.5, 0.5, Some(child0)));
        mcts.arena[child0].children.insert(0, gc);

        let (action, _) = mcts.select_interior_child(0, &config);
        assert_eq!(
            action, 1,
            "should prefer child with higher Q from parent's perspective"
        );
    }
}
