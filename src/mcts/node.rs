//! 蒙特卡洛树搜索 (MCTS) — Gumbel Zero (Gumbel AlphaZero)
//!
//! 严格对齐 minizero 的 Gumbel AlphaZero 实现：
//! - Gumbel 噪声加在 logit 空间（`policy_logit = ln(prior) + gumbel_noise`）
//! - Sequential Halving 用带噪声 logit + normalized Q 做候选排序
//! - Completed Q policy 用干净 logit（`logit_without_noise = logit - noise`）
//! - 动作选择用 softmax over visit counts（temperature=1.0, value_threshold=0.1）
//!
//! ## 噪声模式
//!
//! - **纯 Gumbel**（`GumbelConfig::pure_gumbel()`）：
//!   prior = NN 干净输出，`policy_logit = ln(prior) + gumbel_noise`。
//!
//! - **混合噪声**（`GumbelConfig::mixed()`）：
//!   prior = (1-ε)*NN + ε*Dirichlet，`policy_logit = ln(NN_prior) + gumbel_noise`。
//!   PUCT walk 用 noisy prior。

use crate::game::board::{Board, ENCODE_CHANNELS, NUM_POSITIONS};
use crate::inference::Evaluator;

// ============================================================
//  常量
// ============================================================

/// PUCT 初始项（对齐 minizero `actor_mcts_puct_init`）
const PUCT_INIT: f32 = 1.25;
/// PUCT 基项（对齐 minizero `actor_mcts_puct_base`）
const PUCT_BASE: f32 = 19652.0;

/// 获胜模拟的价值
const WIN_VALUE: f32 = 1.0;
/// 平局模拟的价值
const DRAW_VALUE: f32 = 0.0;

// ============================================================
//  GumbelConfig
// ============================================================

#[derive(Debug, Clone)]
pub struct GumbelConfig {
    pub num_simulations: usize,
    pub sample_size: usize,
    pub sigma_visit_c: f32,
    pub sigma_scale_c: f32,
    pub pure_gumbel_noise: bool,
    /// 动作选择 softmax 温度（minizero 默认 1.0）
    pub select_temperature: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            num_simulations: 32,
            sample_size: 16,
            sigma_visit_c: 50.0,
            sigma_scale_c: 1.0,
            pure_gumbel_noise: true,
            select_temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }
}

impl GumbelConfig {
    pub fn pure_gumbel(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            pure_gumbel_noise: true,
            ..Default::default()
        }
    }
    pub fn mixed(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            pure_gumbel_noise: false,
            ..Default::default()
        }
    }
}

// ============================================================
//  Node
// ============================================================

#[derive(Clone)]
pub struct Node {
    pub visit_count: u32,
    pub total_value: f32,
    /// 先验概率（PUCT walk 使用；纯 Gumbel 模式 = NN prior，混合模式 = noisy prior）
    pub prior: f32,
    /// 带 Gumbel 噪声的 logit：ln(NN_prior) + gumbel_noise
    /// Sequential Halving 候选排序用这个。
    pub policy_logit: f32,
    /// Gumbel 噪声值（用于 recovered clean logit = policy_logit - gumbel_noise）
    pub gumbel_noise: f32,
    pub children: Vec<Option<usize>>,
    pub expanded: bool,
}

impl Node {
    pub fn new(prior: f32, gumbel_logit: f32, gumbel_noise: f32) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            prior,
            policy_logit: gumbel_logit + gumbel_noise,
            gumbel_noise,
            children: vec![None; NUM_POSITIONS],
            expanded: false,
        }
    }

    /// 去掉 Gumbel 噪声后的干净 logit：用于 completed Q policy。
    #[inline]
    pub fn logit_without_noise(&self) -> f32 {
        self.policy_logit - self.gumbel_noise
    }

    #[inline]
    pub fn visit_count_f32(&self) -> f32 {
        self.visit_count as f32
    }

    #[inline]
    pub fn q(&self) -> f32 {
        if self.visit_count > 0 {
            self.total_value / self.visit_count as f32
        } else {
            0.0
        }
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

pub struct MCTS {
    pub arena: Vec<Node>,
}

pub struct SearchResult {
    pub best_move: usize,
    pub policy: Vec<f32>,
    pub root_value: f32,
}

impl MCTS {
    pub fn new() -> Self {
        Self {
            arena: vec![Node::new(0.0, 0.0, 0.0)],
        }
    }

    #[inline]
    fn root(&self) -> &Node {
        &self.arena[0]
    }
    #[inline]
    fn root_mut(&mut self) -> &mut Node {
        &mut self.arena[0]
    }
    #[inline]
    fn node(&self, idx: usize) -> &Node {
        &self.arena[idx]
    }
    #[inline]
    fn node_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.arena[idx]
    }

    fn push_node(&mut self, node: Node) -> usize {
        self.arena.push(node);
        self.arena.len() - 1
    }

    // ── expand ──

    fn expand_children(
        &mut self,
        parent: usize,
        legal: &[(usize, usize)],
        gumbel_noises: &[f32],
        // (NN_prior, clean_logit) per child
        child_data: &[(f32, f32)],
    ) {
        let mut child_ids = Vec::with_capacity(legal.len());
        for i in 0..legal.len() {
            let (prior, clean_logit) = child_data[i];
            child_ids.push(self.push_node(Node::new(prior, clean_logit, gumbel_noises[i])));
        }
        let parent_node = self.node_mut(parent);
        for (i, &(r, c)) in legal.iter().enumerate() {
            let idx = Board::pos_to_idx(r, c);
            parent_node.children[idx] = Some(child_ids[i]);
        }
    }

    // ── PUCT walk（对齐 minizero AlphaZero PUCT 公式）──

    /// 计算当前节点的 PUCT bias。
    /// 对齐 minizero `getNormalizedPUCTScore`:
    ///   `puct_bias = init + ln((1 + total_simulation + base) / base)`
    #[inline]
    fn puct_bias(total_simulation: f32) -> f32 {
        PUCT_INIT + ((1.0 + total_simulation + PUCT_BASE) / PUCT_BASE).ln()
    }

    /// 计算当前节点对未访问子节点的初始 Q 估计。
    /// 对齐 minizero `calculateInitQValue`（board games 分支）：
    ///   `init_q = (sum_q - 1) / (visited_count + 1)`
    fn init_q_value(&self, node_idx: usize) -> f32 {
        let node = self.node(node_idx);
        let mut sum_q = 0.0f32;
        let mut visited = 0.0f32;
        for c in node.children.iter().flatten() {
            let child = self.node(*c);
            if child.visit_count > 0 {
                sum_q += child.q();
                visited += 1.0;
            }
        }
        if visited > 0.0 {
            (sum_q - 1.0) / (visited + 1.0)
        } else {
            0.0
        }
    }

    fn puct_walk(
        &self,
        start_node_idx: usize,
        board: &mut Board,
        legal: &mut Vec<(usize, usize)>,
    ) -> (Vec<(usize, usize, usize)>, usize) {
        let mut path: Vec<(usize, usize, usize)> = Vec::new();
        let mut node_idx = start_node_idx;

        loop {
            if board.game_over {
                return (path, node_idx);
            }
            board.fill_legal_moves(legal);
            if legal.is_empty() {
                return (path, node_idx);
            }

            let node = self.node(node_idx);
            if !node.expanded {
                return (path, node_idx);
            }

            // total_simulation = node->getCountWithVirtualLoss() - 1（对齐 minizero）
            let total_simulation = node.visit_count.saturating_sub(1) as f32;
            let bias = Self::puct_bias(total_simulation);
            let sqrt_total = total_simulation.sqrt();
            let init_q = self.init_q_value(node_idx);

            let children = &node.children;
            let mut best: Option<(usize, usize, f32, f32)> = None;
            for &(r, c) in legal.iter() {
                let idx = Board::pos_to_idx(r, c);
                if let Some(ci) = children[idx] {
                    let child = self.node(ci);
                    let n = child.visit_count as f32;
                    let q = if n > 0.0 {
                        child.total_value / n
                    } else {
                        init_q
                    };
                    let score = q + bias * child.prior * sqrt_total / (1.0 + n);
                    let policy = child.prior;
                    if best.map_or(true, |(_, _, s, p)| score > s || (score == s && policy > p)) {
                        best = Some((idx, ci, score, policy));
                    }
                }
            }

            let (move_idx, child_idx) = match best {
                Some((mi, ci, _, _)) => (mi, ci),
                None => {
                    let idx = Board::pos_to_idx(legal[0].0, legal[0].1);
                    (idx, children[idx].unwrap())
                }
            };

            path.push((node_idx, move_idx, child_idx));
            board.play_idx(move_idx);
            node_idx = child_idx;
        }
    }

    fn root_move_idx_for_child(&self, child_idx: usize) -> Option<usize> {
        self.root()
            .children
            .iter()
            .position(|c| c == &Some(child_idx))
    }

    // ── Gumbel Zero search ──

    pub fn search<E: Evaluator>(
        &mut self,
        board: &mut Board,
        evaluator: &E,
        config: &GumbelConfig,
    ) -> SearchResult {
        let legal_moves = board.legal_moves();
        if legal_moves.is_empty() {
            return SearchResult {
                best_move: NUM_POSITIONS,
                policy: vec![0.0; NUM_POSITIONS],
                root_value: 0.0,
            };
        }

        let num_sim = config.num_simulations;
        let sample_n = config.sample_size;

        // ── Phase 1: 根节点 NN 评估 ──
        let mut root_encoding = vec![0.0f32; ENCODE_CHANNELS * NUM_POSITIONS];
        board.encode_into(&mut root_encoding);
        let (root_logits, root_values) = evaluator.evaluate_batch(&[root_encoding]);
        let root_nn_value = root_values[0];

        let policy_probs = Self::softmax_legal(&root_logits[0], &legal_moves);

        // 干净 logit：ln(NN prior)
        let clean_logits: Vec<f32> = legal_moves
            .iter()
            .map(|&(r, c)| policy_probs[Board::pos_to_idx(r, c)].ln())
            .collect();

        // Gumbel 噪声（所有场景下都生成）
        let gumbel_noises = Self::gumbel_noise(legal_moves.len());

        // ── Phase 1b: expand root ──
        let child_data: Vec<(f32, f32)> = if config.pure_gumbel_noise {
            // 纯 Gumbel：prior = NN prior（干净）
            legal_moves
                .iter()
                .enumerate()
                .map(|(i, &(r, c))| {
                    let idx = Board::pos_to_idx(r, c);
                    (policy_probs[idx], clean_logits[i])
                })
                .collect()
        } else {
            // 混合噪声：prior = (1-ε)*NN + ε*Dirichlet
            let dir = Self::dirichlet_noise(legal_moves.len(), config.dirichlet_alpha);
            legal_moves
                .iter()
                .enumerate()
                .map(|(i, &(r, c))| {
                    let idx = Board::pos_to_idx(r, c);
                    let prior = (1.0 - config.dirichlet_epsilon) * policy_probs[idx]
                        + config.dirichlet_epsilon * dir[i];
                    (prior, clean_logits[i])
                })
                .collect()
        };

        self.expand_children(0, &legal_moves, &gumbel_noises, &child_data);
        self.root_mut().expanded = true;

        // ── Phase 2: 候选集（按带噪声的 policy_logit 排序取 top-k） ──
        let children = &self.root().children;
        let mut candidates: Vec<usize> = Vec::with_capacity(legal_moves.len());
        for &(r, c) in &legal_moves {
            let idx = Board::pos_to_idx(r, c);
            if let Some(ci) = children[idx] {
                candidates.push(ci);
            }
        }
        // 初期候选按 policy_logit 降序排序
        candidates.sort_by(|&a, &b| {
            self.node(b)
                .policy_logit
                .partial_cmp(&self.node(a).policy_logit)
                .unwrap()
        });
        if candidates.len() > sample_n {
            candidates.truncate(sample_n);
        }

        // ── Phase 3: 模拟循环 ──
        let mut cur_sample = sample_n;
        let mut sim_budget = Self::gumbel_budget(num_sim, cur_sample);

        let mut sim_board = board.clone();
        let mut legal_buf = Vec::with_capacity(NUM_POSITIONS);

        // 第一轮模拟：标准 PUCT walk（对齐 minizero selection 中 numSimulation==0 的分支）
        for sim_i in 0..num_sim {
            // ── selection ──
            let path: Vec<(usize, usize, usize)>;

            if sim_i == 0 {
                // 第一次模拟：从 root 走标准 PUCT
                sim_board.clone_from(board);
                path = self.puct_walk(0, &mut sim_board, &mut legal_buf).0;
            } else {
                // 从 count 最小候选出发（ties broken by higher logit）
                candidates.sort_by(|&a, &b| {
                    self.node(a)
                        .visit_count
                        .cmp(&self.node(b).visit_count)
                        .then_with(|| {
                            self.node(b)
                                .policy_logit
                                .partial_cmp(&self.node(a).policy_logit)
                                .unwrap()
                        })
                });
                let start_ci = candidates[0];
                let move_idx = self.root_move_idx_for_child(start_ci).unwrap_or(0);

                // 推进棋盘到候选子节点对应的局面，否则 puct_walk 会在 root 局面上搜索候选节点的子树
                sim_board.clone_from(board);
                sim_board.play_idx(move_idx);

                let (p_from, _leaf) = self.puct_walk(start_ci, &mut sim_board, &mut legal_buf);
                let mut full = vec![(0, move_idx, start_ci)];
                full.extend(p_from);
                path = full;
            }

            let leaf_idx = path.last().map(|&(_, _, ci)| ci).unwrap_or(0);

            // ── evaluate & expand & backup ──
            if sim_board.game_over {
                let v = match sim_board.winner {
                    Some(_) => WIN_VALUE,
                    None => DRAW_VALUE,
                };
                self.backprop_path(&path, v);
            } else if !self.node(leaf_idx).expanded {
                let mut encoding = vec![0.0f32; ENCODE_CHANNELS * NUM_POSITIONS];
                sim_board.encode_into(&mut encoding);
                sim_board.fill_legal_moves(&mut legal_buf);
                let legal_leaf = std::mem::take(&mut legal_buf);

                let (policies_batch, values_batch) = evaluator.evaluate_batch(&[encoding]);
                let probs = Self::softmax_legal(&policies_batch[0], &legal_leaf);

                // 非根节点不需要噪声（PUCT walk 子节点只用 prior，不参与 Gumbel 候选）
                let gumbel_noises_leaf = vec![0.0f32; legal_leaf.len()];
                let child_data_leaf: Vec<(f32, f32)> = legal_leaf
                    .iter()
                    .map(|&(r, c)| {
                        let idx = Board::pos_to_idx(r, c);
                        (probs[idx], probs[idx].ln())
                    })
                    .collect();
                self.expand_children(leaf_idx, &legal_leaf, &gumbel_noises_leaf, &child_data_leaf);
                self.node_mut(leaf_idx).expanded = true;
                self.backprop_path(&path, -values_batch[0]);
            }

            // ── sequentialHalving ──
            let all_reached = candidates
                .iter()
                .all(|&ci| self.node(ci).visit_count_f32() >= sim_budget as f32);

            if all_reached {
                // 对齐 minizero：始终用初始 sample_size 的 log2
                let next_budget = Self::gumbel_budget_halved(num_sim, cur_sample, sample_n);
                if next_budget > 0 && cur_sample > 2 {
                    cur_sample /= 2;
                    // 按 σ-score 重排（带噪声 logit + normalized Q）
                    let max_n = Self::max_root_count(self);
                    candidates.sort_by(|&a, &b| {
                        let sa = Self::sigma_score(self.node(a), max_n, config);
                        let sb = Self::sigma_score(self.node(b), max_n, config);
                        sb.partial_cmp(&sa).unwrap()
                    });
                    if candidates.len() > cur_sample {
                        candidates.truncate(cur_sample);
                    }
                    // 对齐 minizero: simulation_budget_ = candidates_[0]->getCount() + next_budget
                    let base = self.node(candidates[0]).visit_count;
                    sim_budget = base + next_budget;
                }
            }
        }

        // ── Phase 4: 构建 completed Q 策略（干净 logit） ──
        let (policy, root_value) = self.build_completed_q_policy(root_nn_value, num_sim, config);

        // ── Phase 5: 动作选择（softmax over visit counts） ──
        let best_move = self.select_by_softmax_count(config.select_temperature);

        SearchResult {
            best_move,
            policy,
            root_value,
        }
    }

    // ── Gumbel 辅助 ──

    /// 初始 budget：B = N / (log2(sample_size) * sample_size)
    /// 对齐 minizero `sequentialHalving` 中 numSimulation==1 的公式。
    fn gumbel_budget(num_sim: usize, sample_n: usize) -> u32 {
        let d = (sample_n as f32).log2() * sample_n as f32;
        (num_sim as f32 / d).floor().max(1.0) as u32
    }

    /// Halving budget：B' = N / (log2(initial_sample_size) * cur_sample / 2)
    /// 对齐 minizero：始终用初始 sample_size 的 log2，而非当前值。
    fn gumbel_budget_halved(num_sim: usize, cur_sample: usize, initial_sample_size: usize) -> u32 {
        let d = (initial_sample_size as f32).log2() * (cur_sample as f32) / 2.0;
        (num_sim as f32 / d).floor() as u32
    }

    fn max_root_count(mcts: &MCTS) -> f32 {
        mcts.root()
            .children
            .iter()
            .filter_map(|c| c.map(|ci| mcts.node(ci).visit_count_f32()))
            .fold(0.0, f32::max)
    }

    /// σ-score：带噪声 logit + Q 项。
    /// 对齐 minizero `sortCandidatesByScore`。
    fn sigma_score(node: &Node, max_n: f32, config: &GumbelConfig) -> f32 {
        if node.visit_count == 0 {
            return f32::NEG_INFINITY;
        }
        node.policy_logit + (config.sigma_visit_c + max_n) * config.sigma_scale_c * node.q()
    }

    /// Completed Q policy：用**去噪声** logit + Q 项，再做 softmax。
    /// 对齐 minizero `getMCTSPolicy`。
    fn build_completed_q_policy(
        &self,
        root_nn_value: f32,
        num_sim: usize,
        config: &GumbelConfig,
    ) -> (Vec<f32>, f32) {
        let children = &self.root().children;

        // pi_sum, q_sum（只算 visited）
        let mut pi_sum = 0.0f32;
        let mut q_sum = 0.0f32;
        for c in children.iter().flatten() {
            let child = self.node(*c);
            if child.visit_count > 0 {
                pi_sum += child.prior;
                q_sum += child.prior * child.q();
            }
        }

        // 未访问节点价值估计
        let non_visited_value = if pi_sum > 0.0 {
            1.0 / (1.0 + num_sim as f32) * (root_nn_value + (num_sim as f32 / pi_sum) * q_sum)
        } else {
            root_nn_value
        };

        let max_n = Self::max_root_count(self);
        let sv = config.sigma_visit_c;
        let sc = config.sigma_scale_c;

        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(NUM_POSITIONS);
        let mut max_score = f32::NEG_INFINITY;
        for (idx, c) in children.iter().enumerate() {
            if let Some(ci) = c {
                let child = self.node(*ci);
                let value = if child.visit_count > 0 {
                    child.q()
                } else {
                    non_visited_value
                };
                // 干净 logit（去掉 Gumbel 噪声）
                let score = child.logit_without_noise() + (sv + max_n) * sc * value;
                scores.push((idx, score));
                if score > max_score {
                    max_score = score;
                }
            }
        }

        // softmax
        let mut policy = vec![0.0f32; NUM_POSITIONS];
        let mut sum = 0.0f32;
        for &(idx, score) in &scores {
            let exp = (score - max_score).exp();
            policy[idx] = exp;
            sum += exp;
        }
        if sum > 0.0 {
            for &(idx, _) in &scores {
                policy[idx] /= sum;
            }
        }

        // root_value
        let sum_n: f32 = children
            .iter()
            .filter_map(|c| c.map(|ci| self.node(ci).visit_count_f32()))
            .sum();
        let root_value = if sum_n > 0.0 {
            children
                .iter()
                .filter_map(|c| {
                    c.and_then(|ci| {
                        let child = self.node(ci);
                        if child.visit_count > 0 {
                            Some(child.visit_count_f32() / sum_n * child.q())
                        } else {
                            None
                        }
                    })
                })
                .sum()
        } else {
            root_nn_value
        };

        (policy, root_value)
    }

    /// 动作选择：softmax over visit_counts，temperature 默认 1.0。
    /// 对齐 minizero `selectChildBySoftmaxCount`。
    fn select_by_softmax_count(&self, temperature: f32) -> usize {
        use rand::distr::{Distribution, weighted::WeightedIndex};
        let children = &self.root().children;

        // 找到最大 count，计算 value_threshold 的 reference Q
        let max_count_child = children
            .iter()
            .filter_map(|c| *c)
            .max_by_key(|ci| self.node(*ci).visit_count);
        let threshold_q = max_count_child
            .map(|ci| self.node(ci).q() - 0.1)
            .unwrap_or(f32::NEG_INFINITY);

        let probs: Vec<f64> = children
            .iter()
            .enumerate()
            .map(|(_idx, c)| {
                if let Some(ci) = c {
                    let child = self.node(*ci);
                    if child.visit_count == 0 || child.q() < threshold_q {
                        0.0
                    } else {
                        (child.visit_count as f64).powf(1.0 / temperature as f64)
                    }
                } else {
                    0.0
                }
            })
            .collect();

        let total: f64 = probs.iter().sum();
        if total > 0.0 {
            let dist = WeightedIndex::new(&probs).unwrap();
            dist.sample(&mut rand::rng())
        } else {
            // fallback: argmax over completed Q policy
            let (policy, _) = self.build_completed_q_policy(0.0, 1, &GumbelConfig::default());
            policy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    // ── backprop ──

    fn backprop_path(&mut self, path: &[(usize, usize, usize)], mut value: f32) {
        for &(_, _, ci) in path.iter().rev() {
            self.node_mut(ci).add_visit(value);
            value = -value;
        }
    }

    // ── utils ──

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
                probs[Board::pos_to_idx(r, c)] /= sum;
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
        let mut v: Vec<f32> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
        let s: f32 = v.iter().sum();
        for x in &mut v {
            *x /= s;
        }
        v
    }

    fn gumbel_noise(n: usize) -> Vec<f32> {
        use rand::distr::Distribution;
        let gumbel = rand_distr::Gumbel::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        (0..n).map(|_| gumbel.sample(&mut rng)).collect()
    }
}

// ============================================================
//  tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_logit_without_noise() {
        let n = Node::new(0.6, 0.6f32.ln(), 0.5);
        assert!((n.policy_logit - (0.6f32.ln() + 0.5)).abs() < 1e-6);
        assert!((n.logit_without_noise() - 0.6f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_node_q_basic() {
        let mut n = Node::new(0.5, 0.5f32.ln(), 0.0);
        n.add_visit(0.8);
        assert_eq!(n.q(), 0.8);
    }

    #[test]
    fn test_gumbel_noise() {
        let v = MCTS::gumbel_noise(10);
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_dirichlet_noise() {
        let v = MCTS::dirichlet_noise(5, 1.0);
        assert_eq!(v.len(), 5);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_legal() {
        let logits = vec![0.0f32; NUM_POSITIONS];
        let legal = vec![(0, 0), (0, 1)];
        let probs = MCTS::softmax_legal(&logits, &legal);
        let sum: f32 = legal
            .iter()
            .map(|&(r, c)| probs[Board::pos_to_idx(r, c)])
            .sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backprop_consistency() {
        let mut m = MCTS::new();
        let n1 = m.push_node(Node::new(0.5, 0.5f32.ln(), 0.0));
        let n2 = m.push_node(Node::new(0.3, 0.3f32.ln(), 0.0));
        m.backprop_path(&[(0, 0, n1), (n1, 1, n2)], 0.8);
        assert!((m.node(n1).visit_count_f32() - 1.0).abs() < 1e-6);
        assert!((m.node(n2).total_value - 0.8).abs() < 1e-6);
        assert!((m.node(n1).total_value + 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_gumbel_config() {
        let c = GumbelConfig::pure_gumbel(64);
        assert!(c.pure_gumbel_noise);
        assert_eq!(c.num_simulations, 64);
        assert_eq!(c.select_temperature, 1.0);
    }

    // ── PUCT 辅助函数 ──

    #[test]
    fn test_puct_bias_values() {
        let b0 = MCTS::puct_bias(0.0);
        let expected0 = PUCT_INIT + ((1.0 + 0.0 + PUCT_BASE) / PUCT_BASE).ln();
        assert!((b0 - expected0).abs() < 1e-6);
        let b100 = MCTS::puct_bias(100.0);
        assert!(b100 > b0);
    }

    #[test]
    fn test_init_q_value_no_visits() {
        let m = MCTS::new();
        assert!((m.init_q_value(0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_init_q_value_with_visits() {
        let mut m = MCTS::new();
        let child = m.push_node(Node::new(0.5, 0.5f32.ln(), 0.0));
        m.node_mut(child).add_visit(0.7);
        m.root_mut().children[0] = Some(child);
        let expected = (0.7 - 1.0) / (1.0 + 1.0);
        assert!((m.init_q_value(0) - expected).abs() < 1e-6);
    }

    // ── Gumbel budget ──

    #[test]
    fn test_gumbel_budget_formula() {
        assert_eq!(MCTS::gumbel_budget(32, 16), 1);
        assert_eq!(MCTS::gumbel_budget(200, 16), 3);
    }

    #[test]
    fn test_gumbel_budget_halved_uses_initial_log2() {
        assert_eq!(MCTS::gumbel_budget_halved(32, 16, 16), 1);
        assert_eq!(MCTS::gumbel_budget_halved(32, 8, 16), 2);
        // 关键：halving 到 4 时仍用初始 log2(16)=4 而非 log2(4)=2
        assert_eq!(MCTS::gumbel_budget_halved(32, 4, 16), 4);
    }

    // ── sigma_score ──

    #[test]
    fn test_sigma_score_zero_visits() {
        let n = Node::new(0.5, 0.5f32.ln(), 0.3);
        let config = GumbelConfig::default();
        assert!(MCTS::sigma_score(&n, 10.0, &config).is_infinite());
        assert!(MCTS::sigma_score(&n, 10.0, &config) < 0.0);
    }

    #[test]
    fn test_sigma_score_with_visits() {
        let mut n = Node::new(0.6, 0.6f32.ln(), 0.4);
        n.add_visit(0.8);
        n.add_visit(0.6);
        let config = GumbelConfig::default();
        let c = config.sigma_visit_c;
        let scale = config.sigma_scale_c;
        let max_n = 5.0;
        let expected = n.policy_logit + (c + max_n) * scale * n.q();
        assert!((MCTS::sigma_score(&n, max_n, &config) - expected).abs() < 1e-6);
    }

    // ── max_root_count ──

    #[test]
    fn test_max_root_count() {
        let mut m = MCTS::new();
        let c1 = m.push_node(Node::new(0.5, 0.5f32.ln(), 0.0));
        let c2 = m.push_node(Node::new(0.3, 0.3f32.ln(), 0.0));
        m.root_mut().children[0] = Some(c1);
        m.root_mut().children[1] = Some(c2);
        m.node_mut(c1).add_visit(0.5);
        m.node_mut(c1).add_visit(0.5);
        m.node_mut(c2).add_visit(0.3);
        assert!((MCTS::max_root_count(&m) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_root_count_empty() {
        assert_eq!(MCTS::max_root_count(&MCTS::new()), 0.0);
    }

    // ── backprop 多层 ──

    #[test]
    fn test_backprop_sign_flip_deep() {
        let mut m = MCTS::new();
        let n1 = m.push_node(Node::new(0.5, 0.5f32.ln(), 0.0));
        let n2 = m.push_node(Node::new(0.3, 0.3f32.ln(), 0.0));
        let n3 = m.push_node(Node::new(0.2, 0.2f32.ln(), 0.0));
        m.root_mut().children[0] = Some(n1);
        m.node_mut(n1).children[0] = Some(n2);
        m.node_mut(n2).children[0] = Some(n3);
        m.backprop_path(&[(0, 0, n1), (n1, 0, n2), (n2, 0, n3)], 0.8);
        assert!((m.node(n3).q() - 0.8).abs() < 1e-6);
        assert!((m.node(n2).q() + 0.8).abs() < 1e-6);
        assert!((m.node(n1).q() - 0.8).abs() < 1e-6);
    }

    // ── Completed Q policy ──

    #[test]
    fn test_build_completed_q_policy_normalizes() {
        let mut m = MCTS::new();
        let c1 = m.push_node(Node::new(0.6, 0.6f32.ln(), 0.1));
        let c2 = m.push_node(Node::new(0.4, 0.4f32.ln(), 0.2));
        m.root_mut().children[0] = Some(c1);
        m.root_mut().children[1] = Some(c2);
        m.node_mut(c1).add_visit(0.8);
        m.node_mut(c2).add_visit(-0.3);
        let (policy, root_value) = m.build_completed_q_policy(0.0, 4, &GumbelConfig::default());
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "policy sum = {}", sum);
        assert!(root_value >= -1.0 && root_value <= 1.0);
    }

    #[test]
    fn test_build_completed_q_policy_no_visits() {
        let m = MCTS::new();
        let (policy, root_value) = m.build_completed_q_policy(0.5, 4, &GumbelConfig::default());
        assert!(policy.iter().all(|&p| p == 0.0));
        assert!((root_value - 0.5).abs() < 1e-6);
    }

    // ── 集成测试：完整 search 流程 ──

    struct UniformEvaluator;

    impl Evaluator for UniformEvaluator {
        fn evaluate_batch(&self, states: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
            let policies: Vec<Vec<f32>> =
                states.iter().map(|_| vec![0.0f32; NUM_POSITIONS]).collect();
            let values: Vec<f32> = states.iter().map(|_| 0.0f32).collect();
            (policies, values)
        }
    }

    #[test]
    fn test_search_basic() {
        let mut board = Board::new();
        let config = GumbelConfig {
            num_simulations: 16,
            sample_size: 8,
            pure_gumbel_noise: true,
            ..Default::default()
        };
        let mut mcts = MCTS::new();
        let result = mcts.search(&mut board, &UniformEvaluator, &config);
        assert!(result.best_move < NUM_POSITIONS);
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "policy sum = {}", sum);
        assert!(result.root_value >= -1.0 && result.root_value <= 1.0);
    }

    #[test]
    fn test_search_no_panic_on_revisit() {
        let mut board = Board::new();
        let config = GumbelConfig {
            num_simulations: 32,
            sample_size: 4,
            pure_gumbel_noise: true,
            ..Default::default()
        };
        let mut mcts = MCTS::new();
        let result = mcts.search(&mut board, &UniformEvaluator, &config);
        assert!(result.best_move < NUM_POSITIONS);
    }

    #[test]
    fn test_search_sequential_halving_triggers() {
        let mut board = Board::new();
        let config = GumbelConfig {
            num_simulations: 50,
            sample_size: 16,
            pure_gumbel_noise: true,
            ..Default::default()
        };
        let mut mcts = MCTS::new();
        let result = mcts.search(&mut board, &UniformEvaluator, &config);
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_search_candidates_visited() {
        let mut board = Board::new();
        let config = GumbelConfig {
            num_simulations: 20,
            sample_size: 8,
            pure_gumbel_noise: true,
            ..Default::default()
        };
        let mut mcts = MCTS::new();
        let _ = mcts.search(&mut board, &UniformEvaluator, &config);
        let visited: Vec<_> = mcts
            .root()
            .children
            .iter()
            .flatten()
            .filter(|&&ci| mcts.node(ci).visit_count > 0)
            .collect();
        // 第一次模拟从 root 走标准 PUCT，可能访问非候选节点，
        // 所以 visited 数量可能超过 sample_size
        assert!(visited.len() > 0, "no children visited at all");
    }

    // ── 边界条件 ──

    #[test]
    fn test_empty_legal_moves() {
        let mut board = Board::new();
        board.game_over = true;
        let mut mcts = MCTS::new();
        let result = mcts.search(&mut board, &UniformEvaluator, &GumbelConfig::default());
        assert_eq!(result.best_move, NUM_POSITIONS);
    }

    #[test]
    fn test_logit_without_noise_roundtrip() {
        let prior = 0.6f32;
        let clean_logit = prior.ln();
        let noise = 0.5;
        let n = Node::new(prior, clean_logit, noise);
        assert!((n.policy_logit - (clean_logit + noise)).abs() < 1e-6);
        assert!((n.logit_without_noise() - clean_logit).abs() < 1e-6);
    }
}
