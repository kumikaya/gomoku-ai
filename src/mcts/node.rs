//! 蒙特卡洛树搜索 (MCTS) — Gumbel Zero (Gumbel AlphaZero)
//!
//! 严格对齐 minizero 的 Gumbel AlphaZero 实现：
//! - Gumbel 噪声加在 logit 空间（`policy_logit = ln(prior) + gumbel_noise`）
//! - Sequential Halving 用带噪声 logit + normalized Q 做候选排序
//! - Completed Q policy 用干净 logit（`logit_without_noise = logit - noise`）
//! - 动作选择用 softmax over visit counts（temperature=1.0, value_threshold=0.1）
//! - path = Vec<usize>（纯节点索引序列，root 在 path[0]，对齐 minizero）
//!
//! ## 噪声模式
//!
//! - **纯 Gumbel**（`GumbelConfig::pure_gumbel()`）：
//!   prior = NN 干净输出，`policy_logit = ln(prior) + gumbel_noise`。
//!
//! - **混合噪声**（`GumbelConfig::mixed()`）：
//!   prior = (1-ε)*NN + ε*Dirichlet，`policy_logit = ln(NN_prior) + gumbel_noise`。
//!   PUCT walk 用 noisy prior。

use super::table::Table;
use crate::game::board::{Board, Color, ENCODE_LEN, NUM_POSITIONS};
use crate::inference::Evaluator;
use rand::RngExt;

/// 对数计算的最小概率下限，防止 ln(0) = -inf
const MIN_PROB: f32 = 1e-15;

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
    /// 每次 GPU forward 攒批的叶子数（默认 8，对齐 minizero actor_mcts_think_batch_size）
    pub think_batch_size: usize,
    pub sigma_visit_c: f32,
    pub sigma_scale_c: f32,
    pub pure_gumbel_noise: bool,
    /// 动作选择 softmax 温度（minizero 默认 1.0）
    pub select_temperature: f32,
    /// 根节点策略先验 softmax 温度（>1 抑制过早收敛，对齐 KataGo）
    /// 仅在根节点参与 `ln(prior)` 之前对 NN 先验做 `probs^(1/T) / Z` 缩放。
    pub root_policy_temperature: f32,
    /// 是否启用 Dynamic Variance-Scaled cPUCT
    pub dynamic_cpuct: bool,
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
            root_policy_temperature: 1.0,
            dynamic_cpuct: true,
            dirichlet_alpha: 0.1,
            dirichlet_epsilon: 0.25,
            think_batch_size: 8,
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
    /// Σ(value²)：用于计算节点效用方差（Dynamic Variance-Scaled cPUCT）
    pub value_sq_sum: f32,
    /// 先验概率（PUCT walk 使用；纯 Gumbel 模式 = NN prior，混合模式 = noisy prior）
    pub prior: f32,
    /// 带 Gumbel 噪声的 logit：ln(NN_prior) + gumbel_noise
    /// Sequential Halving 候选排序用这个。
    pub policy_logit: f32,
    /// Gumbel 噪声值（用于 recovered clean logit = policy_logit - gumbel_noise）
    pub gumbel_noise: f32,
    pub children: Table<usize>,
    pub expanded: bool,
    /// 该节点对应哪个玩家的回合
    pub player: Color,
    /// Virtual loss：攒批 evaluate 期间让 PUCT walk 发散
    pub virtual_loss: u32,
}

impl Node {
    pub fn new(prior: f32, gumbel_logit: f32, gumbel_noise: f32, player: Color) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            value_sq_sum: 0.0,
            prior,
            policy_logit: gumbel_logit + gumbel_noise,
            gumbel_noise,
            children: Table::new(NUM_POSITIONS),
            expanded: false,
            player,
            virtual_loss: 0,
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

    /// 效用方差（Dynamic Variance-Scaled cPUCT）。
    ///
    /// 对访问数较少的节点混入先验方差，避免除零和不合理的极值。
    /// 值域约 [0.04, 1.0]。
    #[inline]
    pub fn variance(&self) -> f32 {
        // 值域 [-1,1] 下合理的先验方差
        const PRIOR_VAR: f32 = 0.25;
        const PRIOR_WEIGHT: f32 = 3.0;
        if self.visit_count > 1 {
            let n = self.visit_count as f32;
            let mean = self.total_value / n;
            let sample_var = (self.value_sq_sum / n) - (mean * mean);
            ((sample_var * n) + (PRIOR_VAR * PRIOR_WEIGHT)) / (n + PRIOR_WEIGHT)
        } else {
            PRIOR_VAR
        }
    }

    #[inline]
    pub fn add_visit(&mut self, value: f32) {
        self.visit_count += 1;
        self.total_value += value;
        self.value_sq_sum += value * value;
    }

    /// visit_count 含 virtual loss（用于 PUCT walk 时让后续 walk 发散）
    #[inline]
    pub fn effective_visits(&self) -> u32 {
        self.visit_count + self.virtual_loss
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
    /// 根节点 NN 干净先验（不含 noise/temperature），用于 Policy Surprise Weighting
    pub root_nn_prior: Vec<f32>,
}

impl MCTS {
    pub fn new() -> Self {
        Self {
            arena: vec![Node::new(0.0, 0.0, 0.0, Color::Black)],
        }
    }

    #[inline]
    pub fn root(&self) -> &Node {
        &self.arena[0]
    }

    #[inline]
    pub fn root_mut(&mut self) -> &mut Node {
        &mut self.arena[0]
    }

    #[inline]
    pub fn node(&self, idx: usize) -> &Node {
        &self.arena[idx]
    }

    #[inline]
    fn node_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.arena[idx]
    }

    fn reset(&mut self, player: Color) {
        self.arena.clear();
        self.arena.push(Node::new(0.0, 0.0, 0.0, player));
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
        let child_player = self.node(parent).player.opponent();
        let mut child_ids = Vec::with_capacity(legal.len());
        for i in 0..legal.len() {
            let (prior, clean_logit) = child_data[i];
            child_ids.push(self.push_node(Node::new(
                prior,
                clean_logit,
                gumbel_noises[i],
                child_player,
            )));
        }
        let parent_node = self.node_mut(parent);
        for (i, &(r, c)) in legal.iter().enumerate() {
            let idx = Board::pos_to_idx(r, c);
            parent_node.children.set(idx, child_ids[i]);
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
        for c in node.children.values_copied() {
            let child = self.node(c);
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

    /// PUCT walk，返回从 `start_node_idx`（含）到叶子节点（含）的路径。
    /// 对齐 minizero `selectFromNode`：起始节点始终在 path[0]。
    /// PUCT 公式中 `n` 用 effective_visits（visit_count + virtual_loss），
    /// 使攒批期间的连续 walk 自然分到不同叶子。
    ///
    /// 若 `dynamic_cpuct` 启用，则 bias 乘以 `sqrt(child_variance)`，
    /// 使效用波动大的局面多探索（对齐 KataGo Dynamic Variance-Scaled cPUCT）。
    fn puct_walk(
        &self,
        start_node_idx: usize,
        board: &mut Board,
        legal: &mut Vec<(usize, usize)>,
        dynamic_cpuct: bool,
    ) -> Vec<usize> {
        // 方差缩放钳位：防止极端方差导致探索失控
        const VAR_CLAMP_MIN: f32 = 0.1;
        const VAR_CLAMP_MAX: f32 = 4.0;

        let mut path: Vec<usize> = vec![start_node_idx];
        let mut node_idx = start_node_idx;

        loop {
            if board.game_over || !self.node(node_idx).expanded {
                return path;
            }
            board.fill_legal_moves(legal);
            if legal.is_empty() {
                return path;
            }

            let node = self.node(node_idx);
            let total_simulation = node.effective_visits().saturating_sub(1) as f32;
            let bias = Self::puct_bias(total_simulation);
            let sqrt_total = total_simulation.sqrt();
            let init_q = self.init_q_value(node_idx);

            let children = &node.children;
            let mut best: Option<(usize, usize, f32, f32)> = None;
            for &(r, c) in legal.iter() {
                let idx = Board::pos_to_idx(r, c);
                if let Some(ci) = children[idx] {
                    let child = self.node(ci);
                    let n = child.effective_visits() as f32;
                    let q = if child.visit_count > 0 {
                        child.total_value / child.visit_count as f32
                    } else {
                        init_q
                    };
                    let var_scale = if dynamic_cpuct {
                        child.variance().sqrt().clamp(VAR_CLAMP_MIN, VAR_CLAMP_MAX)
                    } else {
                        1.0
                    };
                    let score = q + bias * var_scale * child.prior * sqrt_total / (1.0 + n);
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

            path.push(child_idx);
            board.play_idx(move_idx);
            node_idx = child_idx;
        }
    }

    // ── Gumbel Zero search ──

    pub fn search<E: Evaluator>(
        &mut self,
        board: &mut Board,
        evaluator: &E,
        config: &GumbelConfig,
        rng: &mut impl RngExt,
    ) -> SearchResult {
        self.reset(board.current_player);
        let legal_moves = board.legal_moves();
        if legal_moves.is_empty() {
            return SearchResult {
                best_move: NUM_POSITIONS,
                policy: vec![0.0; NUM_POSITIONS],
                root_value: 0.0,
                root_nn_prior: vec![0.0; NUM_POSITIONS],
            };
        }

        let num_sim = config.num_simulations;
        let sample_n = config.sample_size;

        // ── Phase 1: 根节点 NN 评估 ──
        let mut root_encoding = vec![0i32; ENCODE_LEN];
        board.encode_into(&mut root_encoding);
        let (root_logits, root_values) = evaluator.evaluate_batch(&[root_encoding]);
        let root_nn_value = root_values[0];

        let raw_policy_probs = Self::softmax_legal(&root_logits[0], &legal_moves);

        let policy_probs = Self::apply_root_temperature(
            &raw_policy_probs,
            &legal_moves,
            config.root_policy_temperature,
        );

        // 干净 logit：ln(NN prior)，加 epsilon 防止 prior=0 导致 -inf
        let clean_logits: Vec<f32> = legal_moves
            .iter()
            .map(|&(r, c)| policy_probs[Board::pos_to_idx(r, c)].max(MIN_PROB).ln())
            .collect();

        // Gumbel 噪声（所有场景下都生成）
        let gumbel_noises = Self::gumbel_noise(legal_moves.len(), rng);

        // ── Phase 1b: expand root ──
        let child_data: Vec<(f32, f32)> = if config.pure_gumbel_noise {
            legal_moves
                .iter()
                .enumerate()
                .map(|(i, &(r, c))| {
                    let idx = Board::pos_to_idx(r, c);
                    (policy_probs[idx], clean_logits[i])
                })
                .collect()
        } else {
            let dir = Self::dirichlet_noise(legal_moves.len(), config.dirichlet_alpha, rng);
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
        candidates.sort_by(|&a, &b| {
            self.node(b)
                .policy_logit
                .partial_cmp(&self.node(a).policy_logit)
                .unwrap()
        });
        if candidates.len() > sample_n {
            candidates.truncate(sample_n);
        }

        // ── Phase 3: 模拟循环（攒批 evaluate）──
        // 对标 minizero ZeroActor::step()：
        //   连续做 batch_size 次 PUCT walk → 攒特征 → 一次 GPU forward → 消费结果。
        //   virtual loss 让连续 walk 自然发散到不同叶子。
        let batch_size = (sample_n.min(config.think_batch_size)).max(1);

        let mut cur_sample = sample_n;
        let mut sim_budget = Self::gumbel_budget(num_sim, cur_sample);

        let mut sim_board = board.clone();
        let mut legal_buf = Vec::with_capacity(NUM_POSITIONS);
        let mut sim_i = 0;

        loop {
            if sim_i >= num_sim {
                break;
            }

            // ── 一轮批量：连续 walk 攒 batch_size 个叶子 ──
            let round = (num_sim - sim_i).min(batch_size);

            // 暂存本轮每个 walk 的上下文
            struct EvalCtx {
                path: Vec<usize>,
                encoding: Vec<i32>,
                legal: Vec<(usize, usize)>,
                game_over: bool,
                winner: Option<Color>,
            }
            let mut contexts: Vec<EvalCtx> = Vec::with_capacity(round);
            let mut encodings: Vec<Vec<i32>> = Vec::with_capacity(round);

            for _ in 0..round {
                // ── selection ──
                let path: Vec<usize>;

                if sim_i == 0 {
                    sim_board.clone_from(board);
                    path = self.puct_walk(0, &mut sim_board, &mut legal_buf, config.dynamic_cpuct);
                } else {
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

                    sim_board.clone_from(board);
                    let move_idx = self.root().children.position_of(&start_ci).unwrap_or(0);
                    sim_board.play_idx(move_idx);

                    let p_from = self.puct_walk(
                        start_ci,
                        &mut sim_board,
                        &mut legal_buf,
                        config.dynamic_cpuct,
                    );
                    let mut full = vec![0];
                    full.extend(p_from);
                    path = full;
                }

                // add virtual loss 到这条路径上
                for &ni in &path {
                    self.node_mut(ni).virtual_loss += 1;
                }

                let leaf_idx = *path.last().unwrap();

                if sim_board.game_over {
                    contexts.push(EvalCtx {
                        path,
                        encoding: Vec::new(),
                        legal: Vec::new(),
                        game_over: true,
                        winner: sim_board.winner,
                    });
                } else if !self.node(leaf_idx).expanded {
                    let mut encoding = vec![0i32; ENCODE_LEN];
                    sim_board.encode_into(&mut encoding);
                    sim_board.fill_legal_moves(&mut legal_buf);
                    let legal_leaf = std::mem::take(&mut legal_buf);
                    encodings.push(encoding.clone());
                    contexts.push(EvalCtx {
                        path,
                        encoding,
                        legal: legal_leaf,
                        game_over: false,
                        winner: None,
                    });
                } else {
                    // 已展开的节点（理论上不应该到这里，但保留处理）
                    contexts.push(EvalCtx {
                        path,
                        encoding: Vec::new(),
                        legal: Vec::new(),
                        game_over: false,
                        winner: None,
                    });
                }

                sim_i += 1;
            }

            // ── 批量 GPU forward ──
            let eval_result = if !encodings.is_empty() {
                Some(evaluator.evaluate_batch(&encodings))
            } else {
                None
            };

            // ── 消费结果：expand + backprop，清除 virtual loss ──
            let mut batch_result_idx = 0;
            for ctx in contexts.into_iter() {
                // 清除 virtual loss
                for &ni in &ctx.path {
                    self.node_mut(ni).virtual_loss = self.node(ni).virtual_loss.saturating_sub(1);
                }

                if ctx.game_over {
                    let v = match ctx.winner {
                        Some(_) => WIN_VALUE,
                        None => DRAW_VALUE,
                    };
                    self.backprop_path(&ctx.path, v);
                } else if !ctx.encoding.is_empty() {
                    let (policies_batch, values_batch) = eval_result.as_ref().unwrap();
                    let probs = Self::softmax_legal(&policies_batch[batch_result_idx], &ctx.legal);

                    let leaf_idx = *ctx.path.last().unwrap();
                    let gumbel_noises_leaf = vec![0.0f32; ctx.legal.len()];
                    let child_data_leaf: Vec<(f32, f32)> = ctx
                        .legal
                        .iter()
                        .map(|&(r, c)| {
                            let idx = Board::pos_to_idx(r, c);
                            let p = probs[idx];
                            (p, p.max(MIN_PROB).ln())
                        })
                        .collect();
                    self.expand_children(
                        leaf_idx,
                        &ctx.legal,
                        &gumbel_noises_leaf,
                        &child_data_leaf,
                    );
                    self.node_mut(leaf_idx).expanded = true;
                    self.backprop_path(&ctx.path, -values_batch[batch_result_idx]);
                    batch_result_idx += 1;
                }
                // else: 已展开节点，无操作
            }

            // ── sequentialHalving（每轮结束后检查一次）──
            let all_reached = candidates
                .iter()
                .all(|&ci| self.node(ci).visit_count_f32() >= sim_budget as f32);

            if all_reached {
                let next_budget = Self::gumbel_budget_halved(num_sim, cur_sample, sample_n);
                if next_budget > 0 && cur_sample > 2 {
                    cur_sample /= 2;
                    let max_n = Self::max_root_count(self);
                    candidates.sort_by(|&a, &b| {
                        let sa = Self::sigma_score(self.node(a), max_n, config);
                        let sb = Self::sigma_score(self.node(b), max_n, config);
                        sb.partial_cmp(&sa).unwrap()
                    });
                    if candidates.len() > cur_sample {
                        candidates.truncate(cur_sample);
                    }
                    let base = self.node(candidates[0]).visit_count;
                    sim_budget = base + next_budget;
                }
            }
        }

        // ── Phase 4: 构建 completed Q 策略（干净 logit） ──
        let (policy, root_value) = self.build_completed_q_policy(root_nn_value, num_sim, config);

        // ── Phase 5: 动作选择（softmax over visit counts） ──
        let best_move = self.select_by_softmax_count(config.select_temperature, rng);

        SearchResult {
            best_move,
            policy,
            root_value,
            root_nn_prior: raw_policy_probs,
        }
    }

    // ── Gumbel 辅助 ──

    /// 初始 budget：B = N / (log2(sample_size) * sample_size)
    fn gumbel_budget(num_sim: usize, sample_n: usize) -> u32 {
        let d = (sample_n as f32).log2() * sample_n as f32;
        (num_sim as f32 / d).floor().max(1.0) as u32
    }

    /// Halving budget：B' = N / (log2(initial_sample_size) * cur_sample / 2)
    fn gumbel_budget_halved(num_sim: usize, cur_sample: usize, initial_sample_size: usize) -> u32 {
        let d = (initial_sample_size as f32).log2() * (cur_sample as f32) / 2.0;
        (num_sim as f32 / d).floor() as u32
    }

    fn max_root_count(mcts: &MCTS) -> f32 {
        mcts.root()
            .children
            .values()
            .map(|&ci| mcts.node(ci).visit_count_f32())
            .fold(0.0, f32::max)
    }

    /// σ-score：带噪声 logit + Q 项。
    fn sigma_score(node: &Node, max_n: f32, config: &GumbelConfig) -> f32 {
        if node.visit_count == 0 {
            return f32::NEG_INFINITY;
        }
        let var_scale = if config.dynamic_cpuct {
            node.variance().sqrt().clamp(0.1, 4.0)
        } else {
            1.0
        };
        node.policy_logit
            + (config.sigma_visit_c + max_n) * config.sigma_scale_c * var_scale * node.q()
    }

    /// Completed Q policy：用**去噪声** logit + Q 项，再做 softmax。
    fn build_completed_q_policy(
        &self,
        root_nn_value: f32,
        num_sim: usize,
        config: &GumbelConfig,
    ) -> (Vec<f32>, f32) {
        let children = &self.root().children;

        let mut pi_sum = 0.0f32;
        let mut q_sum = 0.0f32;
        for c in children.values_copied() {
            let child = self.node(c);
            if child.visit_count > 0 {
                pi_sum += child.prior;
                q_sum += child.prior * child.q();
            }
        }

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
        for (idx, ci) in children.occupied() {
            let child = self.node(*ci);
            let value = if child.visit_count > 0 {
                child.q()
            } else {
                non_visited_value
            };
            let score = child.logit_without_noise() + (sv + max_n) * sc * value;
            scores.push((idx, score));
            if score > max_score {
                max_score = score;
            }
        }

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

        let sum_n: f32 = children
            .values()
            .map(|&ci| self.node(ci).visit_count_f32())
            .sum();
        let root_value = if sum_n > 0.0 {
            children
                .values()
                .filter_map(|&ci| {
                    let child = self.node(ci);
                    if child.visit_count > 0 {
                        Some(child.visit_count_f32() / sum_n * child.q())
                    } else {
                        None
                    }
                })
                .sum()
        } else {
            root_nn_value
        };

        (policy, root_value)
    }

    /// 动作选择：softmax over visit_counts，temperature 默认 1.0。
    fn select_by_softmax_count(&self, temperature: f32, rng: &mut impl RngExt) -> usize {
        use rand::distr::{Distribution, weighted::WeightedIndex};
        let children = &self.root().children;

        let max_count_child = children
            .iter()
            .filter_map(|(_, c)| c.copied())
            .max_by_key(|ci| self.node(*ci).visit_count);
        let threshold_q = max_count_child
            .map(|ci| self.node(ci).q() - 0.1)
            .unwrap_or(f32::NEG_INFINITY);

        let probs: Vec<f64> = children
            .iter()
            .map(|(_, c)| {
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
            dist.sample(rng)
        } else {
            let (policy, _) = self.build_completed_q_policy(0.0, 1, &GumbelConfig::default());
            policy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// 反向传播 value 到路径上所有节点（含 root）。
    /// 对齐 minizero `backup`：node_path 包含 root，依次 flip value。
    fn backprop_path(&mut self, path: &[usize], value: f32) {
        // 叶子节点是 path 最后一个
        let leaf_player = self.node(*path.last().unwrap()).player;
        for &node_idx in path.iter().rev() {
            let node_player = self.node(node_idx).player;
            let v = if node_player == leaf_player {
                value
            } else {
                -value
            };
            self.node_mut(node_idx).add_visit(v);
        }
    }

    // ── utils ──

    /// Root Policy Softmax Temperature (KataGo):
    /// Scale root priors via `p^(1/T) / Z` before computing ln(prior).
    /// T=1.0 returns unchanged. T>1 pushes distribution toward uniform.
    fn apply_root_temperature(
        probs: &[f32],
        legal: &[(usize, usize)],
        temperature: f32,
    ) -> Vec<f32> {
        if temperature == 1.0 {
            return probs.to_vec();
        }
        let mut scaled = vec![0.0f32; NUM_POSITIONS];
        let mut sum = 0.0;
        for &(r, c) in legal {
            let idx = Board::pos_to_idx(r, c);
            let v = probs[idx].max(MIN_PROB).powf(1.0 / temperature);
            scaled[idx] = v;
            sum += v;
        }
        if sum > 0.0 {
            for &(r, c) in legal {
                scaled[Board::pos_to_idx(r, c)] /= sum;
            }
        }
        scaled
    }

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

    fn dirichlet_noise(n: usize, alpha: f32, rng: &mut impl RngExt) -> Vec<f32> {
        use rand::distr::Distribution;
        let gamma = rand_distr::Gamma::new(alpha, 1.0).unwrap();
        let mut v: Vec<f32> = (0..n).map(|_| gamma.sample(rng)).collect();
        let s: f32 = v.iter().sum();
        for x in &mut v {
            *x /= s;
        }
        v
    }

    fn gumbel_noise(n: usize, rng: &mut impl RngExt) -> Vec<f32> {
        use rand::distr::Distribution;
        let gumbel = rand_distr::Gumbel::new(0.0, 1.0).unwrap();
        (0..n)
            .map(|_| (gumbel.sample(rng) as f32).clamp(-40.0_f32, 40.0_f32))
            .collect()
    }
}
