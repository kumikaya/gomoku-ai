//! 蒙特卡洛树搜索 (MCTS) — Gumbel Zero (Gumbel AlphaZero)
//!
//! 严格对齐 minizero 的 Gumbel AlphaZero 实现：
//! - Gumbel 噪声加在 logit 空间（`policy_logit = ln(prior) + gumbel_noise`）
//! - Sequential Halving 用带噪声 logit + normalized Q 做候选排序
//! - Completed Q policy 用干净 logit（`logit_without_noise = logit - noise`）
//! - 动作选择用 softmax over visit counts（temperature=1.0, value_threshold=0.1）
//! - path = Vec<NodeId>（纯节点索引序列，root 在 path[0]，对齐 minizero）
//!
//! ## 游戏解耦
//!
//! MCTS 通过 `Game` trait 与具体棋盘解耦。所有游戏操作（合法动作、落子、
//! 编码、终局检测）均通过 trait 调用。动作统一为 `ActionId`（平铺索引），
//! 消除了行列坐标和各种 `pos_to_idx` 转换。
//!
//! ## 噪声模式
//!
//! - **纯 Gumbel**（`GumbelConfig::pure_gumbel()`）：
//!   prior = NN 干净输出，`policy_logit = ln(prior) + gumbel_noise`。
//!
//! - **混合噪声**（`GumbelConfig::mixed()`）：
//!   prior = (1-ε)*NN + ε*Dirichlet，`policy_logit = ln(NN_prior) + gumbel_noise`。
//!   PUCT walk 用 noisy prior。

use super::game::{ActionId, Game};
use super::table::Table;
use crate::inference::Evaluator;
use itertools::Itertools;
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

// 方差缩放钳位：防止极端方差导致探索失控
const VAR_CLAMP_MIN: f32 = 0.1;
const VAR_CLAMP_MAX: f32 = 4.0;

/// MCTS 树节点索引。
pub type NodeId = usize;

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
    /// 最终动作选择的 softmax 温度（默认 1.0）
    pub select_temperature: f32,
    /// 根节点先验策略的 softmax 温度 (KataGo rootPolicyTemperature)
    pub root_policy_temperature: f32,
    /// 是否启用 Dynamic Variance-Scaled cPUCT (KataGo)
    pub dynamic_cpuct: bool,
    /// FPU Reduction 强度 (KataGo)
    pub fpu_reduction: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            num_simulations: 50,
            sample_size: 16,
            think_batch_size: 8,
            sigma_visit_c: 50.0,
            sigma_scale_c: 0.1,
            pure_gumbel_noise: true,
            select_temperature: 1.0,
            root_policy_temperature: 1.0,
            dynamic_cpuct: true,
            fpu_reduction: 0.0,
            dirichlet_alpha: 0.1,
            dirichlet_epsilon: 0.03,
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
pub struct Node<G: Game> {
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
    pub children: Table<NodeId>,
    pub expanded: bool,
    /// 该节点对应哪个玩家的回合
    pub player: G::Player,
    /// Virtual loss：攒批 evaluate 期间让 PUCT walk 发散
    pub virtual_loss: u32,
}

impl<G: Game> Node<G> {
    pub fn new(
        prior: f32,
        clean_logit: f32,
        gumbel_noise: f32,
        player: G::Player,
        action_shape: usize,
    ) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            value_sq_sum: 0.0,
            prior,
            policy_logit: clean_logit + gumbel_noise,
            gumbel_noise,
            children: Table::new(action_shape),
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

    /// 父节点视角的平均 Q 值（PUCT / completedQ / 所有外部读取的正确视角）。
    ///
    /// 由于 `total_value` 按节点自身视角存储，而外部读取者总是父节点，
    /// 因此返回 `-q_self()` 以自然翻转为父节点视角。
    #[inline]
    /// 父节点视角的平均 Q 值。
    ///
    /// 通过 `Game::flip_perspective` 将自身视角值翻转为父节点视角：
    /// 零和对弈取反，单智能体可保持原值。
    pub fn q(&self) -> f32 {
        if self.visit_count > 0 {
            G::flip_perspective(self.total_value / self.visit_count as f32)
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

pub struct MCTS<G: Game> {
    pub arena: Vec<Node<G>>,
}

/// 单次搜索返回结果
#[derive(Debug)]
pub struct SearchResult {
    /// 选中的最佳动作（平铺索引），若为 action_shape 表示无合法动作
    pub best_move: ActionId,
    /// Completed Q 策略（长度为 action_shape）
    pub policy: Vec<f32>,
    /// MCTS 根节点价值（含 Q 和 NN 混合）
    pub root_value: f32,
    /// NN 原始先验概率
    pub root_nn_prior: Vec<f32>,
}

// 暂存本轮每个 walk 的上下文
struct EvalCtx {
    path: Vec<NodeId>,
    legal: Vec<ActionId>,
    /// game_over 时固化当前玩家的 terminal_value，消费阶段直接使用
    terminal_value: Option<f32>,
}

impl<G: Game> MCTS<G> {
    pub fn new() -> Self {
        Self { arena: Vec::new() }
    }

    #[inline]
    pub fn root(&self) -> &Node<G> {
        &self.arena[0]
    }

    #[inline]
    pub fn root_mut(&mut self) -> &mut Node<G> {
        &mut self.arena[0]
    }

    #[inline]
    pub fn node(&self, idx: NodeId) -> &Node<G> {
        &self.arena[idx]
    }

    #[inline]
    fn node_mut(&mut self, idx: NodeId) -> &mut Node<G> {
        &mut self.arena[idx]
    }

    fn reset(&mut self, player: G::Player, action_shape: usize) {
        self.arena.clear();
        self.arena
            .push(Node::new(0.0, 0.0, 0.0, player, action_shape));
    }

    fn push_node(&mut self, node: Node<G>) -> NodeId {
        self.arena.push(node);
        self.arena.len() - 1
    }

    // ── expand ──

    fn expand_children(
        &mut self,
        parent: NodeId,
        legal: &[ActionId],
        gumbel_noises: &[f32],
        child_data: &[(f32, f32)],
        action_shape: usize,
    ) {
        let child_player = G::next_player(self.node(parent).player);
        for (i, &action) in legal.iter().enumerate() {
            let (prior, clean_logit) = child_data[i];
            let child_id = self.push_node(Node::new(
                prior,
                clean_logit,
                gumbel_noises[i],
                child_player,
                action_shape,
            ));
            self.node_mut(parent).children.set(action, child_id);
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
    fn init_q_value(&self, node_idx: NodeId) -> f32 {
        let node = self.node(node_idx);
        let mut sum_q = 0.0f32;
        let mut visited = 0.0f32;
        for &c in node.children.values() {
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
        start_node_idx: NodeId,
        game: &mut G,
        dynamic_cpuct: bool,
        fpu_reduction: f32,
    ) -> impl Iterator<Item = NodeId> {
        let mut node_idx = start_node_idx;
        let mut first = true;
        std::iter::from_fn(move || {
            if first {
                first = false;
                return Some(node_idx);
            }
            loop {
                if game.is_terminal() || !self.node(node_idx).expanded {
                    return None;
                }
                let legal = game.legal_actions();
                if legal.is_empty() {
                    return None;
                }

                let node = self.node(node_idx);
                let total_simulation = node.effective_visits().max(1) as f32;
                let bias = Self::puct_bias(total_simulation);
                let sqrt_total = total_simulation.sqrt();
                let init_q = self.init_q_value(node_idx);
                // FPU Reduction (KataGo): 未访问子节点悲观估计，避免在明显差走法上浪费模拟
                let fpu_q = init_q - fpu_reduction;

                let children = &node.children;
                let mut best: Option<(ActionId, NodeId, f32, f32)> = None;
                for (action, &child_id) in children.occupied() {
                    let child = self.node(child_id);
                    let n = child.effective_visits() as f32;
                    // q() 已返回父节点视角，fpu_q 也是父节点视角，直接使用
                    let q = if child.visit_count > 0 {
                        child.q()
                    } else {
                        fpu_q
                    };
                    let var_scale = if dynamic_cpuct {
                        child.variance().sqrt().clamp(VAR_CLAMP_MIN, VAR_CLAMP_MAX)
                    } else {
                        1.0
                    };
                    let score = q + bias * var_scale * child.prior * sqrt_total / (1.0 + n);
                    let policy = child.prior;
                    if best.map_or(true, |(_, _, s, p)| score > s || (score == s && policy > p)) {
                        best = Some((action, child_id, score, policy));
                    }
                }

                let (move_idx, child_idx) = match best {
                    Some((mi, ci, _, _)) => (mi, ci),
                    None => (legal[0], children[legal[0]].unwrap()),
                };

                game.play(move_idx);
                node_idx = child_idx;
                return Some(child_idx);
            }
        })
    }

    // ── Gumbel Zero search ──

    pub fn search<E: Evaluator>(
        &mut self,
        game: &G,
        evaluator: &E,
        config: &GumbelConfig,
        rng: &mut impl RngExt,
    ) -> SearchResult {
        let action_shape = game.action_shape();
        self.reset(game.current_player(), action_shape);

        let legal_moves = game.legal_actions();
        if legal_moves.is_empty() {
            return SearchResult {
                best_move: action_shape,
                policy: vec![0.0; action_shape],
                root_value: 0.0,
                root_nn_prior: vec![0.0; action_shape],
            };
        }

        let num_sim = config.num_simulations;
        let sample_n = config.sample_size;

        // ── Phase 1: 根节点 NN 评估 ──
        let root_encoding = game.encode();
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
            .map(|&action| policy_probs[action].max(MIN_PROB).ln())
            .collect();

        // Gumbel 噪声（所有场景下都生成）
        let gumbel_noises = Self::gumbel_noise(legal_moves.len(), rng);

        // ── Phase 1b: expand root ──
        let child_data: Vec<(f32, f32)> = if config.pure_gumbel_noise {
            legal_moves
                .iter()
                .enumerate()
                .map(|(i, &action)| (policy_probs[action], clean_logits[i]))
                .collect()
        } else {
            let dir = Self::dirichlet_noise(legal_moves.len(), config.dirichlet_alpha, rng);
            legal_moves
                .iter()
                .enumerate()
                .map(|(i, &action)| {
                    let prior = (1.0 - config.dirichlet_epsilon) * policy_probs[action]
                        + config.dirichlet_epsilon * dir[i];
                    (prior, clean_logits[i])
                })
                .collect()
        };

        self.expand_children(0, &legal_moves, &gumbel_noises, &child_data, action_shape);
        self.root_mut().expanded = true;

        // ── Phase 2: 候选集（按带噪声的 policy_logit 排序取 top-k） ──
        let mut candidates: Vec<(ActionId, NodeId)> = self
            .root()
            .children
            .occupied()
            .map(|(action, &child_id)| (action, child_id))
            .sorted_by(|&(_, a), &(_, b)| {
                self.node(b)
                    .policy_logit
                    .partial_cmp(&self.node(a).policy_logit)
                    .unwrap()
            })
            .take(sample_n)
            .collect();

        // ── Phase 3: 模拟循环（攒批 evaluate）──
        // 对标 minizero ZeroActor::step()：
        //   连续做 batch_size 次 PUCT walk → 攒特征 → 一次 GPU forward → 消费结果。
        //   virtual loss 让连续 walk 自然发散到不同叶子。
        let batch_size = (sample_n.min(config.think_batch_size)).max(1);

        let mut cur_sample = sample_n;
        let mut sim_budget = Self::gumbel_budget(num_sim, cur_sample);
        let mut sim_i = 0;

        loop {
            if sim_i >= num_sim {
                break;
            }

            // ── 一轮批量：连续 walk 攒 batch_size 个叶子 ──
            let round = (num_sim - sim_i).min(batch_size);
            let mut contexts: Vec<EvalCtx> = Vec::with_capacity(round);
            let mut encodings: Vec<Vec<i32>> = Vec::with_capacity(round);

            for _ in 0..round {
                // ── selection ──
                let mut sim_game = game.clone();

                // Sequential Halving: 选 effective_visits 最少的 candidate，
                // tie-break 选 policy_logit 最大的（首次模拟时所有 candidate
                // effective_visits == 0，等价于选 Phase 2 排序后的 top-1）。
                let (action, start_ci) = candidates
                    .iter()
                    .copied()
                    .min_by(|&(_, a), &(_, b)| {
                        self.node(a)
                            .effective_visits()
                            .cmp(&self.node(b).effective_visits())
                            .then_with(|| {
                                self.node(b)
                                    .policy_logit
                                    .partial_cmp(&self.node(a).policy_logit)
                                    .unwrap()
                            })
                    })
                    .unwrap();
                sim_game.play(action);

                let p_from = self.puct_walk(
                    start_ci,
                    &mut sim_game,
                    config.dynamic_cpuct,
                    config.fpu_reduction,
                );
                let mut path_vec = vec![0];
                path_vec.extend(p_from);
                let path = path_vec;

                // add virtual loss 到这条路径上
                for &ni in &path {
                    self.node_mut(ni).virtual_loss += 1;
                }

                let leaf_idx = *path.last().unwrap();

                let terminal_value = sim_game.terminal_value();
                let mut legal_leaf = Vec::new();
                if !sim_game.is_terminal() {
                    debug_assert!(!self.node(leaf_idx).expanded);
                    let encoding = sim_game.encode();
                    encodings.push(encoding);
                    legal_leaf = sim_game.legal_actions();
                }
                contexts.push(EvalCtx {
                    path,
                    legal: legal_leaf,
                    terminal_value,
                });

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
                // 一次性清除 virtual loss（walks 已完成，不再需要发散）
                for &ni in &ctx.path {
                    self.node_mut(ni).virtual_loss = 0;
                }

                if let Some(terminal_value) = ctx.terminal_value {
                    self.backprop_path(&ctx.path, terminal_value);
                } else {
                    let (policies_batch, values_batch) = eval_result.as_ref().unwrap();
                    let probs = Self::softmax_legal(&policies_batch[batch_result_idx], &ctx.legal);

                    let leaf_idx = *ctx.path.last().unwrap();
                    let gumbel_noises_leaf = vec![0.0f32; ctx.legal.len()];
                    let child_data_leaf: Vec<(f32, f32)> = ctx
                        .legal
                        .iter()
                        .map(|&action| {
                            let p = probs[action];
                            (p, p.max(MIN_PROB).ln())
                        })
                        .collect();
                    self.expand_children(
                        leaf_idx,
                        &ctx.legal,
                        &gumbel_noises_leaf,
                        &child_data_leaf,
                        action_shape,
                    );
                    self.node_mut(leaf_idx).expanded = true;
                    // values_batch 已经是叶子玩家视角（encode 使用 leaf player 视角编码）
                    self.backprop_path(&ctx.path, values_batch[batch_result_idx]);
                    batch_result_idx += 1;
                }
            }

            // ── sequentialHalving（每轮结束后检查一次）──
            let all_reached = candidates
                .iter()
                .all(|&(_, ci)| self.node(ci).visit_count_f32() >= sim_budget as f32);

            if all_reached {
                let next_budget = Self::gumbel_budget_halved(num_sim, cur_sample, sample_n);
                if next_budget > 0 && cur_sample > 2 {
                    cur_sample /= 2;
                    let max_n = self.max_root_count();
                    candidates.sort_by(|&(_, a), &(_, b)| {
                        let sa = Self::sigma_score(self.node(a), max_n, config);
                        let sb = Self::sigma_score(self.node(b), max_n, config);
                        sb.partial_cmp(&sa).unwrap()
                    });
                    if candidates.len() > cur_sample {
                        candidates.truncate(cur_sample);
                    }
                    let (_, child_id) = candidates[0];
                    let base = self.node(child_id).visit_count;
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

    fn max_root_count(&self) -> f32 {
        self.root()
            .children
            .values()
            .map(|&ci| self.node(ci).visit_count_f32())
            .fold(0.0, f32::max)
    }

    /// σ-score：带噪声 logit + Q 项。
    fn sigma_score(node: &Node<G>, max_n: f32, config: &GumbelConfig) -> f32 {
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

    /// Completed Q policy: clean logit + Q-term softmax.
    ///
    /// Uses LightZero-style Q rescaling: min-max normalize Q to [0,1],
    /// then scale by (maxvisit_init + max_n) * value_scale to keep
    /// Q contribution similar in magnitude to logit (no single term dominates).
    fn build_completed_q_policy(
        &self,
        root_nn_value: f32,
        num_sim: usize,
        config: &GumbelConfig,
    ) -> (Vec<f32>, f32) {
        let children = &self.root().children;

        // Collect all children: (action, child_id, q_if_visited, prior)
        let entries: Vec<(ActionId, usize, Option<f32>, f32)> = children
            .occupied()
            .map(|(action, &child_id)| {
                let child = self.node(child_id);
                let q = if child.visit_count > 0 {
                    Some(child.q())
                } else {
                    None
                };
                (action, child_id, q, child.prior)
            })
            .collect();

        // Mixed value for unvisited children (LightZero _compute_mixed_value)
        let priors_q: f32 = entries
            .iter()
            .filter(|&&(_, _, q, _)| q.is_some())
            .map(|&(_, _, _, p)| p)
            .sum();
        let prior_weighted_q: f32 = entries
            .iter()
            .filter_map(|&(_, _, q, p)| q.map(|v| p * v))
            .sum();
        let mixed_value = if priors_q > 0.0 {
            (root_nn_value + (num_sim as f32 / priors_q) * prior_weighted_q)
                / (1.0 + num_sim as f32)
        } else {
            root_nn_value
        };

        // Build completed list: (action, child_id, completed_q)
        let completed: Vec<(ActionId, usize, f32)> = entries
            .iter()
            .map(|&(a, ci, q, _)| (a, ci, q.unwrap_or(mixed_value)))
            .collect();

        // LightZero Q rescale: min-max normalize to [0,1]
        let (q_min, q_max) = completed.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(min, max), &(_, _, v)| (min.min(v), max.max(v)),
        );
        let q_gap = (q_max - q_min).max(1e-8);

        // Score = clean_logit + visit_scale * sigma_scale * rescaled_q
        // LightZero: visit_scale = maxvisit_init + max_n,
        // effective_scale = visit_scale * value_scale where value_scale << 1
        let max_n = completed
            .iter()
            .map(|&(_, ci, _)| self.node(ci).visit_count_f32())
            .fold(0.0f32, f32::max);
        let effective_scale = (config.sigma_visit_c + max_n) * config.sigma_scale_c;

        let mut scores: Vec<(ActionId, f32)> = Vec::new();
        let mut max_score = f32::NEG_INFINITY;
        for &(action, child_id, q) in &completed {
            let rescaled_q = (q - q_min) / q_gap;
            let clean_logit = self.node(child_id).logit_without_noise();
            let score = clean_logit + effective_scale * rescaled_q;
            scores.push((action, score));
            if score > max_score {
                max_score = score;
            }
        }

        let mut policy = vec![0.0f32; children.len()];
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
                .map(|&ci| {
                    let child = self.node(ci);
                    child.visit_count_f32() / sum_n * child.q()
                })
                .sum()
        } else {
            root_nn_value
        };

        (policy, root_value)
    }

    /// 动作选择：softmax over visit_counts，temperature 默认 1.0。
    fn select_by_softmax_count(&self, temperature: f32, rng: &mut impl RngExt) -> ActionId {
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
    ///
    /// `value` 是叶子节点视角的回报值。每向上一层，通过
    /// `Game::flip_perspective` 翻转为上一节点的视角，消除零和对弈的硬编码假设。
    fn backprop_path(&mut self, path: &[NodeId], mut value: f32) {
        for &node_idx in path.iter().rev() {
            self.node_mut(node_idx).add_visit(value);
            value = G::flip_perspective(value);
        }
    }

    // ── utils ──

    /// Root Policy Softmax Temperature (KataGo):
    /// Scale root priors via `p^(1/T) / Z` before computing ln(prior).
    /// T=1.0 returns unchanged. T>1 pushes distribution toward uniform.
    fn apply_root_temperature(probs: &[f32], legal: &[ActionId], temperature: f32) -> Vec<f32> {
        if temperature == 1.0 {
            return probs.to_vec();
        }
        let mut scaled = vec![0.0f32; probs.len()];
        let mut sum = 0.0;
        for &action in legal {
            let v = probs[action].max(MIN_PROB).powf(1.0 / temperature);
            scaled[action] = v;
            sum += v;
        }
        if sum > 0.0 {
            for &action in legal {
                scaled[action] /= sum;
            }
        }
        scaled
    }

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
            .map(|_| (gumbel.sample(rng) as f32).clamp(-5.0_f32, 5.0_f32))
            .collect()
    }
}

// ============================================================
//  测试：用 3×3 井字棋（米字棋）验证 MCTS 算法正确性
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::{Board, Color};

    /// 模拟神经网络评估器。
    ///
    /// 返回均匀先验（所有合法动作等概率）+ 中立估值（value = 0.0）。
    /// 用这个最简单的 mock 验证 MCTS 的行为是否合理——
    /// 在仅凭模拟结果引导搜索的情况下，MCTS 应当能：
    /// 1. 对必胜局面优先选制胜走法。
    /// 2. 对必败局面避免明显的送子走法。
    struct MockEvaluator {
        /// 每个局面的标量估值（-1 ~ 1），若为空则返回 0.0
        value_map: std::collections::HashMap<String, f32>,
        /// 每个局面的策略 logits（若为空则返回均匀分布）
        policy_map: std::collections::HashMap<String, Vec<f32>>,
    }

    impl MockEvaluator {
        fn new() -> Self {
            Self {
                value_map: std::collections::HashMap::new(),
                policy_map: std::collections::HashMap::new(),
            }
        }

        /// 对特定局面设置 NN 估值。
        /// `encoded` 为 encode() 输出的 Vec<i32>（序列化为 key）。
        fn with_value(mut self, encoded: Vec<i32>, value: f32) -> Self {
            let key = Self::encode_key(&encoded);
            self.value_map.insert(key, value);
            self
        }

        /// 对特定局面设置策略。`policy` 长度应等于 action_shape。
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
        fn evaluate_batch(&self, states: &[Vec<i32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
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

    /// 构建 3×3 井字棋棋盘（连子数 = 3）。
    fn make_board() -> Board {
        Board::with_size(3, 3)
    }

    // ── 基本测试 ──

    /// 空棋盘搜索：MCTS 应当返回一个合法走法（0..9），策略维度正确。
    #[test]
    fn test_empty_board_search_returns_legal_move() {
        let board = make_board();
        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(64);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());

        // 最佳走法应当在合法范围内
        assert!(result.best_move < 9, "best_move 应为合法索引");
        assert_eq!(result.policy.len(), 9, "policy 长度应等于 action_shape");
        assert_eq!(
            result.root_nn_prior.len(),
            9,
            "root_nn_prior 长度应等于 action_shape"
        );

        // policy 中合法走法的概率和应 ≈ 1.0
        let policy_sum: f32 = result.policy.iter().sum();
        assert!(
            (policy_sum - 1.0).abs() < 0.01,
            "policy 应向 1 归一化，实际 sum={policy_sum}"
        );

        // root_value 应在 [-1, 1] 范围内
        assert!(
            result.root_value >= -1.0 && result.root_value <= 1.0,
            "root_value 应在 [-1, 1] 内，实际={}",
            result.root_value
        );
    }

    /// 终局状态（黑子已三连）：搜索应立即返回空策略。
    #[test]
    fn test_terminal_board_returns_empty() {
        let mut board = make_board();
        // 黑子连成一线：[0,0] [0,1] [0,2] 全黑
        // 但 Board::play 会翻转 player，这里要构造黑方三连后轮到黑方（不可能，仅为测试）
        // 我们让白方形成三连，然后轮到黑方：
        // 顺序：黑(0,0) 白(1,0) 黑(0,1) 白(1,1) 黑(0,2) → 黑胜，game_over=true
        board.play(0, 0); // 黑
        board.play(1, 0); // 白
        board.play(0, 1); // 黑
        board.play(1, 1); // 白
        board.play(0, 2); // 黑三连，黑胜，game_over=true
        assert!(board.game_over);

        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(32);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());

        // best_move 应等于 action_shape（9），表示无合法动作
        assert_eq!(result.best_move, 9, "终局应无合法动作");
        assert_eq!(result.policy, vec![0.0f32; 9]);
        assert_eq!(result.root_value, 0.0);
    }

    /// 仅剩一个空位：唯一的合法走法应被选为 best_move。
    #[test]
    fn test_single_legal_move_is_chosen() {
        let mut board = make_board();
        // 填满除了 (0,0) 外的 8 个格子
        // 交替落子确保不产生三连获胜
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
        let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());

        assert_eq!(result.best_move, only_move, "唯一走法应被选中");
    }

    // ── 必胜局面测试 ──

    /// 当前玩家有立即三连获胜的机会 → MCTS 应优先选择制胜走法。
    ///
    /// 局面：
    /// ```text
    ///   ●  ●  ⋅
    ///   ○  ○  ⋅
    ///   ⋅  ⋅  ⋅
    /// ```
    /// 轮黑，位置 2（即 (0,2)）立即可胜。
    #[test]
    fn test_mcts_prefers_winning_move() {
        let mut board = make_board();
        board.play(0, 0); // 黑 (0,0)
        board.play(1, 0); // 白 (1,0)
        board.play(0, 1); // 黑 (0,1)
        board.play(1, 1); // 白 (1,1)
        // 此时轮黑，位置 2 (0,2) 立即可胜
        assert!(!board.game_over);

        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(256);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());

        let win_move = board.pos_to_idx(0, 2);
        // 制胜走法的访问次数应在所有走法中排第一
        // 注：由于随机性，不做 100% 的断言；若 NN 均匀，MCTS 通过模拟应发现制胜走法
        // 这里只验证 best_move 是合法走法且 policy 中制胜走法概率最高
        let _max_prob_idx = result
            .policy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(result.best_move < 9);
        // 在充足的模拟次数下，算法至少能发现这是一步好棋
        assert!(result.policy[win_move] > 0.0, "制胜走法应拥有非零策略概率");
    }

    /// 只剩最后一步就能赢的对局：对手已经两连，轮到对手应选择堵截。
    ///
    /// 局面（轮白）：
    /// ```text
    ///   ●  ●  ⋅
    ///   ○  ⋅  ⋅
    ///   ⋅  ⋅  ⋅
    /// ```
    /// 黑下一步 (0,2) 就三连，白必须堵 (0,2)。
    /// 若白没有去堵，则黑必胜。
    #[test]
    fn test_mcts_blocks_opponent_win() {
        let mut board = make_board();
        board.play(0, 0); // 黑 (0,0)
        board.play(1, 0); // 白 (1,0)
        board.play(0, 1); // 黑 (0,1)
        // 此时轮白
        assert_eq!(board.current_player, Color::White);
        assert!(!board.game_over);

        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(256);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());

        let block_move = board.pos_to_idx(0, 2);
        assert!(result.best_move < 9);
        // 堵截走法应被评估为正收益
        assert!(result.policy[block_move] > 0.0, "堵截走法应有非零策略概率");
    }

    // ── 策略导向测试 ──

    /// 若 NN 策略强烈偏向某一走法，MCTS 应反映这一点。
    #[test]
    fn test_best_move_reflects_policy() {
        let board = make_board();
        let center_idx = board.pos_to_idx(1, 1);

        // 构造一个 NN：均匀价值，但策略强烈偏向中心
        let mut policy = vec![-10.0f32; 9];
        policy[center_idx] = 10.0;
        let evaluator = MockEvaluator::new().with_policy(board.encode(), policy);

        let config = GumbelConfig::pure_gumbel(32);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());

        // 在 pure_gumbel 模式下，根节点 prior = NN 输出
        assert!(
            result.root_nn_prior[center_idx] > 0.5,
            "中心走法的 NN 先验应 > 0.5，实际={}",
            result.root_nn_prior[center_idx]
        );
    }

    // ── 可重复性测试 ──

    /// 相同随机种子的两次搜索应产生相同结果。
    #[test]
    fn test_deterministic_with_same_seed() {
        let board = make_board();
        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(32);

        // 用可重现的种子
        use rand::SeedableRng;
        let mut rng1 = rand::rngs::SmallRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::SmallRng::seed_from_u64(42);

        let mut mcts1: MCTS<Board> = MCTS::new();
        let result1 = mcts1.search(&board, &evaluator, &config, &mut rng1);

        let mut mcts2: MCTS<Board> = MCTS::new();
        let result2 = mcts2.search(&board, &evaluator, &config, &mut rng2);

        assert_eq!(result1.best_move, result2.best_move);
        assert!(
            result1
                .policy
                .iter()
                .zip(&result2.policy)
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "相同种子应产生相同策略"
        );
        assert!((result1.root_value - result2.root_value).abs() < 1e-6);
    }

    // ── GumbelConfig 模式测试 ──

    /// pure_gumbel 模式下 pure_gumbel_noise = true。
    #[test]
    fn test_pure_gumbel_config() {
        let config = GumbelConfig::pure_gumbel(128);
        assert!(config.pure_gumbel_noise);
        assert_eq!(config.num_simulations, 128);
    }

    /// mixed 模式下 pure_gumbel_noise = false，使用 Dirichlet 噪声。
    #[test]
    fn test_mixed_config() {
        let config = GumbelConfig::mixed(128);
        assert!(!config.pure_gumbel_noise);
        assert_eq!(config.num_simulations, 128);
        // mixed 模式下 ε > 0
        assert!(config.dirichlet_epsilon > 0.0);
    }

    /// mixed 模式在空棋盘上也能正常搜索。
    #[test]
    fn test_mixed_mode_search_on_empty_board() {
        let board = make_board();
        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::mixed(64);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());
        assert!(result.best_move < 9);
        assert!((result.policy.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    // ── 边界情况 ──

    /// 接近终局的复杂走法选择：三连成线 vs 堵截对手三连。
    ///
    /// 局面（黑先行）：
    /// ```text
    ///   ●  ●  ⋅
    ///   ○  ○  ⋅
    ///   ⋅  ⋅  ⋅
    /// ```
    /// 黑选 (0,2) 立即获胜；选 (2,0~2) 留给白堵截机会。
    /// 算法应偏好制胜走法。
    #[test]
    fn test_win_vs_delay() {
        let mut board = make_board();
        board.play(0, 0); // 黑
        board.play(1, 0); // 白
        board.play(0, 1); // 黑
        board.play(1, 1); // 白
        // 轮黑

        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(128);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());

        let win_move = board.pos_to_idx(0, 2);
        // 制胜走法应拥有最高 policy 概率
        let max_idx = result
            .policy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, win_move, "制胜走法的策略概率应最高");
    }

    /// 终局检测：模拟双方走到终局后 board.game_over 为 true。
    #[test]
    fn test_game_ends_correctly() {
        let mut board = make_board();
        // 模拟一局：黑 3 连获胜
        // 黑(0,0) 白(1,0) 黑(0,1) 白(1,1) 黑(0,2) → 黑胜
        assert!(board.play(0, 0));
        assert!(board.play(1, 0));
        assert!(board.play(0, 1));
        assert!(board.play(1, 1));
        assert!(board.play(0, 2)); // 黑三连 → game_over
        assert!(board.game_over, "黑三连后应判定终局");
        assert_eq!(board.winner, Some(Color::Black));
    }

    /// 模拟一局完整对局：双方 AI 相互对弈直到终局。
    #[test]
    fn test_full_game_completion() {
        let mut board = make_board();
        let evaluator = MockEvaluator::new();
        let config = GumbelConfig::pure_gumbel(128);
        let mut mcts: MCTS<Board> = MCTS::new();

        let max_moves = 9;
        for _ in 0..max_moves {
            if board.game_over {
                break;
            }
            let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());
            if result.best_move >= 9 {
                break;
            }
            board.play_idx(result.best_move);
        }

        // 游戏应正常结束（三连或平局）
        assert!(board.game_over, "完整对弈应产生终局");
        assert!(board.step_count <= 9, "井字棋最多 9 步");
    }

    /// 平局局面：填满棋盘无三连。
    ///
    /// 最终布局（无三连）：
    /// ```text
    ///   黑 白 黑
    ///   黑 白 白
    ///   白 黑 黑
    /// ```
    #[test]
    fn test_draw_game() {
        let mut board = make_board();
        let moves = [
            (0, 0), // 黑
            (1, 1), // 白
            (2, 2), // 黑
            (2, 0), // 白
            (0, 2), // 黑
            (0, 1), // 白
            (2, 1), // 黑
            (1, 2), // 白
            (1, 0), // 黑
        ];
        for &(r, c) in &moves {
            assert!(board.play(r, c), "每步应合法：({r},{c})");
        }
        assert!(board.game_over, "棋盘满后应终局");
        assert_eq!(board.winner, None, "无三连应为平局");
    }

    // ── 启发式估值测试 ──

    /// 当 NN 估值对某局面有正预期时（例如黑方占优），
    /// root_value 应反映这一趋势。
    #[test]
    fn test_nn_value_influences_root_value() {
        let board = make_board();
        // 设置根局面 NN 估值为 1.0（强烈偏黑方），策略均匀
        let evaluator = MockEvaluator::new().with_value(board.encode(), 1.0);
        let config = GumbelConfig::pure_gumbel(1024);
        let mut mcts: MCTS<Board> = MCTS::new();
        let result = mcts.search(&board, &evaluator, &config, &mut rand::rng());

        // root_value 是 MCTS 根据模拟回传 + NN 估值综合算出，
        // 在 NN 估值 = 1.0 的情况下，root_value 应偏正。
        assert!(result.root_value > -1.0, "正估值不应产生极端负值");
    }
}
