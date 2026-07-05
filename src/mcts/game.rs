//! 游戏环境抽象 trait。
//!
//! MCTS 通过此 trait 与具体游戏解耦，不依赖棋类对弈的特定概念。
//! `ActionId = usize` 为平铺动作索引，消除了所有 `pos_to_idx`
//! 调用和行列坐标来回转换。
//!
//! ## 设计原则
//!
//! - **无胜负/对手概念**：胜负被泛化为"当前玩家视角的回报值"（`terminal_value`）。
//! - **视角翻转可配置**：通过 `flip_perspective` 控制值在玩家之间的传递方式。
//!   零和对弈默认取反（`-value`），单智能体可覆盖为恒等。
//! - **下一玩家可配置**：`next_player` 对弈返回对手，单智能体返回自己。
//!
//! ## 实现新游戏
//!
//! 只需为本 trait 实现一个适配器即可将 MCTS / Gumbel Zero 复用到任何游戏：
//!
//! ```ignore
//! impl Game for MyGame {
//!     type Player = MyPlayer;
//!     fn encode_shape(&self) -> &[usize] { &[100] }
//!     fn action_shape(&self) -> usize { 100 }
//!     fn current_player(&self) -> MyPlayer { ... }
//!     fn is_terminal(&self) -> bool { ... }
//!     fn terminal_value(&self) -> f32 { ... }
//!     fn legal_actions(&self) -> Vec<ActionId> { ... }
//!     fn play(&mut self, action: ActionId) -> bool { ... }
//!     fn encode(&self) -> Vec<i32> { ... }
//!     fn next_player(current: MyPlayer) -> MyPlayer { ... }
//! }
//! ```

/// 动作标识符（平铺索引）。
pub type ActionId = usize;

/// 游戏环境抽象。MCTS 内部通过此 trait 操作游戏状态，不依赖具体游戏类型。
pub trait Game: Clone + Send {
    /// 玩家类型（如 Black / White，或单智能体可设为 `()`）。
    type Player: Copy + PartialEq + std::fmt::Debug;

    // ── 游戏元信息 ──

    /// 神经网络输入编码的形状。MCTS 用 `iter().product()` 得出缓冲区长度。
    fn encode_shape(&self) -> &[usize];
    /// 动作空间大小（所有可能动作的总数）。
    fn action_shape(&self) -> usize;

    // ── 游戏状态查询 ──

    /// 当前轮到哪个玩家。
    fn current_player(&self) -> Self::Player;

    /// 终局时**从当前玩家视角**的回报值。
    ///
    /// 应在 `is_terminal() == true` 时调用，返回当前玩家在这一局中的回报：
    /// - 零和对弈：赢了 → 正，输了 → 负，平局 → 0
    /// - 单智能体：达成目标 → 正，失败 → 负，否则 → 自定义
    fn terminal_value(&self) -> Option<f32>;

    fn is_terminal(&self) -> bool {
        self.terminal_value().is_some()
    }

    // ── 动作 ──

    /// 返回所有合法动作（平铺索引）。
    fn legal_actions(&self) -> Vec<ActionId>;
    /// 执行一个动作（平铺索引），返回是否成功。
    fn play(&mut self, action: ActionId) -> bool;

    // ── NN 编码 ──

    /// 将游戏状态编码为神经网络输入（当前玩家视角），返回编码向量。
    fn encode(&self) -> Vec<i32>;

    // ── 玩家切换 ──

    /// 落子后轮到哪个玩家。
    ///
    /// 对弈游戏返回对手；单智能体游戏可返回 `current` 自身。
    fn next_player(current: Self::Player) -> Self::Player;

    /// 视角翻转：将 value 从子节点视角转换为父节点视角。
    ///
    /// 零和对弈默认取反（`-value`），因为子节点有利 = 父节点不利。
    /// 单智能体游戏可覆盖为恒等（`value`）或带折扣因子。
    fn flip_perspective(value: f32) -> f32 {
        -value
    }
}
