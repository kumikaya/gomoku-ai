//! N子棋棋盘逻辑：可配置大小的棋盘、落子、胜负判定、状态编码

use ndarray::Array2;

use crate::mcts::game::{ActionId, Game};

/// 玩家颜色
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Color {
    #[default]
    Black = 0,
    White = 1,
}

impl Color {
    #[inline]
    pub fn opponent(self) -> Self {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

/// 棋盘快照，用于撤销走法
#[derive(Clone)]
pub struct BoardSnapshot {
    pub prev_last_move: Option<(usize, usize)>,
    pub prev_player: Color,
    pub prev_game_over: bool,
    pub prev_winner: Option<Color>,
}

/// 棋盘状态
#[derive(Clone)]
pub struct Board {
    /// 棋盘边长（N×N）
    pub board_size: usize,
    /// 棋盘格子（0=空, 1=黑, 2=白）
    cells: Array2<u8>,
    pub current_player: Color,
    pub last_move: Option<(usize, usize)>,
    pub step_count: usize,
    pub game_over: bool,
    pub winner: Option<Color>,
    /// 获胜所需连续棋子数（N子棋的 N）
    pub win_length: usize,
    /// 用于 `encode_shape` 返回值的存储
    encode_shape_storage: [usize; 1],
}

impl Board {
    /// 默认棋盘边长
    pub const DEFAULT_BOARD_SIZE: usize = 6;
    /// 默认获胜所需连子数
    pub const DEFAULT_WIN_LENGTH: usize = 4;

    /// 创建默认棋盘（使用默认边长和默认获胜连子数）。
    pub fn new() -> Self {
        Self::with_size(Self::DEFAULT_BOARD_SIZE, Self::DEFAULT_WIN_LENGTH)
    }

    /// 创建自定义 N 子棋棋盘。
    ///
    /// `n` 为获胜所需连续棋子数，必须满足 2 ≤ n ≤ board_size。
    pub fn with_win_length(n: usize) -> Self {
        Self::with_size(Self::DEFAULT_BOARD_SIZE, n)
    }

    /// 创建指定大小的棋盘。
    ///
    /// `board_size` 为棋盘边长，`win_length` 为获胜所需连续棋子数。
    pub fn with_size(board_size: usize, win_length: usize) -> Self {
        assert!(win_length >= 2, "win_length must be at least 2");
        assert!(
            win_length <= board_size,
            "win_length cannot exceed board_size ({board_size})"
        );
        Self {
            board_size,
            cells: Array2::zeros((board_size, board_size)),
            current_player: Color::Black,
            last_move: None,
            step_count: 0,
            game_over: false,
            winner: None,
            win_length,
            encode_shape_storage: [board_size * board_size],
        }
    }

    /// 棋盘总位置数（board_size²）。
    #[inline]
    pub fn num_positions(&self) -> usize {
        self.board_size * self.board_size
    }

    /// 棋盘编码长度，等同于 `num_positions()`。
    #[inline]
    pub fn encode_len(&self) -> usize {
        self.num_positions()
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> u8 {
        self.cells[[row, col]]
    }

    #[inline]
    pub fn is_empty(&self, row: usize, col: usize) -> bool {
        self.cells[[row, col]] == 0
    }

    pub fn legal_moves(&self) -> Vec<(usize, usize)> {
        let mut moves = Vec::with_capacity(self.num_positions());
        self.fill_legal_moves(&mut moves);
        moves
    }

    /// 将合法走法写入预分配的缓冲区，避免重复分配。
    ///
    /// MCTS 模拟中使用此方法以减少每步的 Vec 分配。
    pub fn fill_legal_moves(&self, buffer: &mut Vec<(usize, usize)>) {
        buffer.clear();
        if self.game_over {
            return;
        }
        for r in 0..self.board_size {
            for c in 0..self.board_size {
                if self.cells[[r, c]] == 0 {
                    buffer.push((r, c));
                }
            }
        }
    }

    #[inline]
    pub fn pos_to_idx(&self, row: usize, col: usize) -> usize {
        row * self.board_size + col
    }

    #[inline]
    pub fn idx_to_pos(&self, idx: usize) -> (usize, usize) {
        (idx / self.board_size, idx % self.board_size)
    }

    /// 在棋盘上落子，返回是否成功。
    ///
    /// 落子后自动执行：
    /// 1. 更新棋盘状态（标记棋子）
    /// 2. 检查是否 N 连获胜
    /// 3. 检查是否棋盘已满（平局）
    /// 4. 翻转当前玩家
    pub fn play(&mut self, row: usize, col: usize) -> bool {
        if self.game_over || row >= self.board_size || col >= self.board_size {
            return false;
        }
        if self.cells[[row, col]] != 0 {
            return false;
        }

        let stone = match self.current_player {
            Color::Black => 1,
            Color::White => 2,
        };
        self.cells[[row, col]] = stone;
        self.last_move = Some((row, col));
        self.step_count += 1;

        if self.check_win(row, col) {
            self.game_over = true;
            self.winner = Some(self.current_player);
        } else if self.step_count >= self.num_positions() {
            self.game_over = true;
            self.winner = None;
        }
        self.current_player = self.current_player.opponent();
        true
    }

    #[inline]
    pub fn play_idx(&mut self, idx: usize) -> bool {
        let (r, c) = self.idx_to_pos(idx);
        self.play(r, c)
    }

    /// 撤销上一步走法，恢复棋盘到落子前的状态。
    ///
    /// `move_row` / `move_col` 是要撤销的那步落子位置。
    /// `snapshot` 是落子前通过 `snapshot()` 保存的状态。
    ///
    /// MCTS 模拟中替代 `board.clone()`：模拟在单一棋盘上操作，
    /// 回溯时通过 undo 恢复，将 O(N²) 的 clone 降为 O(1)。
    pub fn undo(&mut self, move_row: usize, move_col: usize, snapshot: &BoardSnapshot) {
        self.cells[[move_row, move_col]] = 0;
        self.last_move = snapshot.prev_last_move;
        self.current_player = snapshot.prev_player;
        self.step_count -= 1;
        self.game_over = snapshot.prev_game_over;
        self.winner = snapshot.prev_winner;
    }

    /// 创建当前状态的快照，用于后续 undo 恢复。
    pub fn snapshot(&self) -> BoardSnapshot {
        BoardSnapshot {
            prev_last_move: self.last_move,
            prev_player: self.current_player,
            prev_game_over: self.game_over,
            prev_winner: self.winner,
        }
    }

    /// 检查在 (row, col) 处落子后是否形成 N 连。
    ///
    /// N 由 `self.win_length` 决定。
    /// 检查四个方向：水平(→)、垂直(↓)、主对角线(↘)、副对角线(↗)。
    /// 每个方向从落子点向两端延伸，统计连续同色棋子数，达到 N 即获胜。
    fn check_win(&self, row: usize, col: usize) -> bool {
        let stone = self.cells[[row, col]];
        if stone == 0 {
            return false;
        }
        let n = self.win_length;
        let bs = self.board_size as i64;
        // 最多只需检查 n-1 步远
        let max_step = (n - 1) as i64;
        let directions: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for &(dr, dc) in &directions {
            let mut count = 1;
            for i in 1..=max_step {
                let nr = row as i64 + dr as i64 * i;
                let nc = col as i64 + dc as i64 * i;
                if nr < 0 || nr >= bs || nc < 0 || nc >= bs {
                    break;
                }
                if self.cells[[nr as usize, nc as usize]] == stone {
                    count += 1;
                } else {
                    break;
                }
            }
            for i in 1..=max_step {
                let nr = row as i64 - dr as i64 * i;
                let nc = col as i64 - dc as i64 * i;
                if nr < 0 || nr >= bs || nc < 0 || nc >= bs {
                    break;
                }
                if self.cells[[nr as usize, nc as usize]] == stone {
                    count += 1;
                } else {
                    break;
                }
            }
            if count >= n {
                return true;
            }
        }
        false
    }

    /// 编码棋盘为网络输入：扁平 i32 序列，当前玩家视角。
    ///
    /// 0=空, 1=己方棋子, 2=对方棋子。
    pub fn encode_into(&self, data: &mut [i32]) {
        let n = self.encode_len();
        debug_assert!(data.len() >= n, "encode_into buffer too small");

        let (me, opp) = match self.current_player {
            Color::Black => (1u8, 2u8),
            Color::White => (2u8, 1u8),
        };

        for r in 0..self.board_size {
            for c in 0..self.board_size {
                let idx = r * self.board_size + c;
                let cell = self.cells[[r, c]];
                data[idx] = if cell == me {
                    1
                } else if cell == opp {
                    2
                } else {
                    0
                };
            }
        }
    }

    /// 编码棋盘为网络输入（分配新 Vec）。
    pub fn encode_state(&self) -> Vec<i32> {
        let mut data = vec![0i32; self.encode_len()];
        self.encode_into(&mut data);
        data
    }
}

// ============================================================
//  Game trait 实现 — 将 Board 适配到 MCTS 泛型接口
// ============================================================

impl Game for Board {
    type Player = Color;

    fn encode_shape(&self) -> &[usize] {
        &self.encode_shape_storage
    }

    fn action_shape(&self) -> usize {
        self.num_positions()
    }

    fn current_player(&self) -> Color {
        self.current_player
    }

    fn terminal_value(&self) -> Option<f32> {
        if !self.game_over {
            return None;
        }
        let value = match self.winner {
            Some(w) if w == self.current_player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
        Some(value)
    }

    fn legal_actions(&self) -> Vec<ActionId> {
        let mut moves = Vec::with_capacity(self.num_positions());
        if self.game_over {
            return moves;
        }
        for r in 0..self.board_size {
            for c in 0..self.board_size {
                if self.cells[[r, c]] == 0 {
                    moves.push(self.pos_to_idx(r, c));
                }
            }
        }
        moves
    }

    fn play(&mut self, action: ActionId) -> bool {
        self.play_idx(action)
    }

    fn encode(&self) -> Vec<i32> {
        self.encode_state()
    }

    fn next_player(current: Color) -> Color {
        current.opponent()
    }
}

// ============================================================
//  D4 二面体群：棋盘对称变换（用于训练数据增强）
// ============================================================

/// D4 群 8 种对称变换。
///
/// 五子棋在以下变换下保持不变性：
/// - 恒等 (0°)
/// - 旋转 90°、180°、270°
/// - 水平翻转
/// - 垂直翻转
/// - 主对角线翻转
/// - 副对角线翻转
///
/// 构造时预计算所有索引映射表，后续调用零分配。
pub struct D4Symmetry {
    /// 棋盘边长
    board_size: usize,
    /// 预计算的 8 种变换索引映射，每种变换存储 num_positions 个目标索引
    maps: Vec<Vec<usize>>,
}

impl D4Symmetry {
    /// 变换种类数
    pub const COUNT: usize = 8;

    /// 为指定大小的棋盘创建 D4 对称变换表。
    pub fn new(board_size: usize) -> Self {
        let npos = board_size * board_size;
        let last = board_size - 1;
        let mut maps = vec![vec![0usize; npos]; Self::COUNT];

        for r in 0..board_size {
            for c in 0..board_size {
                let src = r * board_size + c;
                // 恒等
                maps[0][src] = r * board_size + c;
                // 旋转 90°
                maps[1][src] = c * board_size + (last - r);
                // 旋转 180°
                maps[2][src] = (last - r) * board_size + (last - c);
                // 旋转 270°
                maps[3][src] = (last - c) * board_size + r;
                // 水平翻转
                maps[4][src] = r * board_size + (last - c);
                // 垂直翻转
                maps[5][src] = (last - r) * board_size + c;
                // 主对角线翻转
                maps[6][src] = c * board_size + r;
                // 副对角线翻转
                maps[7][src] = (last - c) * board_size + (last - r);
            }
        }

        Self { board_size, maps }
    }

    /// 对编码后的棋盘状态和策略分布应用指定的 D4 变换。
    ///
    /// - `state`: 输入形状 [NUM_POSITIONS] 的 i32 编码（扁平序列）
    /// - `policy`: 输入形状 [NUM_POSITIONS] 的策略分布
    /// - `transform_idx`: 变换索引 (0..=7)
    ///
    /// 对状态做空间重排，策略分布也做相同重排。
    pub fn apply_transform(
        &self,
        state: &[i32],
        policy: &[f32],
        transform_idx: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        let npos = self.board_size * self.board_size;
        let map = &self.maps[transform_idx];
        let mut new_state = vec![0i32; state.len()];
        let mut new_policy = vec![0.0f32; npos];

        for src in 0..npos {
            new_state[map[src]] = state[src];
        }

        // 策略分布重排
        for src in 0..npos {
            new_policy[map[src]] = policy[src];
        }

        (new_state, new_policy)
    }

    /// 随机选取一种 D4 变换并应用到状态和策略上。
    ///
    /// 恒等变换（idx=0）的概率为 `identity_prob`，其余 7 种均匀分配。
    pub fn random_augment(
        &self,
        state: &[i32],
        policy: &[f32],
        rng: &mut impl rand::RngExt,
        identity_prob: f32,
    ) -> (Vec<i32>, Vec<f32>) {
        let t: f32 = rng.random();
        let idx = if t < identity_prob {
            0
        } else {
            1 + rng.random_range(0..7) as usize
        };
        self.apply_transform(state, policy, idx)
    }
}
