//! N子棋棋盘逻辑：可配置大小的棋盘、落子、胜负判定、状态编码

/// 棋盘大小
pub const BOARD_SIZE: usize = 6;
/// 总位置数
pub const NUM_POSITIONS: usize = BOARD_SIZE * BOARD_SIZE;
/// 棋盘编码长度（扁平序列：0=空, 1=己方, 2=对方）
pub const ENCODE_LEN: usize = NUM_POSITIONS;
/// 默认获胜所需连子数（五子棋）
pub const DEFAULT_WIN_LENGTH: usize = 4;

/// 玩家颜色
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
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
    cells: [[u8; BOARD_SIZE]; BOARD_SIZE],
    pub current_player: Color,
    pub last_move: Option<(usize, usize)>,
    pub step_count: usize,
    pub game_over: bool,
    pub winner: Option<Color>,
    /// 获胜所需连续棋子数（N子棋的 N）
    pub win_length: usize,
}

impl Board {
    /// 创建默认棋盘（N子棋，默认 5 连获胜）。
    /// 棋盘大小由 `BOARD_SIZE` 常量决定。
    pub fn new() -> Self {
        Self::with_win_length(DEFAULT_WIN_LENGTH)
    }

    /// 创建自定义 N 子棋棋盘。
    ///
    /// `n` 为获胜所需连续棋子数，必须满足 2 ≤ n ≤ BOARD_SIZE。
    pub fn with_win_length(n: usize) -> Self {
        assert!(n >= 2, "win_length must be at least 2");
        assert!(
            n <= BOARD_SIZE,
            "win_length cannot exceed BOARD_SIZE ({BOARD_SIZE})"
        );
        Self {
            cells: [[0; BOARD_SIZE]; BOARD_SIZE],
            current_player: Color::Black,
            last_move: None,
            step_count: 0,
            game_over: false,
            winner: None,
            win_length: n,
        }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> u8 {
        self.cells[row][col]
    }

    #[inline]
    pub fn is_empty(&self, row: usize, col: usize) -> bool {
        self.cells[row][col] == 0
    }

    pub fn legal_moves(&self) -> Vec<(usize, usize)> {
        let mut moves = Vec::with_capacity(NUM_POSITIONS);
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
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                if self.cells[r][c] == 0 {
                    buffer.push((r, c));
                }
            }
        }
    }

    #[inline]
    pub fn pos_to_idx(row: usize, col: usize) -> usize {
        row * BOARD_SIZE + col
    }

    #[inline]
    pub fn idx_to_pos(idx: usize) -> (usize, usize) {
        (idx / BOARD_SIZE, idx % BOARD_SIZE)
    }

    /// 在棋盘上落子，返回是否成功。
    ///
    /// 落子后自动执行：
    /// 1. 更新棋盘状态（标记棋子）
    /// 2. 检查是否五连获胜
    /// 3. 检查是否棋盘已满（平局）
    /// 4. 翻转当前玩家
    pub fn play(&mut self, row: usize, col: usize) -> bool {
        if self.game_over || row >= BOARD_SIZE || col >= BOARD_SIZE {
            return false;
        }
        if self.cells[row][col] != 0 {
            return false;
        }

        let stone = match self.current_player {
            Color::Black => 1,
            Color::White => 2,
        };
        self.cells[row][col] = stone;
        self.last_move = Some((row, col));
        self.step_count += 1;

        if self.check_win(row, col) {
            self.game_over = true;
            self.winner = Some(self.current_player);
        } else if self.step_count >= NUM_POSITIONS {
            self.game_over = true;
            self.winner = None;
        } else {
            self.current_player = self.current_player.opponent();
        }
        true
    }

    #[inline]
    pub fn play_idx(&mut self, idx: usize) -> bool {
        let (r, c) = Self::idx_to_pos(idx);
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
        self.cells[move_row][move_col] = 0;
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
        let stone = self.cells[row][col];
        if stone == 0 {
            return false;
        }
        let n = self.win_length;
        // 最多只需检查 n-1 步远
        let max_step = (n - 1) as i64;
        let directions: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for &(dr, dc) in &directions {
            let mut count = 1;
            for i in 1..=max_step {
                let nr = row as i64 + dr as i64 * i;
                let nc = col as i64 + dc as i64 * i;
                if nr < 0 || nr >= BOARD_SIZE as i64 || nc < 0 || nc >= BOARD_SIZE as i64 {
                    break;
                }
                if self.cells[nr as usize][nc as usize] == stone {
                    count += 1;
                } else {
                    break;
                }
            }
            for i in 1..=max_step {
                let nr = row as i64 - dr as i64 * i;
                let nc = col as i64 - dc as i64 * i;
                if nr < 0 || nr >= BOARD_SIZE as i64 || nc < 0 || nc >= BOARD_SIZE as i64 {
                    break;
                }
                if self.cells[nr as usize][nc as usize] == stone {
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
        debug_assert!(data.len() >= ENCODE_LEN, "encode_into buffer too small");

        let (me, opp) = match self.current_player {
            Color::Black => (1u8, 2u8),
            Color::White => (2u8, 1u8),
        };

        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let idx = r * BOARD_SIZE + c;
                let cell = self.cells[r][c];
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
        let mut data = vec![0i32; ENCODE_LEN];
        self.encode_into(&mut data);
        data
    }
}

// ============================================================
//  D4 二面体群：棋盘对称变换（用于训练数据增强）
// ============================================================

/// D4 群 8 种对称变换的索引映射表。
///
/// 五子棋在以下变换下保持不变性：
/// - 恒等 (0°)
/// - 旋转 90°、180°、270°
/// - 水平翻转
/// - 垂直翻转
/// - 主对角线翻转
/// - 副对角线翻转
///
/// `D4_INDEX_MAPS[t][src_idx]` = 变换 t 下位置 src_idx 映射到的目标索引。
pub struct D4Symmetry;

impl D4Symmetry {
    /// 变换种类数
    pub const COUNT: usize = 8;

    /// 获取预计算的索引映射表（延迟初始化）。
    pub fn index_maps() -> &'static [[usize; NUM_POSITIONS]; 8] {
        use std::sync::OnceLock;
        static MAPS: OnceLock<[[usize; NUM_POSITIONS]; 8]> = OnceLock::new();
        MAPS.get_or_init(|| {
            let mut maps = [[0usize; NUM_POSITIONS]; 8];
            for r in 0..BOARD_SIZE {
                for c in 0..BOARD_SIZE {
                    let src = Board::pos_to_idx(r, c);
                    let last = BOARD_SIZE - 1;
                    // 恒等
                    maps[0][src] = Board::pos_to_idx(r, c);
                    // 旋转 90°
                    maps[1][src] = Board::pos_to_idx(c, last - r);
                    // 旋转 180°
                    maps[2][src] = Board::pos_to_idx(last - r, last - c);
                    // 旋转 270°
                    maps[3][src] = Board::pos_to_idx(last - c, r);
                    // 水平翻转
                    maps[4][src] = Board::pos_to_idx(r, last - c);
                    // 垂直翻转
                    maps[5][src] = Board::pos_to_idx(last - r, c);
                    // 主对角线翻转
                    maps[6][src] = Board::pos_to_idx(c, r);
                    // 副对角线翻转
                    maps[7][src] = Board::pos_to_idx(last - c, last - r);
                }
            }
            maps
        })
    }

    /// 对编码后的棋盘状态和策略分布应用指定的 D4 变换。
    ///
    /// - `state`: 输入形状 [NUM_POSITIONS] 的 i32 编码（0=空, 1=黑, 2=白，扁平序列）
    /// - `policy`: 输入形状 [NUM_POSITIONS] 的策略分布
    /// - `transform_idx`: 变换索引 (0..=7)
    ///
    /// 对状态做空间重排，策略分布也做相同重排。
    pub fn apply_transform(
        state: &[i32],
        policy: &[f32],
        transform_idx: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        let map = &Self::index_maps()[transform_idx];
        let mut new_state = vec![0i32; state.len()];
        let mut new_policy = vec![0.0f32; NUM_POSITIONS];

        for src in 0..NUM_POSITIONS {
            new_state[map[src]] = state[src];
        }

        // 策略分布重排
        for src in 0..NUM_POSITIONS {
            new_policy[map[src]] = policy[src];
        }

        (new_state, new_policy)
    }

    /// 随机选取一种 D4 变换并应用到状态和策略上。
    ///
    /// 恒等变换（idx=0）的概率为 `identity_prob`，其余 7 种均匀分配。
    pub fn random_augment(
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
        Self::apply_transform(state, policy, idx)
    }
}
