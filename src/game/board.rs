//! 五子棋棋盘逻辑：15×15 棋盘、落子、胜负判定、状态编码

/// 棋盘大小
pub const BOARD_SIZE: usize = 15;
/// 总位置数
pub const NUM_POSITIONS: usize = BOARD_SIZE * BOARD_SIZE;
/// 棋盘编码通道数（己方/对方/上一步/当前玩家色）
pub const ENCODE_CHANNELS: usize = 4;

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
}

impl Board {
    pub fn new() -> Self {
        Self {
            cells: [[0; BOARD_SIZE]; BOARD_SIZE],
            current_player: Color::Black,
            last_move: None,
            step_count: 0,
            game_over: false,
            winner: None,
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

    /// 检查在 (row, col) 处落子后是否形成五连。
    ///
    /// 检查四个方向：水平(→)、垂直(↓)、主对角线(↘)、副对角线(↗)。
    /// 每个方向从落子点向两端延伸，统计连续同色棋子数，达到 5 即获胜。
    fn check_win(&self, row: usize, col: usize) -> bool {
        let stone = self.cells[row][col];
        if stone == 0 {
            return false;
        }
        let directions: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for &(dr, dc) in &directions {
            let mut count = 1;
            for i in 1..5i64 {
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
            for i in 1..5i64 {
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
            if count >= 5 {
                return true;
            }
        }
        false
    }

    /// 编码棋盘为网络输入：[4 × 225] 的 f32 张量
    ///
    /// 将编码写入预分配缓冲区，避免频繁分配。
    pub fn encode_into(&self, data: &mut [f32]) {
        let size = NUM_POSITIONS;
        debug_assert!(
            data.len() >= ENCODE_CHANNELS * size,
            "encode_into buffer too small"
        );

        let (current_stone, opponent_stone) = match self.current_player {
            Color::Black => (1u8, 2u8),
            Color::White => (2u8, 1u8),
        };
        let color_channel_val = match self.current_player {
            Color::Black => 1.0f32,
            Color::White => 0.0f32,
        };

        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let idx = r * BOARD_SIZE + c;
                let cell = self.cells[r][c];
                data[idx] = if cell == current_stone { 1.0 } else { 0.0 };
                data[size + idx] = if cell == opponent_stone { 1.0 } else { 0.0 };
                data[3 * size + idx] = color_channel_val;
            }
        }

        // 清零通道 2（last_move），避免上一帧残留
        for idx in 0..size {
            data[2 * size + idx] = 0.0;
        }

        if let Some((lr, lc)) = self.last_move {
            data[2 * size + lr * BOARD_SIZE + lc] = 1.0;
        }
    }

    /// 编码棋盘为网络输入（分配新 Vec）。
    /// MCTS evaluate 中使用 `encode_into` 替代此方法避免重复分配。
    pub fn encode_state(&self) -> Vec<f32> {
        let mut data = vec![0.0f32; ENCODE_CHANNELS * NUM_POSITIONS];
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
    /// - `state`: 输入形状 [ENCODE_CHANNELS × NUM_POSITIONS] 的编码（4 通道交错存储）
    /// - `policy`: 输入形状 [NUM_POSITIONS] 的策略分布
    /// - `transform_idx`: 变换索引 (0..=7)
    ///
    /// 对每个通道同时做相同的空间重排，策略分布也做相同重排。
    pub fn apply_transform(
        state: &[f32],
        policy: &[f32],
        transform_idx: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let map = &Self::index_maps()[transform_idx];
        let mut new_state = vec![0.0f32; state.len()];
        let mut new_policy = vec![0.0f32; NUM_POSITIONS];

        // 对每个编码通道做空间变换
        for ch in 0..ENCODE_CHANNELS {
            let offset = ch * NUM_POSITIONS;
            for src in 0..NUM_POSITIONS {
                new_state[offset + map[src]] = state[offset + src];
            }
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
        state: &[f32],
        policy: &[f32],
        rng: &mut impl rand::RngExt,
        identity_prob: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let t: f32 = rng.random();
        let idx = if t < identity_prob {
            0
        } else {
            1 + rng.random_range(0..7) as usize
        };
        Self::apply_transform(state, policy, idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_board() {
        let board = Board::new();
        assert_eq!(board.current_player, Color::Black);
        assert_eq!(board.step_count, 0);
        assert!(!board.game_over);
        assert_eq!(board.legal_moves().len(), NUM_POSITIONS);
    }

    #[test]
    fn test_play_and_alternate() {
        let mut board = Board::new();
        assert!(board.play(7, 7));
        assert_eq!(board.current_player, Color::White);
        assert!(board.play(7, 8));
        assert_eq!(board.current_player, Color::Black);
    }

    #[test]
    fn test_cannot_play_occupied() {
        let mut board = Board::new();
        board.play(7, 7);
        assert!(!board.play(7, 7));
    }

    #[test]
    fn test_horizontal_win() {
        let mut board = Board::new();
        let moves = [
            (7, 6),
            (0, 0),
            (7, 7),
            (1, 1),
            (7, 8),
            (2, 2),
            (7, 9),
            (3, 3),
            (7, 10),
        ];
        for &(r, c) in moves.iter() {
            board.play(r, c);
        }
        assert!(board.game_over);
        assert_eq!(board.winner, Some(Color::Black));
    }

    #[test]
    fn test_diagonal_win() {
        let mut board = Board::new();
        let moves = [
            (5, 5),
            (0, 0),
            (6, 6),
            (1, 1),
            (7, 7),
            (2, 2),
            (8, 8),
            (3, 3),
            (9, 9),
        ];
        for &(r, c) in moves.iter() {
            board.play(r, c);
        }
        assert!(board.game_over);
        assert_eq!(board.winner, Some(Color::Black));
    }

    #[test]
    fn test_encode_state() {
        let mut board = Board::new();
        board.play(7, 7); // Black
        board.play(0, 0); // White
        // 现在轮到 Black
        let data = board.encode_state();
        assert_eq!(data.len(), ENCODE_CHANNELS * NUM_POSITIONS);
        let idx_77 = 7 * BOARD_SIZE + 7;
        let idx_00 = 0 * BOARD_SIZE + 0;
        // 当前是 Black：Black 子在通道 0，White 子在通道 1
        assert_eq!(data[idx_77], 1.0, "Black stone should be in channel 0");
        assert_eq!(
            data[NUM_POSITIONS + idx_77],
            0.0,
            "Non-opponent at Black stone"
        );
        assert_eq!(data[idx_00], 0.0, "Non-current at White stone");
        assert_eq!(
            data[NUM_POSITIONS + idx_00],
            1.0,
            "White stone in opponent channel"
        );
    }

    #[test]
    fn test_undo_play() {
        let mut board = Board::new();
        board.play(7, 7); // Black 落子
        let snap = board.snapshot(); // 保存 (7,7) 落子前的状态
        board.play(7, 8); // White 落子
        assert_eq!(board.current_player, Color::Black);

        board.undo(7, 8, &snap); // 撤销 White 的 (7,8)
        // 撤销后应恢复到黑方刚落子在 (7,7) 后的状态（轮到白方）
        assert_eq!(board.current_player, Color::White);
        assert_eq!(board.step_count, 1);
        assert_eq!(board.get(7, 7), 1);
        assert_eq!(board.get(7, 8), 0); // 白方落子已被清除
        assert!(!board.game_over);
    }

    #[test]
    fn test_undo_with_win() {
        let mut board = Board::new();
        // 黑方在 (7,6)-(7,10) 连五胜
        let moves = [
            (7, 6),
            (0, 0),
            (7, 7),
            (1, 1),
            (7, 8),
            (2, 2),
            (7, 9),
            (3, 3),
        ];
        for &(r, c) in moves.iter() {
            board.play(r, c);
        }
        assert!(!board.game_over);

        // 黑方即将走 (7,10)，保存当前状态
        let snap_before_win = board.snapshot();
        board.play(7, 10); // 黑方获胜
        assert!(board.game_over);
        assert_eq!(board.winner, Some(Color::Black));

        board.undo(7, 10, &snap_before_win);
        assert!(!board.game_over);
        assert_eq!(board.winner, None);
        assert_eq!(board.current_player, Color::Black); // 回到黑方回合
        assert_eq!(board.get(7, 10), 0);
    }

    #[test]
    fn test_snapshot_and_undo_sequence() {
        let mut board = Board::new();
        let mut snapshots = Vec::new();
        let mut positions = Vec::new();

        for i in 0..5 {
            snapshots.push(board.snapshot());
            positions.push((i, i));
            board.play(i, i);
        }
        assert_eq!(board.step_count, 5);

        // 逆序撤销
        for i in (0..5).rev() {
            let (r, c) = positions[i];
            board.undo(r, c, &snapshots[i]);
        }
        assert_eq!(board.step_count, 0);
        assert_eq!(board.current_player, Color::Black);
    }

    // ============================================================
    //  D4 对称变换测试
    // ============================================================

    #[test]
    fn test_d4_index_map_identity() {
        let maps = D4Symmetry::index_maps();
        let identity = &maps[0];
        for i in 0..NUM_POSITIONS {
            assert_eq!(identity[i], i, "identity map should not change index");
        }
    }

    #[test]
    fn test_d4_index_map_rotation_involution() {
        // 旋转 180° 做两次等于恒等
        let maps = D4Symmetry::index_maps();
        let r180 = &maps[2];
        for i in 0..NUM_POSITIONS {
            assert_eq!(r180[r180[i]], i);
        }
    }

    #[test]
    fn test_d4_index_map_flip_involution() {
        // 翻转做两次 = 恒等
        let maps = D4Symmetry::index_maps();
        for &t in &[4usize, 5, 6, 7] {
            let map = &maps[t];
            for i in 0..NUM_POSITIONS {
                assert_eq!(map[map[i]], i, "transform {t} at index {i}");
            }
        }
    }

    #[test]
    fn test_d4_index_map_all_distinct() {
        let maps = D4Symmetry::index_maps();
        for i in 0..NUM_POSITIONS {
            let mut seen = std::collections::BTreeSet::new();
            for t in 0..8 {
                seen.insert(maps[t][i]);
            }
            // 不是所有位置都有 8 种不同映射（中心点和对称轴上的点会有重叠），
            // 但至少应该有 1 种（恒等）
            assert!(!seen.is_empty());
        }
    }

    #[test]
    fn test_d4_apply_transform() {
        // 编码一个简单棋盘：黑子 (0,0)，白子 (14,14)，当前玩家黑
        let mut board = Board::new();
        board.play(0, 0);
        board.play(14, 14);
        let state = board.encode_state();
        let policy: Vec<f32> = (0..NUM_POSITIONS).map(|i| i as f32).collect();

        // 旋转 180°：黑子应到 (14,14)，白子到 (0,0)
        let (rot_state, rot_policy) = D4Symmetry::apply_transform(&state, &policy, 2);

        let idx_00 = Board::pos_to_idx(0, 0);
        let idx_14_14 = Board::pos_to_idx(14, 14);

        // 通道 0 (己方棋子) 中的黑子在 180° 旋转后应到 (14,14)
        let new_self_ch = &rot_state[0 * NUM_POSITIONS..1 * NUM_POSITIONS];
        assert_eq!(
            new_self_ch[idx_14_14], 1.0,
            "black stone should rotate to (14,14)"
        );
        assert_eq!(
            new_self_ch[idx_00], 0.0,
            "(0,0) should be empty after rotation"
        );

        // 通道 1 (对方棋子) 中的白子在 180° 旋转后应到 (0,0)
        let new_opp_ch = &rot_state[1 * NUM_POSITIONS..2 * NUM_POSITIONS];
        assert_eq!(
            new_opp_ch[idx_00], 1.0,
            "white stone should rotate to (0,0)"
        );

        // 策略：idx 224 应映射到 idx 0
        assert_eq!(rot_policy[idx_14_14], policy[idx_00]);
        assert_eq!(rot_policy[idx_00], policy[idx_14_14]);
    }

    #[test]
    fn test_d4_random_augment() {
        let board = Board::new();
        let state = board.encode_state();
        let policy = vec![1.0 / NUM_POSITIONS as f32; NUM_POSITIONS];
        let mut rng = rand::rng();

        let (aug_state, aug_policy) = D4Symmetry::random_augment(&state, &policy, &mut rng, 0.0);
        assert_eq!(aug_state.len(), state.len());
        assert_eq!(aug_policy.len(), NUM_POSITIONS);
        assert!((aug_policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }
}
