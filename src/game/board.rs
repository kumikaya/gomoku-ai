//! 五子棋棋盘逻辑：15×15 棋盘、落子、胜负判定、状态编码

/// 棋盘大小
pub const BOARD_SIZE: usize = 15;
/// 总位置数
pub const NUM_POSITIONS: usize = BOARD_SIZE * BOARD_SIZE;

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
    /// ## 四个通道的设计
    ///
    /// | 通道 | 含义 | 编码规则 |
    /// |------|------|---------|
    /// | 0 | 当前玩家棋子 | 当前玩家的棋子为 1.0，其余为 0.0 |
    /// | 1 | 对手棋子 | 对手的棋子为 1.0，其余为 0.0 |
    /// | 2 | 最后一步 | 上一步落子位置为 1.0，其余为 0.0 |
    /// | 3 | 当前颜色 | 黑方为 1.0，白方为 0.0（全通道常量）|
    ///
    /// 通道 0 和 1 让网络知道双方棋子分布；
    /// 通道 2 提供最近一步的位置信息（有助于判断对手意图）；
    /// 通道 3 提供当前回合颜色（网络内部没有颜色概念，需要显式告知）。
    ///
    /// 注意：通道编码随当前玩家而变化。例如黑方视角下，黑子出现在通道 0，
    /// 白子出现在通道 1；轮到白方时翻转。这确保网络从任意玩家视角看到的
    /// 输入模式一致，利于学习对称性。
    ///
    /// 通道 0: 当前玩家棋子 | 通道 1: 对手棋子 | 通道 2: 最后一步 | 通道 3: 颜色
    pub fn encode_state(&self) -> Vec<f32> {
        let size = NUM_POSITIONS;
        let mut data = vec![0.0f32; 4 * size];

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

        if let Some((lr, lc)) = self.last_move {
            data[2 * size + lr * BOARD_SIZE + lc] = 1.0;
        }
        data
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
        assert_eq!(data.len(), 4 * NUM_POSITIONS);
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
}
