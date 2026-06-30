//! 人机对弈模块
//!
//! 使用 ratatui / crossterm 在终端渲染 15×15 棋盘，
//! 人类玩家用方向键选位、空格落子，AI 使用 MCTS + 神经网络做出回应。

use std::io;

use crate::game::board::{BOARD_SIZE, Board, Color, NUM_POSITIONS};
use crate::inference::Evaluator;
use crate::mcts::node::{GumbelConfig, MCTS};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color as TuiColor, Modifier, Style},
    widgets::{Block, Borders, Paragraph, Widget},
};

pub fn play_game<E: Evaluator>(evaluator: &E, num_simulations: usize) {
    enable_raw_mode().unwrap();
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).unwrap();
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    let result = run_game_loop(&mut terminal, evaluator, num_simulations);

    disable_raw_mode().unwrap();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
    terminal.show_cursor().unwrap();

    if let Err(e) = result {
        eprintln!("Game error: {}", e);
    }
}

/// 游戏状态
struct GameState {
    board: Board,
    cursor_row: usize,
    cursor_col: usize,
    mcts: MCTS,
    message: String,
    player_color: Color,
    game_over: bool,
}

fn run_game_loop<E: Evaluator>(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    evaluator: &E,
    num_simulations: usize,
) -> io::Result<()> {
    let player_color = Color::Black;

    let mut state = GameState {
        board: Board::new(),
        cursor_row: 7,
        cursor_col: 7,
        mcts: MCTS::new(),
        message: format!(
            "你的回合 ({}) — 箭头键移动, 空格落子, Q 退出",
            color_name(player_color)
        ),
        player_color,
        game_over: false,
    };

    // AI 先手 → 人类先手始终是黑方，直接等待人类输入
    loop {
        terminal.draw(|f| render(f, &state))?;

        if state.game_over {
            // 等待 Q 退出
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    return Ok(());
                }
            }
            continue;
        }

        // 人类回合
        if state.board.current_player == state.player_color {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char(' ') | KeyCode::Enter => {
                        if state.board.play(state.cursor_row, state.cursor_col) {
                            state.message =
                                format!("你落了 ({},{})...", state.cursor_row, state.cursor_col);
                            // 如果人类落子后游戏未结束，轮到 AI
                            if !state.board.game_over {
                                terminal.draw(|f| render(f, &state))?;
                                ai_move(&mut state, evaluator, num_simulations);
                            }
                        } else {
                            state.message = "该位置已有棋子！".into();
                        }
                    }
                    KeyCode::Up => {
                        state.cursor_row = state.cursor_row.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        state.cursor_row = (state.cursor_row + 1).min(BOARD_SIZE - 1);
                    }
                    KeyCode::Left => {
                        state.cursor_col = state.cursor_col.saturating_sub(1);
                    }
                    KeyCode::Right => {
                        state.cursor_col = (state.cursor_col + 1).min(BOARD_SIZE - 1);
                    }
                    _ => {}
                }
            }
        }
    }
}

fn ai_move<E: Evaluator>(state: &mut GameState, evaluator: &E, num_simulations: usize) {
    let config = GumbelConfig::pure_gumbel(num_simulations);
    let result = state.mcts.search(&mut state.board, evaluator, &config);

    if result.best_move < NUM_POSITIONS {
        state.board.play_idx(result.best_move);
        let (r, c) = Board::idx_to_pos(result.best_move);
        state.message = format!("AI 落了 ({},{}) — 你的回合", r, c);
    }
}

fn render(frame: &mut Frame, state: &GameState) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // 棋盘
            Constraint::Length(2), // 信息
        ])
        .split(area);

    // 渲染棋盘
    frame.render_widget(BoardWidget { state }, chunks[0]);

    // 渲染状态信息
    let msg = Paragraph::new(state.message.as_str())
        .block(Block::default().borders(Borders::NONE))
        .style(Style::default().fg(TuiColor::Yellow));
    frame.render_widget(msg, chunks[1]);
}

/// 棋盘渲染组件
struct BoardWidget<'a> {
    state: &'a GameState,
}

impl Widget for BoardWidget<'_> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        // 计算棋盘左上角偏移，居中显示
        let board_width = (BOARD_SIZE * 3 + 1) as u16;
        let board_height = (BOARD_SIZE + 2) as u16;
        let x_offset = area.x + (area.width.saturating_sub(board_width)) / 2;
        let y_offset = area.y + (area.height.saturating_sub(board_height)) / 2;

        // 列标
        for c in 0..BOARD_SIZE {
            let ch = (b'A' + c as u8) as char;
            buf.set_string(
                x_offset + c as u16 * 3 + 2,
                y_offset,
                ch.to_string(),
                Style::default()
                    .fg(TuiColor::Gray)
                    .add_modifier(Modifier::BOLD),
            );
        }

        for r in 0..BOARD_SIZE {
            // 行标
            let row_label = format!("{:2}", r);
            buf.set_string(
                x_offset,
                y_offset + r as u16 + 1,
                row_label,
                Style::default().fg(TuiColor::Gray),
            );

            for c in 0..BOARD_SIZE {
                let sx = x_offset + c as u16 * 3 + 2;
                let sy = y_offset + r as u16 + 1;

                let cell = self.state.board.get(r, c);
                let is_cursor = self.state.cursor_row == r && self.state.cursor_col == c;

                let cell_str = match (cell, is_cursor) {
                    (0, true) => "+",
                    (0, false) => "⋅",
                    (1, _) => "●",
                    (2, _) => "○",
                    _ => "?",
                };

                let style = match (cell, is_cursor) {
                    (0, true) => Style::default()
                        .fg(TuiColor::Green)
                        .add_modifier(Modifier::BOLD),
                    (1, _) => Style::default().fg(TuiColor::Black),
                    (2, _) => Style::default().fg(TuiColor::White),
                    _ => Style::default().fg(TuiColor::DarkGray),
                };

                buf.set_string(sx, sy, cell_str, style);
            }
        }

        // 在 (7,7) 天元位置加标记
        let star_x = x_offset + 7 * 3 + 2;
        let star_y = y_offset + 7 + 1;
        if self.state.board.get(7, 7) == 0
            && !(self.state.cursor_row == 7 && self.state.cursor_col == 7)
        {
            buf.set_string(star_x, star_y, "╋", Style::default().fg(TuiColor::DarkGray));
        }
    }
}

fn color_name(c: Color) -> &'static str {
    match c {
        Color::Black => "黑子 ●",
        Color::White => "白子 ○",
    }
}
