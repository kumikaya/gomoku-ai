//! 棋盘分析模式
//!
//! 玩家自由落子，每步落子后自动运行 MCTS 搜索，在棋盘上显示：
//! - 神经网络的预估价值（NN value）
//! - 先验概率（NN prior）
//! - MCTS 修正后的 Q 值
//! - MCTS 搜索的后验概率（completed Q policy）
//! - 每个落子的模拟访问次数
//!
//! 空位格子的背景色根据 MCTS policy 概率变化（热力图），
//! 从浅灰（低概率）渐变到红色（高概率，≥15%）。
//!
//! 使用 ratatui / crossterm 在终端渲染。

use std::io;

use crate::game::board::{Board, Color};
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

pub fn analyze_game<E: Evaluator>(evaluator: &E, num_simulations: usize) {
    enable_raw_mode().unwrap();
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).unwrap();
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    let result =
        futures_executor::block_on(run_analyze_loop(&mut terminal, evaluator, num_simulations));

    disable_raw_mode().unwrap();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
    terminal.show_cursor().unwrap();

    if let Err(e) = result {
        eprintln!("Analyze error: {}", e);
    }
}

/// 分析模式下的状态
struct AnalyzeState {
    board: Board,
    cursor_row: usize,
    cursor_col: usize,
    mcts: MCTS<Board>,
    message: String,
    /// 当前已执行的上一次搜索的结果
    current_result: Option<AnalyzeResultData>,
    /// 是否正在计算中
    computing: bool,
    /// 棋盘大小
    board_size: usize,
    /// 模拟次数（可动态修改）
    num_simulations: usize,
    /// 棋盘历史（用于悔棋），每个元素是落子前的完整棋盘 clone
    board_history: Vec<Board>,
}

/// 存放搜索后需要显示的各个数据
struct AnalyzeResultData {
    nn_value: f32,
    nn_prior: Vec<f32>,
    mcts_q: Vec<f32>,
    mcts_visits: Vec<u32>,
    mcts_policy: Vec<f32>,
}

/// 输入模拟次数的浮层
struct SimInputOverlay {
    input_buffer: String,
    active: bool,
}

impl SimInputOverlay {
    fn new() -> Self {
        Self {
            input_buffer: String::new(),
            active: false,
        }
    }

    fn activate(&mut self) {
        self.active = true;
        self.input_buffer.clear();
    }

    fn deactivate(&mut self) {
        self.active = false;
        self.input_buffer.clear();
    }
}

async fn run_analyze_loop<E: Evaluator>(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    evaluator: &E,
    num_simulations: usize,
) -> io::Result<()> {
    let board = Board::new();
    let board_size = board.board_size;
    let half = board_size / 2;
    let mut state = AnalyzeState {
        board,
        cursor_row: half,
        cursor_col: half,
        mcts: MCTS::new(),
        message:
            "分析模式 — 方向键移动, 空格落子, [F] 强制刷新, [S] 修改模拟次数, [U] 悔棋, Q 退出"
                .into(),
        current_result: None,
        computing: false,
        board_size,
        num_simulations,
        board_history: Vec::new(),
    };

    let mut sim_overlay = SimInputOverlay::new();

    // 初始分析一次空棋盘
    state.computing = true;
    terminal.draw(|f| render(f, &state, &sim_overlay))?;
    run_analysis(&mut state, evaluator).await;
    state.computing = false;

    loop {
        terminal.draw(|f| render(f, &state, &sim_overlay))?;

        // 如果不在 input overlay 模式
        if sim_overlay.active {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Enter => {
                        if let Ok(n) = sim_overlay.input_buffer.parse::<usize>() {
                            if n > 0 {
                                state.num_simulations = n;
                                state.message = format!("模拟次数已设置为 {}", n);
                            }
                        }
                        sim_overlay.deactivate();
                    }
                    KeyCode::Esc => {
                        sim_overlay.deactivate();
                    }
                    KeyCode::Backspace => {
                        sim_overlay.input_buffer.pop();
                    }
                    KeyCode::Char(c) if c.is_ascii_digit() => {
                        sim_overlay.input_buffer.push(c);
                    }
                    _ => {}
                }
            }
            continue;
        }

        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => return Ok(()),
                KeyCode::Char('f') | KeyCode::Char('F') => {
                    // 强制刷新：在当前棋盘上重新跑搜索
                    if state.board.game_over {
                        state.message = "游戏已结束，无法分析".into();
                        continue;
                    }
                    state.computing = true;
                    state.message = "MCTS 搜索中...".into();
                    terminal.draw(|f| render(f, &state, &sim_overlay))?;
                    run_analysis(&mut state, evaluator).await;
                    state.computing = false;
                }
                KeyCode::Char('s') | KeyCode::Char('S') => {
                    sim_overlay.activate();
                    state.message = "请输入模拟次数 (Enter 确认, Esc 取消):".into();
                }
                KeyCode::Char('u') | KeyCode::Char('U') => {
                    // 悔棋：从历史栈弹出一个状态
                    if let Some(prev_board) = state.board_history.pop() {
                        state.board = prev_board;
                        state.message = "已悔棋".into();
                        // 悔棋后自动分析
                        state.computing = true;
                        terminal.draw(|f| render(f, &state, &sim_overlay))?;
                        run_analysis(&mut state, evaluator).await;
                        state.computing = false;
                    } else {
                        state.message = "无法悔棋（已到棋盘初始状态）".into();
                    }
                }
                KeyCode::Char(' ') | KeyCode::Enter => {
                    if state.board.game_over {
                        state.message = "游戏已结束，按 Q 退出".into();
                        continue;
                    }
                    // 保存落子前的棋盘到历史
                    state.board_history.push(state.board.clone());
                    if state.board.play(state.cursor_row, state.cursor_col) {
                        state.message = format!(
                            "落了 ({},{}) — MCTS 分析中...",
                            state.cursor_row, state.cursor_col
                        );
                        state.computing = true;
                        terminal.draw(|f| render(f, &state, &sim_overlay))?;
                        run_analysis(&mut state, evaluator).await;
                        state.computing = false;

                        if state.board.game_over {
                            state.message = format!(
                                "{} 获胜！按 Q 退出",
                                match state.board.winner {
                                    Some(Color::Black) => "黑子 ●",
                                    Some(Color::White) => "白子 ○",
                                    None => "平局",
                                }
                            );
                        }
                    } else {
                        // 落子失败，回滚历史
                        state.board_history.pop();
                        state.message = "该位置已有棋子！".into();
                    }
                }
                KeyCode::Up => {
                    state.cursor_row = state.cursor_row.saturating_sub(1);
                }
                KeyCode::Down => {
                    state.cursor_row = (state.cursor_row + 1).min(state.board.board_size - 1);
                }
                KeyCode::Left => {
                    state.cursor_col = state.cursor_col.saturating_sub(1);
                }
                KeyCode::Right => {
                    state.cursor_col = (state.cursor_col + 1).min(state.board.board_size - 1);
                }
                _ => {}
            }
        }
    }
}

async fn run_analysis<E: Evaluator>(state: &mut AnalyzeState, evaluator: &E) {
    let config = GumbelConfig::inference(state.num_simulations);
    state.mcts.reset();
    let result = state
        .mcts
        .search(&mut state.board, evaluator, &config, &mut rand::rng())
        .await;

    let mut total_visits: u32 = result.children_visits.iter().sum();
    state.current_result = Some(AnalyzeResultData {
        nn_value: result.root_nn_value,
        nn_prior: result.root_nn_prior,
        mcts_q: result.children_q,
        mcts_visits: result.children_visits,
        mcts_policy: result.policy,
    });

    if total_visits == 0 {
        total_visits = 1;
    }
    state.message = format!(
        "NN 估值: {:.3},  MCTS 根值: {:.3},  总模拟: {}",
        result.root_nn_value, result.root_value, total_visits,
    );
}

/// 核心渲染函数
fn render(frame: &mut Frame, state: &AnalyzeState, sim_overlay: &SimInputOverlay) {
    let area = frame.area();

    let info_height = if state.current_result.is_some() { 3 } else { 2 };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),              // 棋盘
            Constraint::Length(info_height), // 信息栏
        ])
        .split(area);

    // 渲染棋盘
    frame.render_widget(AnalyzeBoardWidget { state }, chunks[0]);

    // 渲染状态信息
    let message = if sim_overlay.active {
        format!(
            "模拟次数: {}_  (输入数字后按 Enter)",
            sim_overlay.input_buffer
        )
    } else if state.computing {
        "计算中...".into()
    } else {
        state.message.clone()
    };

    let msg = Paragraph::new(message).block(Block::default().borders(Borders::NONE));
    frame.render_widget(msg, chunks[1]);

    // 如果计算中且无结果，不渲染侧边栏
    if !state.computing {
        if let Some(ref result) = state.current_result {
            // 在右侧区域打印当前光标指向格子的数值
            let idx = state.board.pos_to_idx(state.cursor_row, state.cursor_col);
            let info = format!(
                " 位置 ({},{}) [{}]\n NN 价值: {:+.3}\n 先验概率: {:.3}%\n MCTS Q: {:+.3}\n MCTS 概率: {:.3}%\n 访问次数: {}",
                state.cursor_row,
                state.cursor_col,
                if state.board.is_empty(state.cursor_row, state.cursor_col) {
                    "空"
                } else {
                    "落子"
                },
                result.nn_value,
                result.nn_prior.get(idx).copied().unwrap_or(0.0) * 100.0,
                result.mcts_q.get(idx).copied().unwrap_or(0.0),
                result.mcts_policy.get(idx).copied().unwrap_or(0.0) * 100.0,
                result.mcts_visits.get(idx).copied().unwrap_or(0),
            );

            let info_panel = Paragraph::new(info)
                .block(Block::default().borders(Borders::ALL).title(" 分析 "))
                .style(Style::default().fg(TuiColor::Cyan));

            // 放在棋盘区域的右下角或右侧
            let board_width = (state.board_size * 3 + 1) as u16;
            let x_rem = area.width.saturating_sub(board_width);
            let info_rect = Rect {
                x: area.x + board_width + 1,
                y: chunks[0].y,
                width: (x_rem.saturating_sub(2)).min(28),
                height: 9,
            };
            if info_rect.width > 0 {
                frame.render_widget(info_panel, info_rect);
            }
        }
    }

    // 模拟输入浮层
    if sim_overlay.active {
        // 简单处理：input 已在上方信息栏中显示
    }
}

/// 棋盘渲染组件（带热力图数字覆盖）
struct AnalyzeBoardWidget<'a> {
    state: &'a AnalyzeState,
}

impl Widget for AnalyzeBoardWidget<'_> {
    fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let bs = self.state.board.board_size;
        let board_width = (bs * 3 + 1) as u16;
        let board_height = (bs + 2) as u16;

        if area.width < board_width || area.height < board_height + 2 {
            buf.set_string(
                area.x,
                area.y,
                "Terminal too small — resize and restart",
                Style::default().fg(TuiColor::Red),
            );
            return;
        }

        let x_offset = area.x + (area.width.saturating_sub(board_width)) / 2;
        let y_offset = area.y + (area.height.saturating_sub(board_height)) / 2;

        // 列标
        for c in 0..bs {
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

        // 如果正在计算中，显示"计算中..."
        if self.state.computing {
            for r in 0..bs {
                let row_label = format!("{:2}", r);
                buf.set_string(
                    x_offset,
                    y_offset + r as u16 + 1,
                    row_label,
                    Style::default().fg(TuiColor::Gray),
                );
                for c in 0..bs {
                    let sx = x_offset + c as u16 * 3 + 2;
                    let sy = y_offset + r as u16 + 1;
                    let cell = self.state.board.get(r, c);
                    let cell_str = match cell {
                        0 => "⋅",
                        1 => "●",
                        2 => "○",
                        _ => "?",
                    };
                    let style = match cell {
                        1 => Style::default().fg(TuiColor::Black),
                        2 => Style::default().fg(TuiColor::White),
                        _ => Style::default().fg(TuiColor::DarkGray),
                    };
                    buf.set_string(sx, sy, cell_str, style);
                }
            }
            return;
        }

        // 获取当前显示数据（NN prior / MCTS Q / MCTS policy / visits）
        // 优先使用 MCTS policy 作为热力图强度依据
        let data: Option<&[f32]> = self
            .state
            .current_result
            .as_ref()
            .map(|r| r.mcts_policy.as_slice());

        for r in 0..bs {
            let row_label = format!("{:2}", r);
            buf.set_string(
                x_offset,
                y_offset + r as u16 + 1,
                row_label,
                Style::default().fg(TuiColor::Gray),
            );

            for c in 0..bs {
                let sx = x_offset + c as u16 * 3 + 2;
                let sy = y_offset + r as u16 + 1;
                let idx = r * bs + c;

                let cell = self.state.board.get(r, c);
                let is_cursor = self.state.cursor_row == r && self.state.cursor_col == c;

                // 获取该位置的 MCTS policy 概率
                let prob = data.and_then(|d| d.get(idx)).copied();

                let (cell_str, style) = match (cell, is_cursor, prob) {
                    // 光标在空格上
                    (0, true, _) => (
                        "+",
                        Style::default()
                            .fg(TuiColor::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    // 空格有热力图数据（概率 > 0.1%）：背景色填充
                    (0, false, Some(p)) if p > 0.001 => (" ", heatmap_style(p)),
                    // 空格无显著数据
                    (0, false, _) => ("⋅", Style::default().fg(TuiColor::DarkGray)),
                    // 黑子
                    (1, _, _) => ("●", Style::default().fg(TuiColor::Black)),
                    // 白子
                    (2, _, _) => ("○", Style::default().fg(TuiColor::White)),
                    _ => ("?", Style::default().fg(TuiColor::DarkGray)),
                };

                buf.set_string(sx, sy, cell_str, style);
            }
        }

        // 在天元位置加标记（仅当没有热力图覆盖时）
        let half_size = bs / 2;
        if self.state.board.get(half_size, half_size) == 0
            && !(self.state.cursor_row == half_size && self.state.cursor_col == half_size)
        {
            let star_prob = data
                .and_then(|d| d.get(half_size * bs + half_size))
                .copied();
            // 只有概率很低或没有数据时才画星位
            let should_draw = star_prob.map_or(true, |p| p <= 0.001);
            if should_draw {
                let star_x = x_offset + half_size as u16 * 3 + 2;
                let star_y = y_offset + half_size as u16 + 1;
                buf.set_string(star_x, star_y, "╋", Style::default().fg(TuiColor::DarkGray));
            }
        }
    }
}

/// 根据 MCTS 概率返回热力图颜色
/// 根据概率返回热力图背景样式（白→红渐变）
fn heatmap_style(prob: f32) -> Style {
    // 将概率映射到 [0, 1]，超过 15% 视为满红
    let t = (prob / 0.15).clamp(0.0, 1.0);
    // 从浅灰渐变到红色
    let r = (220.0 + 35.0 * t) as u8;
    let g = (220.0 * (1.0 - t)) as u8;
    let b = (220.0 * (1.0 - t)) as u8;
    Style::default()
        .fg(TuiColor::Black)
        .bg(TuiColor::Rgb(r, g, b))
}
