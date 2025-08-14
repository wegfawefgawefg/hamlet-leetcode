use crate::solver::{dims, Board, Hamlet, EMPTY};
use colored::*;

pub fn viz_in_board(board: &Board, ham: &Hamlet, show_coords: bool) -> String {
    let (w, h) = dims(board);
    let mut rows: Vec<String> = Vec::new();
    if show_coords {
        let mut header = String::from("   ");
        for x in 0..w {
            header.push_str(&format!("{} ", x % 10));
        }
        rows.push(header.trim_end().to_string());
    }
    for y in 0..h {
        let mut line = String::new();
        if show_coords {
            line.push_str(&format!("{:>2} ", y % 10));
        }
        for x in 0..w {
            let ch = board[y][x];
            if ham.contains(&(x, y)) && ch != EMPTY {
                line.push_str(&format!("{} ", ch.to_string().green().bold()));
            } else {
                line.push_str(&format!("{} ", ch));
            }
        }
        rows.push(line.trim_end().to_string());
    }
    rows.join("\n")
}
