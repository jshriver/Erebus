//! eval: engine-wide evaluation entry point and score constants.
//!
//! The engine is NNUE-only: `evaluate` is the net's output (see `nnue.rs`).
//! `main()` installs the embedded net before any `Board` exists, so a net is
//! always present.

use crate::board::Board;

pub const INF:        i32 = 1_000_000;
pub const MATE_SCORE: i32 = 900_000;
pub const DRAW_SCORE: i32 = 0;

pub fn evaluate(board: &mut Board) -> i32 {
    crate::nnue::evaluate(board)
}
