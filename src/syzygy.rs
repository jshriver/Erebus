//! Syzygy endgame tablebase support via `shakmaty-syzygy` 0.28 / shakmaty 0.30.
//!
//! # Conversion notes (shakmaty 0.30 API)
//!
//! * `Setup` is a plain struct — build it with struct literal syntax.
//! * `castling_rights` is a raw `Bitboard` of rook squares (not a `Castles`).
//!   For standard chess the four corners are A1, H1, A8, H8.
//! * `fullmoves` is `NonZeroU32`.
//! * `promoted` must be set to `Bitboard::EMPTY`.
//! * `Chess::from_setup()` can return errors for castling/ep issues that are
//!   still legally recoverable; we use the `ignore_*` escape hatches.
//! * The WDL probe after a zeroing move is `probe_wdl_after_zeroing()`.
//!   We gate every probe on `halfmove == 0` so this is always correct.
//! * `Board::set_piece_at()` is the only public way to populate a shakmaty Board.

use std::num::NonZeroU32;
use std::path::Path;

use shakmaty::{
    Bitboard as SmBB, Board as SmBoard, CastlingMode, Chess, Color as SmColor,
    File as SmFile, FromSetup, Piece as SmPiece, PositionError,
    Rank as SmRank, Role, Setup, Square as SmSquare,
};
use shakmaty_syzygy::{MaybeRounded, Tablebase, Wdl};

use crate::board::{Board, Color, Piece, CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum WdlResult {
    Win,
    Draw,
    Loss,
    /// Win that counts as a draw under the 50-move rule
    CursedWin,
    /// Loss that counts as a draw under the 50-move rule
    BlessedLoss,
}

pub const TB_WIN_SCORE:  i32 = 800_000;
pub const TB_LOSS_SCORE: i32 = -800_000;

/// Convert a WDL probe result into a search score relative to the side to move.
///
/// `dtz` (distance to zeroing, signed) is used to prefer shorter wins and
/// longer losses, improving practical play without a separate DTZ root search.
///
/// A cursed win / blessed loss cannot be converted inside the 50-move rule,
/// so it is a draw. It scores one centipawn off zero: still a draw to the
/// search, but a cursed win is preferred over a dead draw (the opponent may
/// err) and a blessed loss over a real loss.
pub fn tb_wdl_score(wdl: WdlResult, dtz: Option<i32>, ply: usize) -> i32 {
    match wdl {
        WdlResult::Win => {
            let dtz_offset = dtz.map_or(0, |d| d.abs().min(500));
            TB_WIN_SCORE - ply as i32 - dtz_offset
        }
        WdlResult::Loss => {
            let dtz_offset = dtz.map_or(0, |d| d.abs().min(500));
            TB_LOSS_SCORE + ply as i32 + dtz_offset
        }
        WdlResult::CursedWin   => 1,
        WdlResult::BlessedLoss => -1,
        WdlResult::Draw        => 0,
    }
}

pub struct SyzygyTb {
    tables: Tablebase<Chess>,
}

impl SyzygyTb {
    pub fn new() -> Self {
        SyzygyTb { tables: Tablebase::new() }
    }

    /// Add a directory of `.rtbw` / `.rtbz` files.
    /// Returns the number of tablebase files successfully loaded.
    pub fn add_directory(&mut self, path: &str) -> usize {
        match self.tables.add_directory(Path::new(path)) {
            Ok(n)  => n,
            Err(e) => {
                eprintln!("info string Syzygy error loading {}: {}", path, e);
                0
            }
        }
    }

    /// True if at least one tablebase file is loaded.
    pub fn is_loaded(&self) -> bool {
        self.tables.max_pieces() > 0
    }

    /// Maximum number of pieces covered by the loaded tablebases.
    pub fn max_pieces(&self) -> usize {
        self.tables.max_pieces()
    }

    /// Probe WDL (Win / Draw / Loss) for the current position.
    ///
    /// Returns `None` when the position has more pieces than the loaded
    /// tablebases cover, the 50-move clock is non-zero, or any I/O error.
    pub fn probe_wdl(&self, board: &Board) -> Option<WdlResult> {
        if !self.should_probe(board) { return None; }
        let pos = board_to_chess(board)?;
        match self.tables.probe_wdl_after_zeroing(&pos) {
            Ok(wdl) => Some(convert_wdl(wdl)),
            Err(e)  => {
                eprintln!("info string Syzygy WDL probe error: {}", e);
                None
            }
        }
    }

    /// Probe DTZ (Distance to Zeroing move).
    ///
    /// Positive = side to move is winning; value is half-moves to the next
    /// pawn move or capture.  Negative = losing.  Zero = drawn.
    pub fn probe_dtz(&self, board: &Board) -> Option<i32> {
        if !self.should_probe(board) { return None; }
        let pos = board_to_chess(board)?;
        match self.tables.probe_dtz(&pos) {
            Ok(MaybeRounded::Precise(dtz)) => Some(dtz.0),
            Ok(MaybeRounded::Rounded(dtz)) => Some(dtz.0),
            Err(e) => {
                eprintln!("info string Syzygy DTZ probe error: {}", e);
                None
            }
        }
    }

    /// Gate every probe: only attempt when the position fits within the loaded
    /// tablebases and the 50-move counter is zero.
    ///
    /// `probe_wdl_after_zeroing` is defined for positions immediately after a
    /// zeroing move (capture or pawn advance), i.e. with halfmove clock == 0.
    /// Probing with a non-zero clock can return misleading scores — a position
    /// that appears to be a win might actually be a draw due to the clock
    /// running out before the tablebase can force the zeroing move.
    #[inline]
    fn should_probe(&self, board: &Board) -> bool {
        if !self.is_loaded() { return false; }
        let piece_count = board.occupied().count_ones() as usize;
        piece_count <= self.max_pieces() && board.halfmove == 0
    }
}

/// Convert the engine's `Board` into a `shakmaty::Chess` position.
///
/// Returns `None` if the position cannot be validated (should not happen in
/// practice since we only probe legal positions).

#[allow(clippy::result_large_err)]
pub(crate) fn board_to_chess(board: &Board) -> Option<Chess> {
    let sm_board      = build_sm_board(board);
    let turn          = if board.side == Color::White { SmColor::White } else { SmColor::Black };
    let ep_square     = board.ep_sq.map(sq_to_sm);
    let castling_rights = build_castling_rights(board);
    let halfmoves     = board.halfmove as u32;
    let fullmoves     = NonZeroU32::new(board.fullmove as u32)
                            .unwrap_or(NonZeroU32::MIN);

    let setup = Setup {
        board:            sm_board,
        turn,
        castling_rights,
        ep_square,
        halfmoves,
        fullmoves,

        promoted:         SmBB::EMPTY,
        pockets:          None,
        remaining_checks: None,
    };

    Chess::from_setup(setup, CastlingMode::Standard)
        .or_else(|e: PositionError<Chess>| {
            e.ignore_invalid_ep_square()
        })
        .or_else(|e: PositionError<Chess>| {
            e.ignore_invalid_castling_rights()
        })
        .ok()
}

/// Build a `shakmaty::Board` by iterating over every occupied square.
///
/// `Board::set_piece_at()` is the only stable public API for constructing a
/// shakmaty Board piece by piece (the struct fields are private in 0.30).
fn build_sm_board(board: &Board) -> SmBoard {
    let mut sm = SmBoard::empty();
    for sq in 0u8..64 {
        if let Some((piece, color)) = board.sq_piece[sq as usize] {
            let role = match piece {
                Piece::Pawn   => Role::Pawn,
                Piece::Knight => Role::Knight,
                Piece::Bishop => Role::Bishop,
                Piece::Rook   => Role::Rook,
                Piece::Queen  => Role::Queen,
                Piece::King   => Role::King,
            };
            let color = if color == Color::White { SmColor::White } else { SmColor::Black };
            sm.set_piece_at(sq_to_sm(sq), SmPiece { role, color });
        }
    }
    sm
}

/// Build the castling-rights `Bitboard` expected by shakmaty 0.30.
///
/// In shakmaty 0.30, `Setup::castling_rights` is a raw bitboard of the rook
/// squares that still have castling rights, not a `Castles` object.
/// For standard chess those are the four corner squares: A1, H1, A8, H8.
fn build_castling_rights(board: &Board) -> SmBB {
    let mut bb = SmBB::EMPTY;
    if board.castling & CASTLE_WQ != 0 { bb |= SmBB::from_square(SmSquare::A1); }
    if board.castling & CASTLE_WK != 0 { bb |= SmBB::from_square(SmSquare::H1); }
    if board.castling & CASTLE_BQ != 0 { bb |= SmBB::from_square(SmSquare::A8); }
    if board.castling & CASTLE_BK != 0 { bb |= SmBB::from_square(SmSquare::H8); }
    bb
}

/// Convert an engine square index (0 = a1 … 63 = h8) to `shakmaty::Square`.
#[inline]
fn sq_to_sm(sq: u8) -> SmSquare {
    let file = SmFile::new((sq & 7) as u32);
    let rank = SmRank::new((sq >> 3) as u32);
    SmSquare::from_coords(file, rank)
}

/// Convert `shakmaty::Wdl` to the engine's `WdlResult`.
#[inline]
fn convert_wdl(wdl: Wdl) -> WdlResult {
    match wdl {
        Wdl::Win         => WdlResult::Win,
        Wdl::Draw        => WdlResult::Draw,
        Wdl::Loss        => WdlResult::Loss,
        Wdl::CursedWin   => WdlResult::CursedWin,
        Wdl::BlessedLoss => WdlResult::BlessedLoss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursed_and_blessed_score_as_draws() {
        assert_eq!(tb_wdl_score(WdlResult::CursedWin, None, 5), 1);
        assert_eq!(tb_wdl_score(WdlResult::BlessedLoss, None, 5), -1);
        assert_eq!(tb_wdl_score(WdlResult::Draw, None, 5), 0);

        let fast = tb_wdl_score(WdlResult::Win, Some(3), 5);
        let slow = tb_wdl_score(WdlResult::Win, Some(40), 5);
        assert!(fast > slow && slow > 700_000);
        assert!(tb_wdl_score(WdlResult::Loss, Some(3), 5) < -700_000);
    }
}
