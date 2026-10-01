//! Polyglot opening book probing.
//!
//! A Polyglot `.bin` book is a flat array of 16-byte big-endian entries
//! `key:u64 | move:u16 | weight:u16 | learn:u32`, sorted by key. The key is
//! the Polyglot Zobrist hash of the position, which shakmaty's `Zobrist64`
//! reproduces exactly (shakmaty tests it against the Polyglot reference keys).
//!
//! Move encoding: bits 0-5 to-square, 6-11 from-square (a1 = 0 ... h8 = 63,
//! same as the engine), 12-14 promotion (1 = N, 2 = B, 3 = R, 4 = Q).
//! Castling is stored as king-takes-own-rook (`e1h1`), which we translate to
//! the engine's king-two-squares form (`e1g1`).

use std::path::Path;

use shakmaty::{EnPassantMode, Position, zobrist::Zobrist64};

use crate::board::{Board, Piece};
use crate::moves::Move;
use crate::syzygy::board_to_chess;

#[derive(Copy, Clone)]
struct Entry {
    key:    u64,
    mv:     u16,
    weight: u16,
}

pub struct Book {
    entries: Vec<Entry>,
}

impl Book {
    /// Load a Polyglot book into memory.
    pub fn load(path: impl AsRef<Path>) -> Result<Book, String> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| e.to_string())?;
        if bytes.is_empty() || bytes.len() % 16 != 0 {
            return Err(format!("size {} is not a multiple of 16 bytes", bytes.len()));
        }
        let entries: Vec<Entry> = bytes
            .chunks_exact(16)
            .map(|c| Entry {
                key:    u64::from_be_bytes(c[0..8].try_into().unwrap()),
                mv:     u16::from_be_bytes(c[8..10].try_into().unwrap()),
                weight: u16::from_be_bytes(c[10..12].try_into().unwrap()),
            })
            .collect();
        if !entries.windows(2).all(|w| w[0].key <= w[1].key) {
            return Err("entries are not sorted by key".to_string());
        }
        Ok(Book { entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// All legal book moves for `board` with their weights, in book order.
    /// Entries whose move isn't legal here (corrupt book or hash collision)
    /// are dropped.
    pub fn moves(&self, board: &Board) -> Vec<(Move, u16)> {
        let Some(key) = polyglot_key(board) else { return Vec::new() };
        let start = self.entries.partition_point(|e| e.key < key);
        self.entries[start..]
            .iter()
            .take_while(|e| e.key == key)
            .filter_map(|e| decode_move(board, e.mv).map(|m| (m, e.weight)))
            .collect()
    }

    /// Pick a book move at random, weighted by the book's weights. `rand` is
    /// any uniformly distributed u64. Returns `None` when out of book.
    pub fn pick(&self, board: &Board, rand: u64) -> Option<Move> {
        let moves = self.moves(board);
        let total: u64 = moves.iter().map(|&(_, w)| w as u64).sum();
        if total == 0 {

            return (!moves.is_empty()).then(|| moves[(rand % moves.len() as u64) as usize].0);
        }
        let mut r = rand % total;
        for &(mv, w) in &moves {
            if r < w as u64 {
                return Some(mv);
            }
            r -= w as u64;
        }
        None
    }
}

/// Polyglot Zobrist key of `board`. The en passant square is hashed only when
/// a side-to-move pawn could capture onto it, as the Polyglot spec says.
pub fn polyglot_key(board: &Board) -> Option<u64> {
    let pos = board_to_chess(board)?;
    Some(pos.zobrist_hash::<Zobrist64>(EnPassantMode::PseudoLegal).0)
}

/// Translate a Polyglot move to the matching legal engine move.
fn decode_move(board: &Board, raw: u16) -> Option<Move> {
    let to    = (raw & 0x3F) as u8;
    let from  = ((raw >> 6) & 0x3F) as u8;
    let promo = (raw >> 12) & 0x7;

    let mut to = to;
    if let (Some((Piece::King, kc)), Some((Piece::Rook, rc))) =
        (board.sq_piece[from as usize], board.sq_piece[to as usize])
        && kc == rc
    {
        to = if to > from { from + 2 } else { from - 2 };
    }

    let mut uci = format!("{}{}", sq_name(from), sq_name(to));
    match promo {
        0 => {}
        1 => uci.push('n'),
        2 => uci.push('b'),
        3 => uci.push('r'),
        4 => uci.push('q'),
        _ => return None,
    }
    board.find_uci_move(&uci)
}

fn sq_name(sq: u8) -> String {
    format!("{}{}", (b'a' + (sq & 7)) as char, (b'1' + (sq >> 3)) as char)
}

/// Small xorshift64* generator for picking book moves; no `rand` dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    /// Seeded from the clock -- fine for varying book choices between games.
    pub fn from_time() -> Rng {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Rng::new(nanos ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Polyglot reference keys from the spec (the same set shakmaty checks).
    #[test]
    fn polyglot_reference_keys() {
        let cases = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0x463b_9618_1691_fc9c),
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 0x823c_9b50_fd11_4196),
            ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2", 0x662f_afb9_65db_29d4),
            ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 0x22a4_8b5a_8e47_ff78),
            ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPPKPPP/RNBQ1BNR b kq - 1 3", 0x652a_607c_a3f2_42c1),
            ("rnbqkbnr/p1pppppp/8/8/PpP4P/8/1P1PPPP1/RNBQKBNR b KQkq c3 0 3", 0x3c81_23ea_7b06_7637),
        ];
        for (fen, want) in cases {
            let b = Board::from_fen(fen).unwrap();
            assert_eq!(polyglot_key(&b), Some(want), "{fen}");
        }
    }

    #[test]
    fn castling_decodes_to_king_two_squares() {
        let b = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();

        assert_eq!(decode_move(&b, (4 << 6) | 7).unwrap().to_uci(), "e1g1");
        assert_eq!(decode_move(&b, 4 << 6).unwrap().to_uci(), "e1c1");
    }

    #[test]
    fn promotion_decodes() {
        let b = Board::from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1").unwrap();

        assert_eq!(decode_move(&b, (1 << 12) | (48 << 6) | 56).unwrap().to_uci(), "a7a8n");
        assert_eq!(decode_move(&b, (4 << 12) | (48 << 6) | 56).unwrap().to_uci(), "a7a8q");
    }

    /// The datagen book (tracked in the repo). Walks random weighted lines
    /// from startpos until each leaves the book; every move must be legal,
    /// and the book must cover startpos.
    #[test]
    fn tcec26_random_lines() {
        let book = Book::load("books/TCEC26.bin").expect("books/TCEC26.bin");
        assert!(!book.moves(&Board::start_pos()).is_empty(), "startpos not in book");
        let mut rng = Rng::new(12345);
        for _ in 0..200 {
            let mut b = Board::start_pos();
            let mut ply = 0;
            while let Some(mv) = book.pick(&b, rng.next()) {
                b.make_move(mv);
                ply += 1;
                assert!(ply < 200, "book line never ends");
            }
            assert!(ply > 0);
        }
    }
}
