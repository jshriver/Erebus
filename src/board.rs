use std::sync::Arc;

use crate::bitboard::*;
use crate::moves::{Move, MoveFlag};
use crate::zobrist::ZOBRIST;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Color { White = 0, Black = 1 }

impl Color {
    #[inline] pub fn flip(self) -> Self {
        if self == Color::White { Color::Black } else { Color::White }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Piece { Pawn = 0, Knight = 1, Bishop = 2, Rook = 3, Queen = 4, King = 5 }

impl Piece {
    pub fn from_char(c: char) -> Option<(Piece, Color)> {
        let color = if c.is_uppercase() { Color::White } else { Color::Black };
        let piece = match c.to_ascii_lowercase() {
            'p' => Piece::Pawn,
            'n' => Piece::Knight,
            'b' => Piece::Bishop,
            'r' => Piece::Rook,
            'q' => Piece::Queen,
            'k' => Piece::King,
            _   => return None,
        };
        Some((piece, color))
    }
    pub fn to_char(self, color: Color) -> char {
        let c = match self {
            Piece::Pawn   => 'p',
            Piece::Knight => 'n',
            Piece::Bishop => 'b',
            Piece::Rook   => 'r',
            Piece::Queen  => 'q',
            Piece::King   => 'k',
        };
        if color == Color::White { c.to_ascii_uppercase() } else { c }
    }
}

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

use crate::nnue::FeatDelta;

#[derive(Copy, Clone)]
pub struct IrrevState {
    pub captured:    Option<Piece>,
    pub ep_sq:       Option<u8>,
    pub castling:    u8,
    pub halfmove:    u16,
    pub hash:        u64,
}

/// Cheap to clone: the attack tables are process-wide statics, so a clone is
/// the bitboards, the move history and the NNUE accumulator stack.
#[derive(Clone)]
pub struct Board {
    /// pieces[color][piece]
    pub pieces:   [[Bitboard; 6]; 2],
    /// color_bb[color] = union of all pieces of that color
    pub color_bb: [Bitboard; 2],
    /// piece on each square (for fast lookup during unmake / eval)
    pub sq_piece: [Option<(Piece, Color)>; 64],

    pub side:       Color,
    pub ep_sq:      Option<u8>,
    pub castling:   u8,
    pub halfmove:   u16,
    pub fullmove:   u16,
    pub hash:       u64,

    pub history:    Vec<(Move, IrrevState)>,

    nnue_stack:      crate::nnue::AccStack,
    net:             Option<Arc<crate::nnue::Net>>,
}

impl Board {
    pub fn empty() -> Self {
        Board {
            pieces:   [[0; 6]; 2],
            color_bb: [0; 2],
            sq_piece: [None; 64],
            side:     Color::White,
            ep_sq:    None,
            castling: 0,
            halfmove: 0,
            fullmove: 1,
            hash:     0,
            history:  Vec::new(),
            nnue_stack:  crate::nnue::AccStack::new(),
            net:         crate::nnue::current_net(),
        }
    }

    pub fn start_pos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    #[inline] pub fn occupied(&self) -> Bitboard { self.color_bb[0] | self.color_bb[1] }

    #[inline]
    pub fn piece_bb(&self, color: Color, piece: Piece) -> Bitboard {
        self.pieces[color as usize][piece as usize]
    }

    #[inline]
    fn set_piece(&mut self, sq: u8, piece: Piece, color: Color) {
        let b = bit(sq);
        self.pieces[color as usize][piece as usize] |= b;
        self.color_bb[color as usize] |= b;
        self.sq_piece[sq as usize] = Some((piece, color));
        self.hash ^= ZOBRIST.piece[color as usize][piece as usize][sq as usize];
    }

    #[inline]
    fn clear_piece(&mut self, sq: u8) {
        if let Some((piece, color)) = self.sq_piece[sq as usize] {
            let b = bit(sq);
            self.pieces[color as usize][piece as usize] &= !b;
            self.color_bb[color as usize] &= !b;
            self.sq_piece[sq as usize] = None;
            self.hash ^= ZOBRIST.piece[color as usize][piece as usize][sq as usize];
        }
    }

    /// The board's own net snapshot and this position's accumulators, or
    /// `None` if no net was installed when this board was built. Brings the
    /// lazy accumulator stack up to date; otherwise plain field reads, no
    /// synchronisation. Used by `nnue::evaluate`.
    #[inline]
    pub fn nnue_state(&mut self) -> Option<(&crate::nnue::Net, &crate::nnue::DualAccumulator)> {
        let net = self.net.as_deref()?;
        let acc = self.nnue_stack.current(net);

        #[cfg(debug_assertions)]
        {
            let mut check = crate::nnue::DualAccumulator::zeroed();
            crate::nnue::refresh_all(net, &self.sq_piece, &mut check);
            debug_assert_eq!(check.white.v, acc.white.v, "lazy NNUE accumulator (White) desynced from a full refresh");
            debug_assert_eq!(check.black.v, acc.black.v, "lazy NNUE accumulator (Black) desynced from a full refresh");
        }

        Some((net, acc))
    }

    /// Re-take the global net snapshot and rebuild the accumulators from it.
    /// Called by the UCI layer after a runtime EvalFile swap so this board
    /// stops evaluating with the previous net's weights.
    pub fn reload_net(&mut self) {
        self.net = crate::nnue::current_net();
        self.refresh_nnue();
    }

    /// Full recompute of both NNUE accumulators from the current board
    /// state, as the new base of the accumulator stack. O(pieces) — called
    /// once per FEN parse (see `from_fen`), by `reload_net`, and at the root
    /// of each search. Every move after that is handled incrementally.
    pub fn refresh_nnue(&mut self) {
        let Some(net) = self.net.as_deref() else { return };
        crate::nnue::refresh_all(net, &self.sq_piece, self.nnue_stack.reset());
    }

    pub fn from_fen(fen: &str) -> Option<Self> {
        let mut board = Board::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 { return None; }

        let mut rank: i32 = 7;
        let mut file: i32 = 0;
        for c in parts[0].chars() {
            match c {
                '/' => { rank -= 1; file = 0; }
                '1'..='8' => { file += (c as i32) - ('0' as i32); }
                _ => {
                    if let Some((piece, color)) = Piece::from_char(c) {
                        let sq = (rank * 8 + file) as u8;
                        board.set_piece(sq, piece, color);
                        file += 1;
                    }
                }
            }
        }

        board.side = if parts[1] == "b" { Color::Black } else { Color::White };

        board.castling = 0;
        for c in parts[2].chars() {
            match c {
                'K' => board.castling |= CASTLE_WK,
                'Q' => board.castling |= CASTLE_WQ,
                'k' => board.castling |= CASTLE_BK,
                'q' => board.castling |= CASTLE_BQ,
                _ => {}
            }
        }

        board.ep_sq = str_to_sq(parts[3]);

        board.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        board.fullmove = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        if board.side == Color::Black { board.hash ^= ZOBRIST.side; }
        for i in 0..4 {
            if board.castling & (1 << i) != 0 { board.hash ^= ZOBRIST.castle[i]; }
        }
        if let Some(ep) = board.ep_sq {
            board.hash ^= ZOBRIST.ep_file[(ep & 7) as usize];
        }

        board.refresh_nnue();

        Some(board)
    }

    pub fn to_fen(&self) -> String {

        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0u8;
            for file in 0..8 {
                let sq = rank * 8 + file;
                if let Some((piece, color)) = self.sq_piece[sq as usize] {
                    if empty > 0 { fen.push((b'0' + empty) as char); empty = 0; }
                    fen.push(piece.to_char(color));
                } else {
                    empty += 1;
                }
            }
            if empty > 0 { fen.push((b'0' + empty) as char); }
            if rank > 0 { fen.push('/'); }
        }
        fen.push(' ');
        fen.push(if self.side == Color::White { 'w' } else { 'b' });
        fen.push(' ');
        if self.castling == 0 {
            fen.push('-');
        } else {
            if self.castling & CASTLE_WK != 0 { fen.push('K'); }
            if self.castling & CASTLE_WQ != 0 { fen.push('Q'); }
            if self.castling & CASTLE_BK != 0 { fen.push('k'); }
            if self.castling & CASTLE_BQ != 0 { fen.push('q'); }
        }
        fen.push(' ');
        match self.ep_sq {
            Some(s) => fen.push_str(&sq_to_str(s)),
            None => fen.push('-'),
        }
        fen.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        fen
    }

    pub fn make_move(&mut self, mv: Move) {
        self.make_move_impl(mv, true);
    }

    /// Make a move without any NNUE accumulator bookkeeping, for the test-only
    /// make/unmake reference move generator. MUST be paired with
    /// `unmake_move_no_nnue`.
    #[cfg(test)]
    pub(crate) fn make_move_no_nnue(&mut self, mv: Move) {
        self.make_move_impl(mv, false);
    }

    fn make_move_impl(&mut self, mv: Move, update_nnue: bool) {
        let from  = mv.from();
        let to    = mv.to();
        let flags = mv.flags();
        let us    = self.side;
        let them  = us.flip();

        let irrev = IrrevState {
            captured: self.sq_piece[to as usize].map(|(p, _)| p),
            ep_sq:    self.ep_sq,
            castling: self.castling,
            halfmove: self.halfmove,
            hash:     self.hash,
        };

        let (moving_piece, _) = self.sq_piece[from as usize].unwrap();

        let mut feat: [FeatDelta; 4] = [(Piece::Pawn, Color::White, 0, false); 4];
        let mut nfeat = 0usize;

        feat[nfeat] = (moving_piece, us, from, false); nfeat += 1;

        if moving_piece == Piece::Pawn || mv.is_capture() || mv.is_promotion() {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }

        if let Some(ep) = self.ep_sq {
            self.hash ^= ZOBRIST.ep_file[(ep & 7) as usize];
        }

        self.ep_sq = None;

        self.clear_piece(from);

        match flags {
            MoveFlag::Quiet | MoveFlag::Capture => {
                if let Some(cp) = irrev.captured {
                    feat[nfeat] = (cp, them, to, false); nfeat += 1;
                }
                self.clear_piece(to);
                self.set_piece(to, moving_piece, us);
                feat[nfeat] = (moving_piece, us, to, true); nfeat += 1;
            }
            MoveFlag::DoublePush => {
                self.set_piece(to, moving_piece, us);
                feat[nfeat] = (moving_piece, us, to, true); nfeat += 1;

                let ep = if us == Color::White { to - 8 } else { to + 8 };
                self.ep_sq = Some(ep);
                self.hash ^= ZOBRIST.ep_file[(ep & 7) as usize];
            }
            MoveFlag::CastleKing => {
                self.set_piece(to, Piece::King, us);
                feat[nfeat] = (Piece::King, us, to, true); nfeat += 1;
                let (rook_from, rook_to) = if us == Color::White { (sq::H1, sq::F1) } else { (sq::H8, sq::F8) };
                self.clear_piece(rook_from);
                feat[nfeat] = (Piece::Rook, us, rook_from, false); nfeat += 1;
                self.set_piece(rook_to, Piece::Rook, us);
                feat[nfeat] = (Piece::Rook, us, rook_to, true); nfeat += 1;
            }
            MoveFlag::CastleQueen => {
                self.set_piece(to, Piece::King, us);
                feat[nfeat] = (Piece::King, us, to, true); nfeat += 1;
                let (rook_from, rook_to) = if us == Color::White { (sq::A1, sq::D1) } else { (sq::A8, sq::D8) };
                self.clear_piece(rook_from);
                feat[nfeat] = (Piece::Rook, us, rook_from, false); nfeat += 1;
                self.set_piece(rook_to, Piece::Rook, us);
                feat[nfeat] = (Piece::Rook, us, rook_to, true); nfeat += 1;
            }
            MoveFlag::EnPassant => {
                let cap_sq = if us == Color::White { to - 8 } else { to + 8 };
                feat[nfeat] = (Piece::Pawn, them, cap_sq, false); nfeat += 1;
                self.clear_piece(cap_sq);
                self.set_piece(to, Piece::Pawn, us);
                feat[nfeat] = (Piece::Pawn, us, to, true); nfeat += 1;
            }
            MoveFlag::PromKnight | MoveFlag::PromKnightCapture => {
                if let Some(cp) = irrev.captured { feat[nfeat] = (cp, them, to, false); nfeat += 1; }
                self.clear_piece(to); self.set_piece(to, Piece::Knight, us);
                feat[nfeat] = (Piece::Knight, us, to, true); nfeat += 1;
            }
            MoveFlag::PromBishop | MoveFlag::PromBishopCapture => {
                if let Some(cp) = irrev.captured { feat[nfeat] = (cp, them, to, false); nfeat += 1; }
                self.clear_piece(to); self.set_piece(to, Piece::Bishop, us);
                feat[nfeat] = (Piece::Bishop, us, to, true); nfeat += 1;
            }
            MoveFlag::PromRook | MoveFlag::PromRookCapture => {
                if let Some(cp) = irrev.captured { feat[nfeat] = (cp, them, to, false); nfeat += 1; }
                self.clear_piece(to); self.set_piece(to, Piece::Rook, us);
                feat[nfeat] = (Piece::Rook, us, to, true); nfeat += 1;
            }
            MoveFlag::PromQueen | MoveFlag::PromQueenCapture => {
                if let Some(cp) = irrev.captured { feat[nfeat] = (cp, them, to, false); nfeat += 1; }
                self.clear_piece(to); self.set_piece(to, Piece::Queen, us);
                feat[nfeat] = (Piece::Queen, us, to, true); nfeat += 1;
            }
        }

        if update_nnue && self.net.is_some() && !self.nnue_stack.push(&feat[..nfeat]) {
            self.refresh_nnue();
        }

        let old_castling = self.castling;
        self.castling &= CASTLING_RIGHTS_MASK[from as usize];
        self.castling &= CASTLING_RIGHTS_MASK[to   as usize];

        let changed = old_castling ^ self.castling;
        for i in 0..4u8 {
            if changed & (1 << i) != 0 { self.hash ^= ZOBRIST.castle[i as usize]; }
        }

        self.hash ^= ZOBRIST.side;

        if us == Color::Black { self.fullmove += 1; }

        self.side = them;
        self.history.push((mv, irrev));
    }

    pub fn unmake_move(&mut self) {
        self.unmake_move_impl(true);
    }

    /// Counterpart to `make_move_no_nnue` -- must always be paired with it.
    #[cfg(test)]
    pub(crate) fn unmake_move_no_nnue(&mut self) {
        self.unmake_move_impl(false);
    }

    fn unmake_move_impl(&mut self, update_nnue: bool) {
        let (mv, irrev) = match self.history.pop() {
            Some(x) => x,
            None    => return,
        };

        let from  = mv.from();
        let to    = mv.to();
        let flags = mv.flags();

        self.side = self.side.flip();
        let us    = self.side;

        if us == Color::Black { self.fullmove -= 1; }

        self.ep_sq    = irrev.ep_sq;
        self.castling = irrev.castling;
        self.halfmove = irrev.halfmove;

        let (moved_piece, _) = self.sq_piece[to as usize].unwrap();

        match flags {
            MoveFlag::Quiet | MoveFlag::Capture | MoveFlag::DoublePush => {
                self.clear_piece(to);

                if let Some(cap) = irrev.captured {
                    self.set_piece(to, cap, us.flip());
                }
                let restore = if flags == MoveFlag::DoublePush { Piece::Pawn } else { moved_piece };
                self.set_piece(from, restore, us);
            }
            MoveFlag::CastleKing => {
                self.clear_piece(to);
                self.set_piece(from, Piece::King, us);
                let (rook_from, rook_to) = if us == Color::White { (sq::H1, sq::F1) } else { (sq::H8, sq::F8) };
                self.clear_piece(rook_to);
                self.set_piece(rook_from, Piece::Rook, us);
            }
            MoveFlag::CastleQueen => {
                self.clear_piece(to);
                self.set_piece(from, Piece::King, us);
                let (rook_from, rook_to) = if us == Color::White { (sq::A1, sq::D1) } else { (sq::A8, sq::D8) };
                self.clear_piece(rook_to);
                self.set_piece(rook_from, Piece::Rook, us);
            }
            MoveFlag::EnPassant => {
                self.clear_piece(to);
                self.set_piece(from, Piece::Pawn, us);
                let cap_sq = if us == Color::White { to - 8 } else { to + 8 };
                self.set_piece(cap_sq, Piece::Pawn, us.flip());
            }

            MoveFlag::PromKnight | MoveFlag::PromBishop | MoveFlag::PromRook | MoveFlag::PromQueen => {
                self.clear_piece(to);
                self.set_piece(from, Piece::Pawn, us);
            }
            MoveFlag::PromKnightCapture | MoveFlag::PromBishopCapture
            | MoveFlag::PromRookCapture | MoveFlag::PromQueenCapture => {
                self.clear_piece(to);
                if let Some(cap) = irrev.captured {
                    self.set_piece(to, cap, us.flip());
                }
                self.set_piece(from, Piece::Pawn, us);
            }
        }

        self.hash = irrev.hash;

        if update_nnue && self.net.is_some() && !self.nnue_stack.pop() {
            self.refresh_nnue();
        }
    }

    /// Is `sq` attacked by `attacker` color, given the current occupancy?
    #[inline]
    pub fn is_attacked(&self, sq: u8, attacker: Color) -> bool {
        self.is_attacked_occ(sq, attacker, self.occupied())
    }

    /// Bitboard of every `by`-colored piece that attacks `sq` under the
    /// supplied occupancy. `occ` is passed explicitly so callers can probe a
    /// hypothetical position (e.g. king removed, for king-move legality, or
    /// two pawns removed, for the en-passant discovered-check test) without
    /// mutating the board.
    pub fn attackers_to(&self, sq: u8, by: Color, occ: Bitboard) -> Bitboard {
        let c = by as usize;
        let target = bit(sq);

        let pawn_from = if by == Color::White {
            ((target >> 7) & NOT_FILE_A) | ((target >> 9) & NOT_FILE_H)
        } else {
            ((target << 7) & NOT_FILE_H) | ((target << 9) & NOT_FILE_A)
        };

        (pawn_from & self.pieces[c][Piece::Pawn as usize])
            | (KNIGHT_ATTACKS[sq as usize] & self.pieces[c][Piece::Knight as usize])
            | (KING_ATTACKS[sq as usize] & self.pieces[c][Piece::King as usize])
            | (MAGICS.bishop_attacks(sq, occ)
                & (self.pieces[c][Piece::Bishop as usize] | self.pieces[c][Piece::Queen as usize]))
            | (MAGICS.rook_attacks(sq, occ)
                & (self.pieces[c][Piece::Rook as usize] | self.pieces[c][Piece::Queen as usize]))
    }

    /// Is `sq` attacked by `attacker` color under the supplied occupancy?
    pub fn is_attacked_occ(&self, sq: u8, attacker: Color, occ: Bitboard) -> bool {
        let a = attacker as usize;

        let pawns = self.pieces[a][Piece::Pawn as usize];
        let pawn_attacks = if attacker == Color::White {
            ((pawns << 9) & NOT_FILE_A) | ((pawns << 7) & NOT_FILE_H)
        } else {
            ((pawns >> 7) & NOT_FILE_A) | ((pawns >> 9) & NOT_FILE_H)
        };
        if pawn_attacks & bit(sq) != 0 { return true; }

        if KNIGHT_ATTACKS[sq as usize] & self.pieces[a][Piece::Knight as usize] != 0 { return true; }

        if KING_ATTACKS[sq as usize] & self.pieces[a][Piece::King as usize] != 0 { return true; }

        let diag_attackers = self.pieces[a][Piece::Bishop as usize]
            | self.pieces[a][Piece::Queen as usize];
        if MAGICS.bishop_attacks(sq, occ) & diag_attackers != 0 { return true; }

        let orth_attackers = self.pieces[a][Piece::Rook as usize]
            | self.pieces[a][Piece::Queen as usize];
        if MAGICS.rook_attacks(sq, occ) & orth_attackers != 0 { return true; }

        false
    }

    pub fn king_sq(&self, color: Color) -> u8 {
        lsb(self.pieces[color as usize][Piece::King as usize])
    }

    /// Every square attacked by any piece of `color`, given the current
    /// occupancy. Used by the search for threat-aware move ordering /
    /// history (a quiet move that leaves an attacked square, or moves into
    /// one, is scored differently). This is a cheap union of attack sets,
    /// not a per-square `is_attacked` loop.
    pub fn attacks_by(&self, color: Color) -> Bitboard {
        let c   = color as usize;
        let occ = self.occupied();

        let pawns = self.pieces[c][Piece::Pawn as usize];
        let mut atk = if color == Color::White {
            ((pawns << 9) & NOT_FILE_A) | ((pawns << 7) & NOT_FILE_H)
        } else {
            ((pawns >> 7) & NOT_FILE_A) | ((pawns >> 9) & NOT_FILE_H)
        };

        let mut knights = self.pieces[c][Piece::Knight as usize];
        while knights != 0 {
            atk |= KNIGHT_ATTACKS[pop_lsb(&mut knights) as usize];
        }

        atk |= KING_ATTACKS[self.king_sq(color) as usize];

        let mut diag = self.pieces[c][Piece::Bishop as usize] | self.pieces[c][Piece::Queen as usize];
        while diag != 0 {
            atk |= MAGICS.bishop_attacks(pop_lsb(&mut diag), occ);
        }

        let mut orth = self.pieces[c][Piece::Rook as usize] | self.pieces[c][Piece::Queen as usize];
        while orth != 0 {
            atk |= MAGICS.rook_attacks(pop_lsb(&mut orth), occ);
        }

        atk
    }

    pub fn in_check(&self) -> bool {
        self.is_attacked(self.king_sq(self.side), self.side.flip())
    }

    pub fn print(&self) {
        println!("+---+---+---+---+---+---+---+---+");
        for rank in (0..8).rev() {
            print!("|");
            for file in 0..8 {
                let sq = rank * 8 + file;
                let s = match self.sq_piece[sq] {
                    Some((p, c)) => format!(" {} ", p.to_char(c)),
                    None         => "   ".to_string(),
                };
                print!("{}|", s);
            }
            println!(" {}", rank + 1);
            println!("+---+---+---+---+---+---+---+---+");
        }
        println!("  a   b   c   d   e   f   g   h");
        println!("Side: {:?}  EP: {:?}  Castle: {:04b}",
            self.side, self.ep_sq, self.castling);
    }

    pub fn make_null_move(&mut self) {

        let irrev = IrrevState {
            captured: None,
            ep_sq:    self.ep_sq,
            castling: self.castling,
            halfmove: self.halfmove,
            hash:     self.hash,
        };

        if let Some(ep) = self.ep_sq {
            self.hash ^= ZOBRIST.ep_file[(ep & 7) as usize];
        }
        self.ep_sq = None;

        self.hash ^= ZOBRIST.side;
        self.side = self.side.flip();
        self.halfmove += 1;
        if self.side == Color::White { self.fullmove += 1; }

        self.history.push((Move::NULL, irrev));
    }

    pub fn unmake_null_move(&mut self) {
        let (_, irrev) = match self.history.pop() {
            Some(x) => x,
            None    => return,
        };
        self.side     = self.side.flip();
        self.ep_sq    = irrev.ep_sq;
        self.castling = irrev.castling;
        self.halfmove = irrev.halfmove;
        self.hash     = irrev.hash;
        if self.side == Color::Black { self.fullmove -= 1; }
    }

    /// Apply a sequence of moves from the given position (used by UCI position command)
    pub fn apply_moves(&mut self, move_strs: &[&str]) {
        for ms in move_strs {
            if let Some(mv) = self.find_uci_move(ms) {
                self.make_move(mv);
            }
        }
    }

    /// Find the legal move matching a UCI string
    pub fn find_uci_move(&self, uci: &str) -> Option<Move> {
        use crate::movegen::MoveGen;
        let parsed = Move::from_uci(uci)?;
        let mut moves = crate::moves::MoveList::new();
        MoveGen::generate_all(self, &mut moves);

        moves.iter().copied().find(|mv| {
            mv.from() == parsed.from()
                && mv.to() == parsed.to()
                && mv.promo_piece() == parsed.promo_piece()
        })
    }
}

/// Castling rights mask per square – if a king/rook moves from these squares,
/// the corresponding right is revoked.
const CASTLING_RIGHTS_MASK: [u8; 64] = {
    let mut m = [0xFFu8; 64];
    m[sq::E1 as usize] &= !(CASTLE_WK | CASTLE_WQ);
    m[sq::A1 as usize] &= !CASTLE_WQ;
    m[sq::H1 as usize] &= !CASTLE_WK;
    m[sq::E8 as usize] &= !(CASTLE_BK | CASTLE_BQ);
    m[sq::A8 as usize] &= !CASTLE_BQ;
    m[sq::H8 as usize] &= !CASTLE_BK;
    m
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Every under-promotion string must produce that piece (it used to
    /// resolve to the queen promotion, desyncing the board from the GUI).
    #[test]
    fn uci_underpromotions_resolve_to_the_named_piece() {
        crate::nnue::load_embedded();
        for (s, piece) in [("n", Piece::Knight), ("b", Piece::Bishop), ("r", Piece::Rook), ("q", Piece::Queen)] {

            for uci in [format!("e7e8{s}"), format!("e7d8{s}")] {
                let mut b = Board::from_fen("3r3k/4P3/8/8/8/8/8/K7 w - - 0 1").unwrap();
                b.apply_moves(&[uci.as_str()]);
                assert_eq!(b.sq_piece[bit_index(&uci[2..4])], Some((piece, Color::White)), "{uci}");
                assert_eq!(b.side, Color::Black, "{uci} was not applied");
            }
        }
    }

    /// Regression: the game where the engine then played an illegal move.
    #[test]
    fn knight_underpromotion_from_a_real_game() {
        crate::nnue::load_embedded();
        let mut b = Board::from_fen("rnbqkbnr/pppp1pp1/4p3/7p/4P3/5P2/PPPPK1PP/RNBQ1BNR w kq - 0 4").unwrap();
        let moves = "e2e1 c7c5 d2d4 d7d5 e4d5 e6d5 f1b5 c8d7 b5d7 d8d7 b1c3 c5d4 d1d4 g8e7 g1e2 \
            b8c6 d4d3 e8c8 c1f4 d5d4 c3e4 e7g6 a2a3 g6f4 e2f4 d7c7 f4e2 c6e5 d3b3 d4d3 f3f4 d3e2 \
            f4e5 d8e8 e4g5 f8c5 g5f3 g7g5 f3g5 c7e5 g5f3 e5e7 b3c4 c8b8 c4f4 b8a8 g2g3 h8g8 f4d2 \
            c5b6 d2d5 e7e3 a1d1 e3f2 e1d2 e2e1n d2c1";
        let list: Vec<&str> = moves.split_whitespace().collect();
        b.apply_moves(&list);
        assert_eq!(b.history.len(), list.len(), "every move must apply");
        assert_eq!(b.sq_piece[sq::E1 as usize], Some((Piece::Knight, Color::Black)));
    }

    fn bit_index(s: &str) -> usize {
        str_to_sq(s).unwrap() as usize
    }
}

#[cfg(test)]
mod hash_tests {
    use super::*;

    /// After make + unmake the hash must be exactly what it was, and after a
    /// make it must equal the hash of the same position built from its FEN.
    /// (unmake used to restore the hash before moving the pieces back, which
    /// XOR'd piece keys into it and left it corrupted.)
    #[test]
    fn hash_survives_make_unmake() {
        crate::nnue::load_embedded();
        for fen in [
            "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        ] {
            let mut b = Board::from_fen(fen).unwrap();
            let root = b.hash;
            let mut moves = crate::moves::MoveList::new();
            crate::movegen::MoveGen::generate_all(&b, &mut moves);
            for &mv in moves.iter() {
                b.make_move(mv);
                let want = Board::from_fen(&b.to_fen()).unwrap().hash;
                assert_eq!(b.hash, want, "{fen}: hash after {} differs from its FEN's", mv.to_uci());
                b.unmake_move();
                assert_eq!(b.hash, root, "{fen}: hash not restored after unmaking {}", mv.to_uci());
            }
        }
    }
}
