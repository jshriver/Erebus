/// A move packed into a u32:
///
///  bits  0- 5  : from square (0-63)
///  bits  6-11  : to square   (0-63)
///  bits 12-15  : flags (see MoveFlag)
#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct Move(pub u32);

impl Move {
    pub const NULL: Move = Move(0);

    #[inline]
    pub fn new(from: u8, to: u8, flags: MoveFlag) -> Self {
        Move((from as u32) | ((to as u32) << 6) | ((flags as u32) << 12))
    }

    #[inline] pub fn from(self) -> u8  { (self.0 & 0x3F) as u8 }
    #[inline] pub fn to(self)   -> u8  { ((self.0 >> 6) & 0x3F) as u8 }
    #[inline] pub fn flags(self) -> MoveFlag {
        MoveFlag::from_u32((self.0 >> 12) & 0xF)
    }
    #[inline] pub fn is_null(self) -> bool { self.0 == 0 }

    pub fn is_capture(self) -> bool {
        matches!(self.flags(),
            MoveFlag::Capture | MoveFlag::EnPassant |
            MoveFlag::PromKnightCapture | MoveFlag::PromBishopCapture |
            MoveFlag::PromRookCapture   | MoveFlag::PromQueenCapture)
    }

    /// Promotion flags are 8..=15, i.e. exactly the ones with flag bit 3
    /// (move bit 15) set.
    pub fn is_promotion(self) -> bool {
        (self.0 >> 15) & 1 == 1
    }

    pub fn promo_piece(self) -> Option<crate::board::Piece> {
        use crate::board::Piece;
        use MoveFlag::*;
        match self.flags() {
            PromKnight | PromKnightCapture => Some(Piece::Knight),
            PromBishop | PromBishopCapture => Some(Piece::Bishop),
            PromRook   | PromRookCapture   => Some(Piece::Rook),
            PromQueen  | PromQueenCapture  => Some(Piece::Queen),
            _ => None,
        }
    }

    pub fn to_uci(self) -> String {
        use crate::bitboard::sq_to_str;
        use MoveFlag::*;
        let promo = match self.flags() {
            PromKnight | PromKnightCapture => "n",
            PromBishop | PromBishopCapture => "b",
            PromRook   | PromRookCapture   => "r",
            PromQueen  | PromQueenCapture  => "q",
            _ => "",
        };
        format!("{}{}{}", sq_to_str(self.from()), sq_to_str(self.to()), promo)
    }

    /// Parse a UCI move string into from/to/promotion only. The flags are
    /// incomplete (no capture / castle / ep info); `Board::find_uci_move`
    /// matches it against the legal move list to get the real move.
    pub fn from_uci(s: &str) -> Option<Move> {
        use crate::bitboard::str_to_sq;
        if s.len() < 4 { return None; }
        let from = str_to_sq(&s[0..2])?;
        let to   = str_to_sq(&s[2..4])?;
        let flag = if s.len() == 5 {
            match s.as_bytes()[4] {
                b'n' => MoveFlag::PromKnight,
                b'b' => MoveFlag::PromBishop,
                b'r' => MoveFlag::PromRook,
                b'q' => MoveFlag::PromQueen,
                _ => return None,
            }
        } else {
            MoveFlag::Quiet
        };
        Some(Move::new(from, to, flag))
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

/// Fixed-capacity, stack-allocated move buffer. 256 is above the 218-move
/// theoretical maximum for a legal chess position.
pub struct MoveList {
    moves: [Move; MoveList::CAP],
    len:   usize,
}

impl MoveList {
    pub const CAP: usize = 256;

    #[inline]
    pub fn new() -> Self {
        MoveList { moves: [Move::NULL; Self::CAP], len: 0 }
    }

    #[inline]
    pub fn push(&mut self, m: Move) {
        // Debug builds catch an overflow; release silently drops (a >256
        // legal-move position does not exist, so this is unreachable).
        debug_assert!(self.len < Self::CAP, "MoveList overflow");
        if self.len < Self::CAP {
            self.moves[self.len] = m;
            self.len += 1;
        }
    }

    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Drop everything past index `n` (used by the legal-move filter to keep
    /// the surviving prefix). Stale slots past `n` are inert.
    #[inline]
    pub fn truncate(&mut self, n: usize) {
        debug_assert!(n <= self.len);
        self.len = n;
    }
    #[inline] pub fn as_slice(&self) -> &[Move] { &self.moves[..self.len] }
    #[inline] pub fn iter(&self) -> std::slice::Iter<'_, Move> { self.as_slice().iter() }

    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        self.moves.swap(a, b);
    }
}

impl Default for MoveList {
    fn default() -> Self { Self::new() }
}

impl std::ops::Index<usize> for MoveList {
    type Output = Move;
    #[inline]
    fn index(&self, i: usize) -> &Move { &self.moves[i] }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum MoveFlag {
    Quiet             = 0,
    DoublePush        = 1,
    CastleKing        = 2,
    CastleQueen       = 3,
    Capture           = 4,
    EnPassant         = 5,
    PromKnight        = 8,
    PromBishop        = 9,
    PromRook          = 10,
    PromQueen         = 11,
    PromKnightCapture = 12,
    PromBishopCapture = 13,
    PromRookCapture   = 14,
    PromQueenCapture  = 15,
}

impl MoveFlag {
    fn from_u32(v: u32) -> Self {
        match v {
            1  => MoveFlag::DoublePush,
            2  => MoveFlag::CastleKing,
            3  => MoveFlag::CastleQueen,
            4  => MoveFlag::Capture,
            5  => MoveFlag::EnPassant,
            8  => MoveFlag::PromKnight,
            9  => MoveFlag::PromBishop,
            10 => MoveFlag::PromRook,
            11 => MoveFlag::PromQueen,
            12 => MoveFlag::PromKnightCapture,
            13 => MoveFlag::PromBishopCapture,
            14 => MoveFlag::PromRookCapture,
            15 => MoveFlag::PromQueenCapture,
            _  => MoveFlag::Quiet,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_FLAGS: [MoveFlag; 14] = [
        MoveFlag::Quiet, MoveFlag::DoublePush, MoveFlag::CastleKing, MoveFlag::CastleQueen,
        MoveFlag::Capture, MoveFlag::EnPassant,
        MoveFlag::PromKnight, MoveFlag::PromBishop, MoveFlag::PromRook, MoveFlag::PromQueen,
        MoveFlag::PromKnightCapture, MoveFlag::PromBishopCapture,
        MoveFlag::PromRookCapture, MoveFlag::PromQueenCapture,
    ];

    #[test]
    fn promotion_and_capture_classification() {
        for f in ALL_FLAGS {
            let mv = Move::new(52, 60, f);
            let n = f as u32;
            assert_eq!(mv.is_promotion(), n >= 8, "is_promotion({f:?})");
            assert_eq!(mv.promo_piece().is_some(), n >= 8, "promo_piece({f:?})");
            assert_eq!(mv.is_capture(), matches!(n, 4 | 5 | 12..=15), "is_capture({f:?})");
        }
    }
}
