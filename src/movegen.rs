use crate::bitboard::*;
use crate::board::{Board, Color, Piece, CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ};
use crate::moves::{Move, MoveFlag, MoveList};

pub struct MoveGen;

impl MoveGen {
    /// All legal moves from the current position, appended to `list`.
    ///
    /// Legality is decided arithmetically -- no make/unmake per move. We
    /// compute the king's checkers and the set of pinned pieces once, then:
    ///   * king moves are kept when the destination is unattacked with the
    ///     king lifted out of the blocker set;
    ///   * under double check, only king moves survive;
    ///   * under single check, a non-king move must land on the checker or
    ///     between it and the king (en passant that captures the checker is
    ///     handled in the en-passant branch);
    ///   * a pinned piece may only move along its pin ray;
    ///   * en passant is validated by recomputing king safety with both
    ///     pawns removed (covers the horizontal discovered-check case).
    pub fn generate_all(board: &Board, list: &mut MoveList) {
        Self::generate_pseudo_into(board, list);
        Self::retain_legal(board, list);
    }

    /// Legal captures and promotions only (quiescence search). The caller
    /// only invokes this when NOT in check; the legal filter still handles
    /// the general case.
    pub fn generate_captures_legal(board: &Board, list: &mut MoveList) {
        Self::generate_captures_pseudo_into(board, list);
        Self::retain_legal(board, list);
    }

    fn retain_legal(board: &Board, list: &mut MoveList) {
        let ctx = LegalCtx::new(board);
        let n = list.len();
        let mut w = 0;
        for r in 0..n {
            let mv = list[r];
            if ctx.is_legal(board, mv) {
                list.swap(w, r);
                w += 1;
            }
        }
        list.truncate(w);
    }

    fn generate_pseudo_into(board: &Board, moves: &mut MoveList) {
        let us    = board.side;
        let them  = us.flip();
        let occ   = board.occupied();
        let ours  = board.color_bb[us as usize];
        let theirs= board.color_bb[them as usize];

        Self::gen_pawns(board, us, occ, theirs, moves);
        Self::gen_knights(board, us, ours, theirs, moves);
        Self::gen_bishops(board, us, occ, ours, theirs, moves);
        Self::gen_rooks(board, us, occ, ours, theirs, moves);
        Self::gen_queens(board, us, occ, ours, theirs, moves);
        Self::gen_king(board, us, ours, theirs, moves);
        Self::gen_castles(board, us, occ, moves);
    }

    fn generate_captures_pseudo_into(board: &Board, moves: &mut MoveList) {
        let us    = board.side;
        let them  = us.flip();
        let occ   = board.occupied();
        let theirs= board.color_bb[them as usize];

        Self::gen_pawn_captures(board, us, occ, theirs, moves);
        Self::gen_knight_captures(board, us, theirs, moves);
        Self::gen_bishop_captures(board, us, occ, theirs, moves);
        Self::gen_rook_captures(board, us, occ, theirs, moves);
        Self::gen_queen_captures(board, us, occ, theirs, moves);
        Self::gen_king_captures(board, us, theirs, moves);
    }

    fn gen_pawn_captures(board: &Board, us: Color, occ: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let pawns = board.piece_bb(us, Piece::Pawn);

        if us == Color::White {
            let cap_e = (pawns << 9) & NOT_FILE_A & theirs;
            let cap_w = (pawns << 7) & NOT_FILE_H & theirs;
            Self::add_pawn_moves(cap_e & !RANK_8, 9, MoveFlag::Capture, false, moves);
            Self::add_pawn_moves(cap_w & !RANK_8, 7, MoveFlag::Capture, false, moves);
            Self::add_promotions(((pawns << 8) & !occ) & RANK_8, 8, false, false, moves);
            Self::add_promotions(cap_e & RANK_8, 9, true, false, moves);
            Self::add_promotions(cap_w & RANK_8, 7, true, false, moves);
        } else {
            let cap_e = (pawns >> 7) & NOT_FILE_A & theirs;
            let cap_w = (pawns >> 9) & NOT_FILE_H & theirs;
            Self::add_pawn_moves(cap_e & !RANK_1, 7, MoveFlag::Capture, true, moves);
            Self::add_pawn_moves(cap_w & !RANK_1, 9, MoveFlag::Capture, true, moves);
            Self::add_promotions(((pawns >> 8) & !occ) & RANK_1, 8, false, true, moves);
            Self::add_promotions(cap_e & RANK_1, 7, true, true, moves);
            Self::add_promotions(cap_w & RANK_1, 9, true, true, moves);
        }

    }

    fn gen_pawns(board: &Board, us: Color, occ: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let pawns = board.piece_bb(us, Piece::Pawn);
        let empty = !occ;

        if us == Color::White {
            let push1 = (pawns << 8) & empty;
            let push2 = ((push1 & RANK_3) << 8) & empty;
            let cap_e = (pawns << 9) & NOT_FILE_A & theirs;
            let cap_w = (pawns << 7) & NOT_FILE_H & theirs;

            Self::add_pawn_moves(push1 & !RANK_8, 8,  MoveFlag::Quiet,      false, moves);
            Self::add_pawn_moves(push2,            16, MoveFlag::DoublePush, false, moves);
            Self::add_pawn_moves(cap_e & !RANK_8,  9,  MoveFlag::Capture,   false, moves);
            Self::add_pawn_moves(cap_w & !RANK_8,  7,  MoveFlag::Capture,   false, moves);
            Self::add_promotions(push1 & RANK_8,   8,  false, false, moves);
            Self::add_promotions(cap_e & RANK_8,   9,  true,  false, moves);
            Self::add_promotions(cap_w & RANK_8,   7,  true,  false, moves);
        } else {
            let push1 = (pawns >> 8) & empty;
            let push2 = ((push1 & RANK_6) >> 8) & empty;
            let cap_e = (pawns >> 7) & NOT_FILE_A & theirs;
            let cap_w = (pawns >> 9) & NOT_FILE_H & theirs;

            Self::add_pawn_moves(push1 & !RANK_1, 8,  MoveFlag::Quiet,      true, moves);
            Self::add_pawn_moves(push2,            16, MoveFlag::DoublePush, true, moves);
            Self::add_pawn_moves(cap_e & !RANK_1,  7,  MoveFlag::Capture,   true, moves);
            Self::add_pawn_moves(cap_w & !RANK_1,  9,  MoveFlag::Capture,   true, moves);
            Self::add_promotions(push1 & RANK_1,   8,  false, true,  moves);
            Self::add_promotions(cap_e & RANK_1,   7,  true,  true,  moves);
            Self::add_promotions(cap_w & RANK_1,   9,  true,  true,  moves);
        }
        Self::gen_en_passant(board, us, moves);
    }

    fn gen_en_passant(board: &Board, us: Color, moves: &mut MoveList) {
        let Some(ep) = board.ep_sq else { return };
        let pawns = board.piece_bb(us, Piece::Pawn);
        let ep_bb = bit(ep);
        if us == Color::White {
            if (pawns << 9) & NOT_FILE_A & ep_bb != 0 {
                moves.push(Move::new(ep - 9, ep, MoveFlag::EnPassant));
            }
            if (pawns << 7) & NOT_FILE_H & ep_bb != 0 {
                moves.push(Move::new(ep - 7, ep, MoveFlag::EnPassant));
            }
        } else {
            if (pawns >> 7) & NOT_FILE_A & ep_bb != 0 {
                moves.push(Move::new(ep + 7, ep, MoveFlag::EnPassant));
            }
            if (pawns >> 9) & NOT_FILE_H & ep_bb != 0 {
                moves.push(Move::new(ep + 9, ep, MoveFlag::EnPassant));
            }
        }
    }

    fn add_pawn_moves(mut targets: Bitboard, delta: u8, flag: MoveFlag, black: bool, moves: &mut MoveList) {
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            let from = if black { to + delta } else { to - delta };
            moves.push(Move::new(from, to, flag));
        }
    }

    fn add_promotions(mut targets: Bitboard, delta: u8, capture: bool, black: bool, moves: &mut MoveList) {
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            let from = if black { to + delta } else { to - delta };
            let (qf, rf, bf, nf) = if capture {
                (MoveFlag::PromQueenCapture, MoveFlag::PromRookCapture,
                 MoveFlag::PromBishopCapture, MoveFlag::PromKnightCapture)
            } else {
                (MoveFlag::PromQueen, MoveFlag::PromRook,
                 MoveFlag::PromBishop, MoveFlag::PromKnight)
            };
            moves.push(Move::new(from, to, qf));
            moves.push(Move::new(from, to, rf));
            moves.push(Move::new(from, to, bf));
            moves.push(Move::new(from, to, nf));
        }
    }

    fn gen_knight_captures(board: &Board, us: Color, theirs: Bitboard, moves: &mut MoveList) {
        let mut knights = board.piece_bb(us, Piece::Knight);
        while knights != 0 {
            let from = pop_lsb(&mut knights);
            let mut attacks = KNIGHT_ATTACKS[from as usize] & theirs;
            while attacks != 0 {
                let to = pop_lsb(&mut attacks);
                moves.push(Move::new(from, to, MoveFlag::Capture));
            }
        }
    }

    fn gen_knights(board: &Board, us: Color, ours: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let mut knights = board.piece_bb(us, Piece::Knight);
        while knights != 0 {
            let from = pop_lsb(&mut knights);
            let mut attacks = KNIGHT_ATTACKS[from as usize] & !ours;
            while attacks != 0 {
                let to = pop_lsb(&mut attacks);
                let flag = if bit(to) & theirs != 0 { MoveFlag::Capture } else { MoveFlag::Quiet };
                moves.push(Move::new(from, to, flag));
            }
        }
    }

    fn gen_bishop_captures(board: &Board, us: Color, occ: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let mut bishops = board.piece_bb(us, Piece::Bishop);
        while bishops != 0 {
            let from = pop_lsb(&mut bishops);
            let attacks = MAGICS.bishop_attacks(from, occ) & theirs;
            Self::add_sliding(from, attacks, theirs, moves);
        }
    }

    fn gen_bishops(board: &Board, us: Color, occ: Bitboard, ours: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let mut bishops = board.piece_bb(us, Piece::Bishop);
        while bishops != 0 {
            let from = pop_lsb(&mut bishops);
            let attacks = MAGICS.bishop_attacks(from, occ) & !ours;
            Self::add_sliding(from, attacks, theirs, moves);
        }
    }

    fn gen_rook_captures(board: &Board, us: Color, occ: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let mut rooks = board.piece_bb(us, Piece::Rook);
        while rooks != 0 {
            let from = pop_lsb(&mut rooks);
            let attacks = MAGICS.rook_attacks(from, occ) & theirs;
            Self::add_sliding(from, attacks, theirs, moves);
        }
    }

    fn gen_rooks(board: &Board, us: Color, occ: Bitboard, ours: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let mut rooks = board.piece_bb(us, Piece::Rook);
        while rooks != 0 {
            let from = pop_lsb(&mut rooks);
            let attacks = MAGICS.rook_attacks(from, occ) & !ours;
            Self::add_sliding(from, attacks, theirs, moves);
        }
    }

    fn gen_queen_captures(board: &Board, us: Color, occ: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let mut queens = board.piece_bb(us, Piece::Queen);
        while queens != 0 {
            let from = pop_lsb(&mut queens);
            let attacks = MAGICS.queen_attacks(from, occ) & theirs;
            Self::add_sliding(from, attacks, theirs, moves);
        }
    }

    fn gen_queens(board: &Board, us: Color, occ: Bitboard, ours: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let mut queens = board.piece_bb(us, Piece::Queen);
        while queens != 0 {
            let from = pop_lsb(&mut queens);
            let attacks = MAGICS.queen_attacks(from, occ) & !ours;
            Self::add_sliding(from, attacks, theirs, moves);
        }
    }

    fn add_sliding(from: u8, mut attacks: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        while attacks != 0 {
            let to = pop_lsb(&mut attacks);
            let flag = if bit(to) & theirs != 0 { MoveFlag::Capture } else { MoveFlag::Quiet };
            moves.push(Move::new(from, to, flag));
        }
    }

    fn gen_king_captures(board: &Board, us: Color, theirs: Bitboard, moves: &mut MoveList) {
        let from = board.king_sq(us);
        if from < 64 {
            let mut attacks = KING_ATTACKS[from as usize] & theirs;
            while attacks != 0 {
                let to = pop_lsb(&mut attacks);
                moves.push(Move::new(from, to, MoveFlag::Capture));
            }
        }
    }

    fn gen_king(board: &Board, us: Color, ours: Bitboard, theirs: Bitboard, moves: &mut MoveList) {
        let from = board.king_sq(us);
        if from >= 64 { return; }
        let mut attacks = KING_ATTACKS[from as usize] & !ours;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks);
            let flag = if bit(to) & theirs != 0 { MoveFlag::Capture } else { MoveFlag::Quiet };
            moves.push(Move::new(from, to, flag));
        }
    }

    fn gen_castles(board: &Board, us: Color, occ: Bitboard, moves: &mut MoveList) {
        let (king_sq, kside_bit, qside_bit,
             k_empty, q_empty,
             k_safe1, k_safe2,
             q_safe1, q_dest,
             rook_k, rook_q) = if us == Color::White {
            (sq::E1, CASTLE_WK, CASTLE_WQ,
             bit(sq::F1) | bit(sq::G1),
             bit(sq::B1) | bit(sq::C1) | bit(sq::D1),
             sq::F1, sq::G1,
             sq::D1, sq::C1,
             sq::H1, sq::A1)
        } else {
            (sq::E8, CASTLE_BK, CASTLE_BQ,
             bit(sq::F8) | bit(sq::G8),
             bit(sq::B8) | bit(sq::C8) | bit(sq::D8),
             sq::F8, sq::G8,
             sq::D8, sq::C8,
             sq::H8, sq::A8)
        };

        let them = us.flip();

        if board.castling & kside_bit != 0
            && occ & k_empty == 0
            && !board.is_attacked(king_sq, them)
            && !board.is_attacked(k_safe1, them)
            && !board.is_attacked(k_safe2, them)
            && board.sq_piece[rook_k as usize] == Some((Piece::Rook, us))
        {
            moves.push(Move::new(king_sq, k_safe2, MoveFlag::CastleKing));
        }

        if board.castling & qside_bit != 0
            && occ & q_empty == 0
            && !board.is_attacked(king_sq, them)
            && !board.is_attacked(q_safe1, them)
            && !board.is_attacked(q_dest,  them)
            && board.sq_piece[rook_q as usize] == Some((Piece::Rook, us))
        {
            moves.push(Move::new(king_sq, q_dest, MoveFlag::CastleQueen));
        }
    }

    pub fn perft(board: &mut Board, depth: u32) -> u64 {
        let mut moves = MoveList::new();
        Self::generate_all(board, &mut moves);
        if depth <= 1 { return moves.len() as u64; }
        let mut nodes = 0u64;
        for i in 0..moves.len() {
            let mv = moves[i];
            board.make_move(mv);
            nodes += Self::perft(board, depth - 1);
            board.unmake_move();
        }
        nodes
    }

    pub fn perft_divide(board: &mut Board, depth: u32) -> Vec<(String, u64)> {
        let mut moves = MoveList::new();
        Self::generate_all(board, &mut moves);
        let mut result = Vec::new();
        for i in 0..moves.len() {
            let mv = moves[i];
            board.make_move(mv);
            let nodes = if depth <= 1 { 1 } else { Self::perft(board, depth - 1) };
            board.unmake_move();
            result.push((mv.to_uci(), nodes));
        }
        result
    }
}

struct LegalCtx {
    us:           Color,
    ksq:          u8,
    occ:          Bitboard,
    num_checkers: u32,
    /// checker square | squares between it and the king (single-check only)
    check_block:  Bitboard,
    /// our pieces pinned to the king
    pinned:       Bitboard,
}

impl LegalCtx {
    fn new(board: &Board) -> Self {
        let us   = board.side;
        let them = us.flip();
        let ksq  = board.king_sq(us);
        let occ  = board.occupied();

        let checkers     = board.attackers_to(ksq, them, occ);
        let num_checkers = checkers.count_ones();

        let check_block = if num_checkers == 1 {
            let csq = lsb(checkers);
            checkers | between(ksq, csq)
        } else {
            !0u64
        };

        let tc = them as usize;
        let rq = board.pieces[tc][Piece::Rook as usize] | board.pieces[tc][Piece::Queen as usize];
        let bq = board.pieces[tc][Piece::Bishop as usize] | board.pieces[tc][Piece::Queen as usize];
        let mut snipers = (MAGICS.rook_attacks(ksq, 0) & rq)
                        | (MAGICS.bishop_attacks(ksq, 0) & bq);
        let own = board.color_bb[us as usize];
        let mut pinned = 0u64;
        while snipers != 0 {
            let s = pop_lsb(&mut snipers);
            let b = between(ksq, s) & occ;
            if b.count_ones() == 1 && (b & own) != 0 {
                pinned |= b;
            }
        }

        LegalCtx { us, ksq, occ, num_checkers, check_block, pinned }
    }

    fn is_legal(&self, board: &Board, mv: Move) -> bool {
        let from  = mv.from();
        let to    = mv.to();
        let flags = mv.flags();
        let them  = self.us.flip();

        if matches!(flags, MoveFlag::CastleKing | MoveFlag::CastleQueen) {
            return true;
        }

        if from == self.ksq {
            let occ_after = (self.occ ^ bit(from)) | bit(to);
            return board.attackers_to(to, them, occ_after) == 0;
        }

        if self.num_checkers >= 2 {
            return false;
        }

        if flags == MoveFlag::EnPassant {
            let cap_sq = if self.us == Color::White { to - 8 } else { to + 8 };
            let occ_after = (self.occ ^ bit(from) ^ bit(cap_sq)) | bit(to);
            let tc = them as usize;
            let rq = board.pieces[tc][Piece::Rook as usize] | board.pieces[tc][Piece::Queen as usize];
            let bq = board.pieces[tc][Piece::Bishop as usize] | board.pieces[tc][Piece::Queen as usize];
            let ktgt = bit(self.ksq);
            let pawn_from = if them == Color::White {
                ((ktgt >> 7) & NOT_FILE_A) | ((ktgt >> 9) & NOT_FILE_H)
            } else {
                ((ktgt << 7) & NOT_FILE_H) | ((ktgt << 9) & NOT_FILE_A)
            };
            let attacked =
                   MAGICS.rook_attacks(self.ksq, occ_after) & rq != 0
                || MAGICS.bishop_attacks(self.ksq, occ_after) & bq != 0
                || KNIGHT_ATTACKS[self.ksq as usize] & board.pieces[tc][Piece::Knight as usize] != 0
                || KING_ATTACKS[self.ksq as usize] & board.pieces[tc][Piece::King as usize] != 0
                || pawn_from & board.pieces[tc][Piece::Pawn as usize] & !bit(cap_sq) != 0;
            return !attacked;
        }

        if self.num_checkers == 1 && (bit(to) & self.check_block) == 0 {
            return false;
        }

        if (bit(from) & self.pinned) != 0 && (bit(to) & line(self.ksq, from)) == 0 {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    /// Reference generator: pseudo-legal moves filtered by actually making
    /// each one and checking the king is not left in check. Slow but obviously
    /// correct -- used to cross-check `generate_all`.
    fn legal_slow(board: &mut Board) -> Vec<Move> {
        let mut pseudo = MoveList::new();
        MoveGen::generate_pseudo_into(board, &mut pseudo);
        let mut out = Vec::new();
        for i in 0..pseudo.len() {
            let mv = pseudo[i];
            board.make_move_no_nnue(mv);
            let legal = !board.is_attacked(board.king_sq(board.side.flip()), board.side);
            board.unmake_move_no_nnue();
            if legal { out.push(mv); }
        }
        out
    }

    fn sorted_uci(mut v: Vec<Move>) -> Vec<String> {
        let mut s: Vec<String> = v.drain(..).map(|m| m.to_uci()).collect();
        s.sort();
        s
    }

    fn assert_matches(fen: &str) {
        let mut b = Board::from_fen(fen).expect("fen");
        let fast = {
            let mut l = MoveList::new();
            MoveGen::generate_all(&b, &mut l);
            (0..l.len()).map(|i| l[i]).collect::<Vec<_>>()
        };
        let slow = legal_slow(&mut b);
        assert_eq!(sorted_uci(fast), sorted_uci(slow), "mismatch on {fen}");
    }

    /// Order-sensitive: `generate_all` must emit exactly the same moves in
    /// the same order as "pseudo order, then drop the illegal ones", so the
    /// search's move ordering is deterministic. Walked recursively so it
    /// covers checks, pins and en passant deep in the tree.
    fn assert_order_matches(board: &mut Board, depth: u32) {
        let fast = {
            let mut l = MoveList::new();
            MoveGen::generate_all(board, &mut l);
            (0..l.len()).map(|i| l[i].to_uci()).collect::<Vec<_>>()
        };
        let slow = legal_slow(board).iter().map(|m| m.to_uci()).collect::<Vec<_>>();
        assert_eq!(fast, slow, "order mismatch at {}", board.to_fen());
        if depth == 0 { return; }
        let mut l = MoveList::new();
        MoveGen::generate_all(board, &mut l);
        for i in 0..l.len() {
            let mv = l[i];
            board.make_move(mv);
            assert_order_matches(board, depth - 1);
            board.unmake_move();
        }
    }

    #[test]
    fn order_matches_reference_deep() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ] {
            let mut b = Board::from_fen(fen).unwrap();
            assert_order_matches(&mut b, 3);
        }
    }

    #[test]
    fn legal_matches_reference() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",

            "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",

            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",

            "4k3/8/8/8/8/8/3n4/4K1r1 w - - 0 1",
        ] {
            assert_matches(fen);
        }
    }

    fn perft(b: &mut Board, d: u32) -> u64 { MoveGen::perft(b, d) }

    #[test]
    fn perft_startpos() {
        let mut b = Board::start_pos();
        assert_eq!(perft(&mut b, 4), 197281);
    }

    #[test]
    fn perft_kiwipete() {
        let mut b = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
        assert_eq!(perft(&mut b, 3), 97862);
    }

    #[test]
    fn perft_pos3() {
        let mut b = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap();
        assert_eq!(perft(&mut b, 5), 674624);
    }

    #[test]
    fn perft_pos4() {
        let mut b = Board::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1").unwrap();
        assert_eq!(perft(&mut b, 4), 422333);
    }

    #[test]
    fn perft_pos5() {
        let mut b = Board::from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8").unwrap();
        assert_eq!(perft(&mut b, 3), 62379);
    }

    #[test]
    fn perft_pos6() {
        let mut b = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10").unwrap();
        assert_eq!(perft(&mut b, 4), 3894594);
    }
}
