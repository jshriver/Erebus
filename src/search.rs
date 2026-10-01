//! search: negamax alpha-beta with the heuristic set ported from `chal.c`.
//!
//! Structure and constants follow chal's `search` / `qsearch` closely:
//!   * principal-variation search (null-window scout + re-search)
//!   * transposition table cutoffs + TT-move / TT-eval-bound use
//!   * internal iterative reductions
//!   * reverse futility pruning, razoring, adaptive null-move pruning
//!   * per-move: history pruning, late-move pruning, futility pruning,
//!     SEE pruning, late-move reductions
//!   * threat-aware history: history is indexed
//!     `[stm][from][to][from-attacked?][to-attacked?]`, with a gravity
//!     update and a malus applied to the quiets that did *not* cause the
//!     cutoff
//!   * mate-distance pruning, material-draw detection
//!   * aspiration windows with widening + a complexity-scaled soft limit
//!
//!   * counter-move heuristic; history and counter-moves persist across the
//!     moves of a game (`SearchTables`), killers reset per search
//!
//! Engine-specific on top of that: Lazy-SMP `thread_id` staggering, the
//! Syzygy WDL/DTZ probe, ponder / `ponderhit` time handling, and the
//! triangular PV table.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::board::{Board, Color, Piece};
use crate::eval::{evaluate, DRAW_SCORE, INF, MATE_SCORE};
use crate::movegen::MoveGen;
use crate::moves::{Move, MoveList};
use crate::see::see;
use crate::syzygy::{tb_wdl_score, SyzygyTb, WdlResult, TB_WIN_SCORE};
use crate::tt::{Bound, TranspositionTable, EVAL_NONE};

const MAX_PLY: usize = 128;

const SCORE_TT_MOVE:      i32 = 200_000;
const SCORE_PROMO_BASE:   i32 = 60_000;
const SCORE_CAPTURE_BASE: i32 = 40_000;
const SCORE_KILLER_1:     i32 = 25_000;
const SCORE_KILLER_2:     i32 = 20_000;
const SCORE_COUNTER:      i32 = 18_000;

const MAX_HISTORY: i32 = 16_384;
const MAX_BONUS:   i32 = 2_000;

const CORR_SIZE:  usize = 16_384;
const CORR_GRAIN: i32 = 256;
const CORR_SCALE: i32 = 256;
const CORR_MAX:   i32 = CORR_GRAIN * 64;

/// `pawn_corr[stm][pawn-structure index]`
type CorrTable = [[i32; CORR_SIZE]; 2];

/// Index of the position's pawn structure in a `CorrTable` row.
#[inline]
fn pawn_corr_index(board: &Board) -> usize {
    let w = board.pieces[Color::White as usize][Piece::Pawn as usize];
    let b = board.pieces[Color::Black as usize][Piece::Pawn as usize];
    let h = w.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(29);
    (h.wrapping_mul(0xFF51_AFD7_ED55_8CCD) >> 50) as usize
}

pub const SEE_PRUNE_DEPTH_DEFAULT: i32 = 3;
pub const SEE_PRUNE_DEPTH_MAX: i32 = 20;
pub static SEE_PRUNE_DEPTH: AtomicI32 = AtomicI32::new(SEE_PRUNE_DEPTH_DEFAULT);

pub const SEE_NOISY_MARGIN_DEFAULT: i32 = 120;
pub const SEE_QUIET_MARGIN_DEFAULT: i32 = 60;
/// A margin this large never prunes: a quiet move can lose at most a queen.
pub const SEE_MARGIN_MAX: i32 = 1000;
pub static SEE_NOISY_MARGIN: AtomicI32 = AtomicI32::new(SEE_NOISY_MARGIN_DEFAULT);
pub static SEE_QUIET_MARGIN: AtomicI32 = AtomicI32::new(SEE_QUIET_MARGIN_DEFAULT);

const RFP_MAX_DEPTH: u32 = 6;
const RFP_MARGIN:    i32 = 75;

const NMP_MIN_DEPTH:   u32 = 3;
const NMP_EVAL_MARGIN: i32 = 30;

const FP_MAX_DEPTH: u32 = 3;
const FP_MARGIN:    i32 = 90;

const LMP_MAX_DEPTH: u32 = 5;

const HP_MAX_DEPTH: u32 = 4;
const HP_MARGIN:    i32 = 2_048;

const RAZORING_MAX_DEPTH: u32 = 3;
const RAZORING_MARGIN:    i32 = 150;

const ASP_MIN_DEPTH: u32 = 5;
const ASP_START_DELTA: i32 = 12;

const MVV_LVA: [[i32; 6]; 6] = [
    [15, 14, 13, 12, 11, 10],
    [25, 24, 23, 22, 21, 20],
    [35, 34, 33, 32, 31, 30],
    [45, 44, 43, 42, 41, 40],
    [55, 54, 53, 52, 51, 50],
    [0, 0, 0, 0, 0, 0],
];

fn build_lmr_table() -> [[i32; 64]; 64] {
    let mut t = [[0i32; 64]; 64];
    for (d, row) in t.iter_mut().enumerate().skip(1) {
        for (m, r) in row.iter_mut().enumerate().skip(1) {
            let v = (d as f64).ln() * (m as f64).ln() / 2.1350 + 0.2319;
            *r = v.max(0.0) as i32;
        }
    }
    t
}

#[inline]
fn history_bonus(depth: u32) -> i32 {
    ((depth * depth) as i32).min(MAX_BONUS)
}

pub struct SearchLimits {
    pub max_depth:        u32,
    pub move_time:        Option<Duration>,
    pub soft_time:        Option<Duration>,
    pub nodes:            Option<u64>,
    /// Stop after the iterative-deepening iteration during which the node
    /// count reached this (datagen: whole iterations, so the score and move
    /// agree). `nodes` stays the hard cap.
    pub soft_nodes:       Option<u64>,
    pub infinite:         bool,
    /// `go ponder`: search infinitely until `ponderhit` (then switch to the
    /// `ponder_*` clock) or `stop`.
    pub ponder:           bool,
    pub ponder_move_time: Option<Duration>,
    pub ponder_soft_time: Option<Duration>,
    /// When the GUI's clock started for this search (the `position` / `go`
    /// line arriving). Time limits are measured from here, so command
    /// parsing, thread spawns and board clones count against the budget.
    /// `None` = now, at `Searcher::new`.
    pub start:            Option<Instant>,
    /// No `info` output (datagen).
    pub quiet:            bool,
}

impl Default for SearchLimits {
    fn default() -> Self {
        SearchLimits {
            max_depth:        64,
            move_time:        None,
            soft_time:        None,
            nodes:            None,
            soft_nodes:       None,
            infinite:         false,
            ponder:           false,
            ponder_move_time: None,
            ponder_soft_time: None,
            start:            None,
            quiet:            false,
        }
    }
}

struct PvTable {
    table:  [[Move; MAX_PLY]; MAX_PLY],
    length: [usize; MAX_PLY],
}

impl PvTable {
    fn new() -> Self {
        PvTable {
            table:  [[Move::NULL; MAX_PLY]; MAX_PLY],
            length: [0; MAX_PLY],
        }
    }

    #[inline]
    fn update(&mut self, ply: usize, mv: Move) {
        self.table[ply][0] = mv;
        let child_len = self.length[ply + 1];
        for i in 0..child_len {
            self.table[ply][i + 1] = self.table[ply + 1][i];
        }
        self.length[ply] = child_len + 1;
    }

    #[inline]
    fn clear(&mut self, ply: usize) { self.length[ply] = 0; }

    fn root_pv_string(&self) -> String {
        self.table[0][..self.length[0]]
            .iter()
            .map(|m| m.to_uci())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The move the opponent is expected to play -- the second move of the
    /// root PV. `Move::NULL` when the search never got a PV that deep.
    fn ponder_move(&self) -> Move {
        if self.length[0] >= 2 { self.table[0][1] } else { Move::NULL }
    }
}

/// `history[stm][from][to][from_attacked][to_attacked]`
type HistoryTable = [[[[[i32; 2]; 2]; 64]; 64]; 2];

/// `counter_moves[prev_from][prev_to]` — the quiet move that most recently
/// produced a beta cutoff as a reply to the move `prev_from -> prev_to`.
type CounterTable = [[Move; 64]; 64];

/// Move-ordering tables that persist across the moves of a game (see
/// `uci::SmpState::tables_pool`). Handed to a `Searcher` at construction and
/// handed back via `Searcher::into_tables` when the search finishes, so
/// history/counter-move knowledge learned on one move seeds the next instead
/// of starting from zero every `go`.
pub struct SearchTables {
    history:       Box<HistoryTable>,
    counter_moves: Box<CounterTable>,
    pawn_corr:     Box<CorrTable>,
}

impl Default for SearchTables {
    fn default() -> Self {
        SearchTables {
            history:       Box::new([[[[[0i32; 2]; 2]; 64]; 64]; 2]),
            counter_moves: Box::new([[Move::NULL; 64]; 64]),
            pawn_corr:     Box::new([[0; CORR_SIZE]; 2]),
        }
    }
}

/// Decay every history counter toward zero (`*= 3/4`). Run once at the start
/// of each search so stale counts from earlier moves fade rather than
/// dominating this search's ordering.
fn decay_history(h: &mut HistoryTable) {
    for by_from in h.iter_mut() {
        for by_to in by_from.iter_mut() {
            for by_srcth in by_to.iter_mut() {
                for by_dstth in by_srcth.iter_mut() {
                    for v in by_dstth.iter_mut() {
                        *v -= *v / 4;
                    }
                }
            }
        }
    }
}

pub struct Searcher {
    pub nodes:     u64,
    /// Portion of `nodes` already folded into `nodes_global`.
    nodes_flushed: u64,
    pub start:     Instant,
    /// Base for the time limits. Equals `start` except after a ponderhit,
    /// where it restarts so our clock is counted from the opponent's move
    /// (`start` keeps measuring the whole search for the info line / nps).
    clock:         Instant,
    pub limits:    SearchLimits,
    pub best_move: Move,
    /// Predicted opponent reply, captured at the end of the last fully
    /// completed iterative-deepening iteration (see `search`). Survives a
    /// clock-interrupted final iteration; read via `ponder_move()`.
    saved_ponder:  Move,
    /// The root move `saved_ponder` is a reply to.
    pv_root:       Move,
    /// Score of the last completed iteration (side-to-move relative).
    last_score:    i32,
    pub thread_id: usize,
    /// `board.history.len()` at the root of the current search: entries at
    /// or below this index were played in the game, above it in the tree.
    root_hist_len: usize,

    stopped: bool,

    pv:            PvTable,
    killers:       [[Move; 2]; MAX_PLY],
    history:       Box<HistoryTable>,
    counter_moves: Box<CounterTable>,
    pawn_corr:     Box<CorrTable>,
    lmr:           [[i32; 64]; 64],
    see_prune_depth:  u32,
    see_noisy_margin: i32,
    see_quiet_margin: i32,

    tt:            Arc<TranspositionTable>,
    tb:            Arc<Option<SyzygyTb>>,
    /// Successful Syzygy probes, shared across all Lazy-SMP threads so the
    /// `tbhits` field of the info line is a true whole-search total rather
    /// than just the main thread's slice. Bumped once per position that
    /// `probe_wdl` actually resolves.
    tbhits:        Arc<AtomicU64>,
    /// Node count summed across every Lazy-SMP thread. Each thread folds its
    /// local `nodes` delta in at every timing checkpoint (and once when it
    /// finishes) so the info line's `nodes` / `nps` reflect the whole
    /// search's work rather than just the main thread's slice.
    nodes_global:  Arc<AtomicU64>,
    stop:          Arc<AtomicBool>,
    ponderhit:     Arc<AtomicBool>,
}

impl Searcher {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        limits:        SearchLimits,
        tt:            Arc<TranspositionTable>,
        tb:            Arc<Option<SyzygyTb>>,
        tbhits:        Arc<AtomicU64>,
        nodes_global:  Arc<AtomicU64>,
        stop:          Arc<AtomicBool>,
        ponderhit:     Arc<AtomicBool>,
        tables:        SearchTables,
        thread_id:     usize,
    ) -> Self {
        let t0 = limits.start.unwrap_or_else(Instant::now);
        Searcher {
            nodes:     0,
            nodes_flushed: 0,
            start:     t0,
            clock:     t0,
            limits,
            best_move: Move::NULL,
            saved_ponder: Move::NULL,
            pv_root:      Move::NULL,
            last_score:   0,
            thread_id,
            root_hist_len: 0,
            stopped:   false,
            pv:            PvTable::new(),
            killers:       [[Move::NULL; 2]; MAX_PLY],
            history:       tables.history,
            counter_moves: tables.counter_moves,
            pawn_corr:     tables.pawn_corr,
            lmr:           build_lmr_table(),
            see_prune_depth:  SEE_PRUNE_DEPTH.load(Ordering::Relaxed) as u32,
            see_noisy_margin: SEE_NOISY_MARGIN.load(Ordering::Relaxed),
            see_quiet_margin: SEE_QUIET_MARGIN.load(Ordering::Relaxed),
            tt,
            tb,
            tbhits,
            nodes_global,
            stop,
            ponderhit,
        }
    }

    /// Reclaim the persistent move-ordering tables so the caller can hand
    /// them to the next search on this thread (see `uci::SmpState`).
    pub fn into_tables(self) -> SearchTables {
        SearchTables {
            history:       self.history,
            counter_moves: self.counter_moves,
            pawn_corr:     self.pawn_corr,
        }
    }

    /// The predicted opponent reply, for the `bestmove <best> ponder <this>`
    /// line and the autonomous-ponder position. Taken from the last completed
    /// ID iteration, so a clock-cut final iteration doesn't blank it.
    /// `Move::NULL` when no iteration ever produced a 2-ply PV.
    pub fn ponder_move(&self) -> Move {
        self.saved_ponder
    }

    /// Score of the last completed iteration of the most recent `search`,
    /// from the side to move's point of view.
    pub fn score(&self) -> i32 {
        self.last_score
    }

    /// Fold this thread's un-counted node delta into the shared total. Cheap
    /// between checkpoints (a subtract and a branch); the atomic add only
    /// fires at the ~1-2k-node cadence of the callers.
    #[inline]
    fn flush_nodes(&mut self) {
        let delta = self.nodes - self.nodes_flushed;
        if delta != 0 {
            self.nodes_global.fetch_add(delta, Ordering::Relaxed);
            self.nodes_flushed = self.nodes;
        }
    }

    /// Hard stop: the search must abandon whatever it is doing. Latches:
    /// once true it stays true for the rest of this `search` call.
    fn time_up(&mut self) -> bool {
        if self.stopped { return true; }
        self.check_ponderhit();
        let hit = self.stop.load(Ordering::Relaxed)
            || self.limits.move_time.is_some_and(|mt| self.clock.elapsed() >= mt)
            || self.limits.nodes.is_some_and(|n| self.nodes >= n);
        if hit { self.stopped = true; }
        hit
    }

    /// Ponderhit transition: an infinite `go ponder` becomes a normal timed
    /// search the moment the opponent plays our ponder move. Checked at every
    /// node-cadence checkpoint (via `time_up`) so the hard limit applies
    /// mid-iteration, not only once the current iteration finishes. The
    /// budget is measured from the ponderhit (`clock`), not from the `go`.
    fn check_ponderhit(&mut self) {
        if !(self.limits.ponder && self.limits.infinite) { return; }
        if !self.ponderhit.load(Ordering::Relaxed) { return; }
        self.limits.infinite  = false;
        self.limits.move_time = self.limits.ponder_move_time
            .or(Some(Duration::from_millis(1000)));
        self.limits.soft_time = self.limits.ponder_soft_time
            .or(Some(Duration::from_millis(500)));
        self.clock = Instant::now();
    }

    /// Soft stop, checked only between completed iterative-deepening
    /// iterations on the main thread. `score` is the score just returned by
    /// that iteration and `board` the root position, used to widen the
    /// budget on a volatile position (search score far from static eval).
    fn soft_time_up(&mut self, board: &mut Board, score: i32) -> bool {
        self.check_ponderhit();

        if self.limits.infinite { return false; }
        let Some(soft) = self.limits.soft_time else { return false; };

        let mut budget = soft;
        let complexity = (score - evaluate(board)).abs().min(200) as f64;
        let factor = 0.8 + 0.4 * (complexity / 200.0);
        budget = budget.mul_f64(factor);

        self.clock.elapsed() >= budget
    }

    pub fn search(&mut self, board: &mut Board) -> Move {
        self.best_move = Move::NULL;
        self.stopped = false;
        self.last_score = 0;
        self.root_hist_len = board.history.len();

        board.refresh_nnue();

        decay_history(&mut self.history);

        let mut root_moves = MoveList::new();
        MoveGen::generate_all(board, &mut root_moves);
        let mut best = if root_moves.is_empty() { Move::NULL } else { root_moves[0] };
        self.best_move = best;

        let start_depth = if self.thread_id == 0 {
            1
        } else {
            1 + (self.thread_id % 3) as u32
        };

        let mut prev_score = 0i32;

        for depth in start_depth..=self.limits.max_depth.min(MAX_PLY as u32 - 2) {
            let score = self.aspiration(board, depth, prev_score);

            if self.time_up() {
                if !self.best_move.is_null() { best = self.best_move; }
                break;
            }

            best = self.best_move;
            prev_score = score;
            self.last_score = score;

            let pm = self.pv.ponder_move();
            if !pm.is_null() {
                self.saved_ponder = pm;
                self.pv_root = best;
            }

            if self.thread_id == 0 && !self.limits.quiet {
                self.flush_nodes();
                let total_nodes = self.nodes_global.load(Ordering::Relaxed);
                let elapsed_ms = self.start.elapsed().as_millis();
                let nps = (total_nodes as u128 * 1000).checked_div(elapsed_ms).unwrap_or(0);
                println!(
                    "info depth {} score {} nodes {} nps {} hashfull {} tbhits {} time {} pv {}",
                    depth,
                    score_str(score),
                    total_nodes,
                    nps,
                    self.tt.hashfull(),
                    self.tbhits.load(Ordering::Relaxed),
                    elapsed_ms,
                    self.pv.root_pv_string(),
                );
            }

            if score.abs() >= MATE_SCORE - MAX_PLY as i32 { break; }
            if self.limits.soft_nodes.is_some_and(|n| self.nodes >= n) { break; }
            if depth >= 4 && self.soft_time_up(board, score) { break; }
        }

        if self.pv_root != best {
            self.saved_ponder = Move::NULL;
        }

        if self.thread_id == 0 {
            loop {
                self.check_ponderhit();
                if !self.limits.infinite || self.stop.load(Ordering::Relaxed) { break; }
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        self.flush_nodes();
        best
    }

    /// One iterative-deepening iteration: plain full-window search for
    /// shallow depths, aspiration window with widening beyond that.
    fn aspiration(&mut self, board: &mut Board, depth: u32, prev_score: i32) -> i32 {
        if depth < ASP_MIN_DEPTH {
            return self.negamax(board, depth, -INF, INF, 0, false);
        }

        let mut delta = ASP_START_DELTA;
        let mut alpha = (prev_score - delta).max(-INF);
        let mut beta  = (prev_score + delta).min(INF);
        let mut search_depth = depth;

        loop {
            if alpha < -2000 { alpha = -INF; }
            if beta  >  2000 { beta  =  INF; }

            let score = self.negamax(board, search_depth, alpha, beta, 0, false);
            if self.time_up() { return score; }

            if score <= alpha {
                beta  = (alpha + beta) / 2;
                alpha = (alpha - delta).max(-INF);
                search_depth = depth;
            } else if score >= beta {
                beta = (beta + delta).min(INF);
                if search_depth > 1 { search_depth -= 1; }
            } else {
                return score;
            }
            delta += delta / 2;
        }
    }

    fn negamax(
        &mut self,
        board:     &mut Board,
        mut depth: u32,
        mut alpha: i32,
        mut beta:  i32,
        ply:       usize,
        was_null:  bool,
    ) -> i32 {
        if ply < MAX_PLY {
            self.pv.clear(ply);
        }

        if (self.nodes & 2047) == 0 {
            self.flush_nodes();
            if self.time_up() {
                return 0;
            }
        }

        if ply >= MAX_PLY - 1 {
            return evaluate(board);
        }
        if ply > 0 && self.is_draw(board, ply) {
            return DRAW_SCORE;
        }

        if ply > 0 {
            alpha = alpha.max(-(MATE_SCORE - ply as i32));
            beta  = beta.min(MATE_SCORE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        let in_check = board.in_check();
        if in_check && ply > 0 {
            depth += 1;
        }

        let is_pv = beta - alpha > 1;

        let mut tt_move  = Move::NULL;
        let mut tt_score = 0i32;
        let mut tt_bound: Option<Bound> = None;
        let mut tt_eval  = EVAL_NONE;
        if let Some(e) = self.tt.probe(board.hash) {
            tt_move  = e.mv;
            tt_eval  = e.eval;
            tt_score = TranspositionTable::score_from_tt(e.score, ply);
            tt_bound = Some(e.bound);
            if e.depth as u32 >= depth && !is_pv && ply > 0 {
                let usable = match e.bound {
                    Bound::Exact      => true,
                    Bound::Lower => tt_score >= beta,
                    Bound::Upper => tt_score <= alpha,
                };
                if usable {
                    return tt_score;
                }
            }
        }

        if depth >= 4 && tt_move.is_null() && !in_check {
            depth -= 1;
        }
        if depth == 0 {
            return self.quiescence(board, alpha, beta, ply);
        }

        self.nodes += 1;

        let raw_eval = if in_check {
            EVAL_NONE
        } else if tt_eval != EVAL_NONE {
            tt_eval
        } else {
            evaluate(board)
        };
        let mut static_eval = if in_check { -INF } else { self.corrected_eval(board, raw_eval) };

        let corr_eval = static_eval;

        if !in_check && let Some(b) = tt_bound {
            match b {
                Bound::Lower | Bound::Exact if tt_score > static_eval => static_eval = tt_score,
                Bound::Upper | Bound::Exact if tt_score < static_eval => static_eval = tt_score,
                _ => {}
            }
        }

        if !in_check
            && let Some(tb) = self.tb.as_ref()
            && let Some(wdl) = tb.probe_wdl(board)
        {
            self.tbhits.fetch_add(1, Ordering::Relaxed);

            let decisive = matches!(wdl, WdlResult::Win | WdlResult::Loss);
            let dtz = if decisive { tb.probe_dtz(board) } else { None };
            let score = tb_wdl_score(wdl, dtz, ply);

            let bound = match wdl {
                WdlResult::Win  => Bound::Lower,
                WdlResult::Loss => Bound::Upper,
                _               => Bound::Exact,
            };
            self.tt.store(
                board.hash,
                Move::NULL,
                TranspositionTable::score_to_tt(score, ply),
                raw_eval,
                depth as u8,
                bound,
            );
            if !is_pv || !decisive {
                return score;
            }
            if score >= beta { return score; }
            if score > alpha { alpha = score; }
        }

        let non_mate_beta = beta < MATE_SCORE - MAX_PLY as i32;

        if !is_pv && ply > 0 && !in_check && depth <= RFP_MAX_DEPTH && non_mate_beta {
            let margin = RFP_MARGIN * depth as i32;
            if static_eval - margin >= beta {
                return static_eval - margin;
            }
        }

        if !is_pv
            && ply > 0
            && !in_check
            && depth <= RAZORING_MAX_DEPTH
            && alpha > -(MATE_SCORE - MAX_PLY as i32)
            && static_eval + RAZORING_MARGIN * depth as i32 <= alpha
        {
            let razor = self.quiescence(board, alpha, beta, ply);
            if self.time_up() { return 0; }
            if razor <= alpha {
                return razor;
            }
        }

        if !is_pv
            && ply > 0
            && !was_null
            && !in_check
            && depth >= NMP_MIN_DEPTH
            && non_mate_beta
            && self.has_non_pawn_material(board)
            && static_eval >= beta + NMP_EVAL_MARGIN
        {
            let r = (3 + depth / 4 + ((static_eval - beta) / 200) as u32).min(6);
            board.make_null_move();
            let null_score = -self.negamax(
                board,
                depth.saturating_sub(1 + r),
                -beta,
                -beta + 1,
                ply + 1,
                true,
            );
            board.unmake_null_move();
            if self.time_up() { return 0; }
            if null_score >= beta {
                return if null_score >= MATE_SCORE - MAX_PLY as i32 { beta } else { null_score };
            }
        }

        let mut moves = MoveList::new();
        MoveGen::generate_all(board, &mut moves);
        if moves.is_empty() {
            return if in_check { -(MATE_SCORE - ply as i32) } else { DRAW_SCORE };
        }

        let prev_mv = board.history.last().map(|&(m, _)| m).unwrap_or(Move::NULL);
        let counter = if prev_mv.is_null() {
            Move::NULL
        } else {
            self.counter_moves[prev_mv.from() as usize][prev_mv.to() as usize]
        };

        let opp_threats = board.attacks_by(board.side.flip());
        let mut scores = [0i32; MoveList::CAP];
        for i in 0..moves.len() {
            let mv = moves[i];
            scores[i] = if mv == tt_move {
                SCORE_TT_MOVE
            } else {
                self.score_move(board, mv, ply, opp_threats, counter)
            };
        }

        let stm = board.side as usize;
        let original_alpha = alpha;
        let mut best_move  = Move::NULL;
        let mut best_score  = -INF;
        let mut bound       = Bound::Upper;
        let mut searched    = 0u32;
        let mut quiets = MoveList::new();

        for i in 0..moves.len() {

            let mut best_j = i;
            for j in (i + 1)..moves.len() {
                if scores[j] > scores[best_j] {
                    best_j = j;
                }
            }
            moves.swap(i, best_j);
            scores.swap(i, best_j);

            let mv = moves[i];
            let from = mv.from() as usize;
            let to   = mv.to() as usize;
            let src_th = ((opp_threats >> from) & 1) as usize;
            let dst_th = ((opp_threats >> to) & 1) as usize;
            let is_noisy = mv.is_capture() || mv.is_promotion();
            let is_killer = ply < MAX_PLY
                && (mv == self.killers[ply][0] || mv == self.killers[ply][1]);
            let hist = self.history[stm][from][to][src_th][dst_th];

            let can_prune = !is_pv && !in_check && best_score > -INF;

            if can_prune
                && depth <= HP_MAX_DEPTH
                && !is_noisy
                && !is_killer
                && hist < -HP_MARGIN * (depth as i32 - 1)
            {
                continue;
            }

            if can_prune
                && depth <= LMP_MAX_DEPTH
                && !is_noisy
                && searched >= 3 + depth * depth
            {
                continue;
            }

            if can_prune
                && depth <= FP_MAX_DEPTH
                && !is_noisy
                && static_eval + FP_MARGIN * depth as i32 <= alpha
            {
                continue;
            }

            if !in_check && depth <= self.see_prune_depth && best_score > -INF {
                let margin = if is_noisy {
                    -self.see_noisy_margin * depth as i32
                } else {
                    -self.see_quiet_margin * depth as i32
                };
                if !self.see_ge(board, mv, margin) {
                    continue;
                }
            }

            board.make_move(mv);
            searched += 1;
            if !is_noisy {
                quiets.push(mv);
            }

            let gives_check = board.in_check();

            let mut score;
            if depth > 1 && searched > 1 && !(is_pv && is_noisy) {
                let d_idx = (depth as usize).min(63);
                let m_idx = (searched as usize).min(63);
                let mut r = self.lmr[d_idx][m_idx] - hist / 2048;
                if !is_pv { r += 2; }
                if is_killer { r -= 2; }
                if gives_check { r -= 1; }
                if !is_noisy && src_th == 1 && dst_th == 0 { r -= 1; }
                r = r.clamp(1, depth as i32 - 1);

                score = -self.negamax(
                    board,
                    depth - 1 - r as u32,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    false,
                );

                if score > alpha {
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, false);
                }
            } else if !is_pv || searched > 1 {
                score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, false);
            } else {
                score = alpha;
            }

            if is_pv && (searched == 1 || (score > alpha && score < beta)) {
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, false);
            }

            board.unmake_move();

            if self.time_up() {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move  = mv;

                if ply == 0 && score > alpha {
                    self.best_move = mv;
                    if ply < MAX_PLY - 1 {
                        self.pv.update(ply, mv);
                    }
                }
            }

            if score >= beta {
                if !is_noisy {
                    if ply < MAX_PLY && self.killers[ply][0] != mv {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = mv;
                    }
                    if !prev_mv.is_null() {
                        self.counter_moves[prev_mv.from() as usize][prev_mv.to() as usize] = mv;
                    }
                    let bonus = history_bonus(depth);
                    self.update_history(stm, from, to, src_th, dst_th, bonus);

                    for &qm in quiets.iter().take(quiets.len().saturating_sub(1)) {
                        let qf = qm.from() as usize;
                        let qt = qm.to() as usize;
                        let qs = ((opp_threats >> qf) & 1) as usize;
                        let qd = ((opp_threats >> qt) & 1) as usize;
                        self.update_history(stm, qf, qt, qs, qd, -bonus);
                    }
                }
                if !in_check {
                    self.update_corr(board, depth, raw_eval, corr_eval, score, Bound::Lower, mv);
                }
                self.tt.store(
                    board.hash,
                    mv,
                    TranspositionTable::score_to_tt(score, ply),
                    raw_eval,
                    depth as u8,
                    Bound::Lower,
                );
                return score;
            }

            if score > alpha {
                alpha = score;
                bound = Bound::Exact;
                if is_pv && ply < MAX_PLY - 1 {
                    self.pv.update(ply, mv);
                }
            }
        }

        if best_score == -INF {

            best_score = alpha;
        }

        let store_bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > original_alpha {
            Bound::Exact
        } else {
            bound
        };
        if !in_check {
            self.update_corr(board, depth, raw_eval, corr_eval, best_score, store_bound, best_move);
        }
        self.tt.store(
            board.hash,
            best_move,
            TranspositionTable::score_to_tt(best_score, ply),
            raw_eval,
            depth as u8,
            store_bound,
        );

        best_score
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if ply < MAX_PLY {
            self.pv.clear(ply);
        }

        if (self.nodes & 1023) == 0 {
            self.flush_nodes();
            if self.time_up() {
                return 0;
            }
        }
        if ply > 0 && self.is_draw(board, ply) {
            return DRAW_SCORE;
        }

        let in_check = board.in_check();
        if ply >= MAX_PLY - 1 {
            return if in_check { DRAW_SCORE } else { evaluate(board) };
        }

        let is_pv = beta - alpha > 1;

        let mut tt_move = Move::NULL;
        let mut tt_eval = EVAL_NONE;
        if let Some(e) = self.tt.probe(board.hash) {
            tt_move = e.mv;
            tt_eval = e.eval;
            if !is_pv && ply > 0 {
                let tt_score = TranspositionTable::score_from_tt(e.score, ply);
                let usable = match e.bound {
                    Bound::Exact      => true,
                    Bound::Lower => tt_score >= beta,
                    Bound::Upper => tt_score <= alpha,
                };
                if usable {
                    return tt_score;
                }
            }
        }

        let old_alpha = alpha;
        let mut best_score = -INF;

        let raw_eval = if in_check {
            EVAL_NONE
        } else if tt_eval != EVAL_NONE {
            tt_eval
        } else {
            evaluate(board)
        };
        if !in_check {
            let stand_pat = self.corrected_eval(board, raw_eval);
            if stand_pat >= beta {
                return stand_pat;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
            best_score = stand_pat;
        }

        let mut moves = MoveList::new();
        if in_check {
            MoveGen::generate_all(board, &mut moves);
        } else {
            MoveGen::generate_captures_legal(board, &mut moves);
        }

        if in_check && moves.is_empty() {
            return -(MATE_SCORE - ply as i32);
        }

        let mut scores = [0i32; MoveList::CAP];
        for i in 0..moves.len() {
            let mv = moves[i];
            scores[i] = if mv == tt_move {
                SCORE_TT_MOVE
            } else {
                self.score_move(board, mv, ply, 0, Move::NULL)
            };
        }

        let mut best_move = Move::NULL;

        for i in 0..moves.len() {
            let mut best_j = i;
            for j in (i + 1)..moves.len() {
                if scores[j] > scores[best_j] {
                    best_j = j;
                }
            }
            moves.swap(i, best_j);
            scores.swap(i, best_j);
            let mv = moves[i];

            if !in_check && !self.see_ge(board, mv, 0) {
                continue;
            }

            board.make_move(mv);
            let score = -self.quiescence(board, -beta, -alpha, ply + 1);
            board.unmake_move();

            if self.time_up() {
                return 0;
            }
            if score > best_score {
                best_score = score;
                best_move = mv;
            }
            if score >= beta {
                self.tt.store(
                    board.hash,
                    mv,
                    TranspositionTable::score_to_tt(score, ply),
                    raw_eval,
                    0,
                    Bound::Lower,
                );
                return score;
            }
            if score > alpha {
                alpha = score;
                if is_pv && ply < MAX_PLY - 1 {
                    self.pv.update(ply, mv);
                }
            }
        }

        if !self.stopped {
            let b = if best_score <= old_alpha { Bound::Upper } else { Bound::Exact };
            self.tt.store(
                board.hash,
                best_move,
                TranspositionTable::score_to_tt(best_score, ply),
                raw_eval,
                0,
                b,
            );
        }
        best_score
    }

    fn score_move(&self, board: &Board, mv: Move, ply: usize, opp_threats: u64, counter: Move) -> i32 {
        let from = mv.from() as usize;
        let to   = mv.to() as usize;

        if mv.is_promotion() {

            let rank = match mv.promo_piece() {
                Some(Piece::Queen)  => 4,
                Some(Piece::Rook)   => 3,
                Some(Piece::Bishop) => 2,
                _                   => 1,
            };
            return SCORE_PROMO_BASE + rank * 100;
        }

        if mv.is_capture() {
            let victim = board.sq_piece[to].map_or(Piece::Pawn as usize, |(p, _)| p as usize);
            let attacker = board.sq_piece[from].map_or(0, |(p, _)| p as usize);
            return SCORE_CAPTURE_BASE + MVV_LVA[victim][attacker];
        }

        if ply < MAX_PLY {
            if mv == self.killers[ply][0] { return SCORE_KILLER_1; }
            if mv == self.killers[ply][1] { return SCORE_KILLER_2; }
        }

        if !counter.is_null() && mv == counter {
            return SCORE_COUNTER;
        }

        let stm = board.side as usize;
        let src_th = ((opp_threats >> from) & 1) as usize;
        let dst_th = ((opp_threats >> to) & 1) as usize;
        self.history[stm][from][to][src_th][dst_th]
    }

    /// Static eval plus the learned pawn-structure correction.
    #[inline]
    fn corrected_eval(&self, board: &Board, raw_eval: i32) -> i32 {
        raw_eval + self.pawn_corr[board.side as usize][pawn_corr_index(board)] / CORR_GRAIN
    }

    /// Fold a finished node's result into the correction history. Skipped
    /// when the result says nothing about the eval's error: a noisy best
    /// move (the score reflects material won, not a misjudged position), a
    /// fail-high that doesn't beat the eval, a fail-low that doesn't fall
    /// below it, and mate / TB scores.
    #[allow(clippy::too_many_arguments)]
    fn update_corr(
        &mut self,
        board:     &Board,
        depth:     u32,
        raw_eval:  i32,
        corr_eval: i32,
        best:      i32,
        bound:     Bound,
        best_move: Move,
    ) {
        if self.stopped
            || best_move.is_capture()
            || best_move.is_promotion()
            || best.abs() >= TB_WIN_SCORE - 1000
            || (bound == Bound::Lower && best <= corr_eval)
            || (bound == Bound::Upper && best >= corr_eval)
        {
            return;
        }
        let target = (best - raw_eval).clamp(-1024, 1024) * CORR_GRAIN;
        let w = (depth as i32 + 1).min(16);
        let e = &mut self.pawn_corr[board.side as usize][pawn_corr_index(board)];
        *e = ((*e * (CORR_SCALE - w) + target * w) / CORR_SCALE).clamp(-CORR_MAX, CORR_MAX);
    }

    fn update_history(&mut self, stm: usize, from: usize, to: usize, s: usize, d: usize, bonus: i32) {
        let clamped = bonus.clamp(-MAX_BONUS, MAX_BONUS);
        let cur = self.history[stm][from][to][s][d];
        self.history[stm][from][to][s][d] =
            cur + clamped - (cur * clamped.abs()) / MAX_HISTORY;
    }

    fn see_ge(&self, board: &Board, mv: Move, threshold: i32) -> bool {
        see(board, mv) >= threshold
    }

    fn is_draw(&self, board: &Board, ply: usize) -> bool {
        if board.halfmove >= 100 {
            return true;
        }
        if Self::is_material_draw(board) {
            return true;
        }
        if ply > 0 && self.is_repetition(board) {
            return true;
        }
        false
    }

    /// Insufficient-material / known-drawn minor endings, ported from chal.
    pub(crate) fn is_material_draw(board: &Board) -> bool {
        let p = |c: Color, pc: Piece| board.pieces[c as usize][pc as usize];
        if p(Color::White, Piece::Pawn) | p(Color::Black, Piece::Pawn)
            | p(Color::White, Piece::Rook) | p(Color::Black, Piece::Rook)
            | p(Color::White, Piece::Queen) | p(Color::Black, Piece::Queen)
            != 0
        {
            return false;
        }
        let wn = p(Color::White, Piece::Knight).count_ones();
        let bn = p(Color::Black, Piece::Knight).count_ones();
        let wb = p(Color::White, Piece::Bishop).count_ones();
        let bb = p(Color::Black, Piece::Bishop).count_ones();
        let wm = wn + wb;
        let bm = bn + bb;

        (wm == 0 && bm <= 1)
            || (bm == 0 && wm <= 1)
            || (wm == 1 && bm == 1)
            || (wm == 2 && wn == 2 && bm == 0)
            || (bm == 2 && bn == 2 && wm == 0)
    }

    /// Repetition draw.
    ///
    /// * An earlier occurrence *inside the search tree* (after the root):
    ///   the first recurrence is a draw -- a single repeat already means
    ///   neither side is making progress.
    /// * Earlier occurrences only in the *game* (at or before the root): a
    ///   real 3-fold is needed, i.e. two earlier occurrences. A position
    ///   seen once in the game is not a draw yet; the opponent can deviate.
    ///
    /// Only positions with the same side to move (every 2nd ply) within the
    /// reversible stretch (`halfmove`) can match. The scan stops at a null
    /// move: positions before it were never actually followed by this line.
    /// The root position itself is exempt via the `ply > 0` guard at the
    /// `is_draw` call sites.
    fn is_repetition(&self, board: &Board) -> bool {
        let current = board.hash;
        let hist = &board.history;
        let n = hist.len();
        let lookback = (board.halfmove as usize).min(n);
        let mut game_hits = 0;
        for k in 1..=lookback {
            let i = n - k;
            let (mv, irrev) = &hist[i];
            if mv.is_null() {
                break;
            }
            if k % 2 == 0 && irrev.hash == current {
                if i > self.root_hist_len {
                    return true;
                }
                game_hits += 1;
                if game_hits >= 2 {
                    return true;
                }
            }
        }
        false
    }

    fn has_non_pawn_material(&self, board: &Board) -> bool {
        let us = board.side as usize;
        board.pieces[us][Piece::Knight as usize] != 0
            || board.pieces[us][Piece::Bishop as usize] != 0
            || board.pieces[us][Piece::Rook as usize] != 0
            || board.pieces[us][Piece::Queen as usize] != 0
    }
}

fn score_str(score: i32) -> String {
    if score.abs() >= MATE_SCORE - 1000 {
        let moves_to_mate = (MATE_SCORE - score.abs() + 1) / 2;
        let sign = if score > 0 { "" } else { "-" };
        format!("mate {sign}{moves_to_mate}")
    } else {
        format!("cp {score}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_searcher() -> Searcher {
        Searcher::new(
            SearchLimits { max_depth: 6, ..Default::default() },
            Arc::new(TranspositionTable::new(8)),
            Arc::new(None),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            SearchTables::default(),
            0,
        )
    }

    #[test]
    fn finds_mate_in_one() {
        crate::nnue::load_embedded();
        let mut board = Board::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();

        let mut s = mk_searcher();
        s.limits.max_depth = 4;
        let mv = s.search(&mut board);
        assert_eq!(mv.to_uci(), "a1a8");
    }

    #[test]
    fn wins_hanging_queen() {
        crate::nnue::load_embedded();

        let mut board = Board::from_fen("4k3/8/8/8/3q4/4P3/8/4K3 w - - 0 1").unwrap();
        let mut s = mk_searcher();
        s.limits.max_depth = 6;
        let mv = s.search(&mut board);
        assert_eq!(mv.to_uci(), "e3d4");
    }

    fn after(moves: &str) -> Board {
        let mut b = Board::start_pos();
        b.apply_moves(&moves.split_whitespace().collect::<Vec<_>>());
        b
    }

    const KNIGHT_CYCLE: &str = "g1f3 g8f6 f3g1 f6g8";

    #[test]
    fn single_game_repetition_is_not_a_draw() {
        crate::nnue::load_embedded();

        let b = after(KNIGHT_CYCLE);
        let mut s = mk_searcher();
        s.root_hist_len = b.history.len();
        assert!(!s.is_repetition(&b));
    }

    #[test]
    fn threefold_game_repetition_is_a_draw() {
        crate::nnue::load_embedded();
        let b = after(&format!("{KNIGHT_CYCLE} {KNIGHT_CYCLE}"));
        let mut s = mk_searcher();
        s.root_hist_len = b.history.len();
        assert!(s.is_repetition(&b));
    }

    #[test]
    fn in_tree_repetition_is_a_draw() {
        crate::nnue::load_embedded();

        let b = after(&format!("{KNIGHT_CYCLE} g1f3"));
        let mut s = mk_searcher();
        s.root_hist_len = 0;
        assert!(s.is_repetition(&b));
    }

    #[test]
    fn repetition_scan_stops_at_null_move() {
        crate::nnue::load_embedded();

        let mut b = after("g1f3 g8f6");
        b.make_null_move();
        b.make_null_move();
        let s = mk_searcher();
        assert!(!s.is_repetition(&b));
    }

    #[test]
    fn material_draw_detected() {
        let b = Board::from_fen("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1").unwrap();
        assert!(Searcher::is_material_draw(&b));
        let b2 = Board::from_fen("4k3/8/8/8/8/8/4P3/4K1N1 w - - 0 1").unwrap();
        assert!(!Searcher::is_material_draw(&b2));
    }
}
