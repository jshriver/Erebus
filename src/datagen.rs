//! Self-play training data generation, written as Stockfish binpacks.
//!
//! Every game is played from the start position, so each binpack chain is a
//! whole game: the opening (book moves, then a few random plies) and the
//! searched remainder. Every position is searched, including the opening
//! plies, so each entry carries a real score; the stored move is the one
//! actually played, which is what binpack chain compression requires.
//!
//! Single-threaded fixed-budget search is deterministic -- the same position
//! always yields the same move -- so diversity is injected on purpose:
//!   1. book moves picked at random (weighted, or uniform with `--book-uniform`),
//!   2. `--random-plies` uniformly random legal moves after leaving the book,
//!   3. a jittered node budget per move (`nodes` .. `nodes * 1.5`),
//!   4. a run-wide set of opening end positions: a repeat is rejected.
//!
//! Each worker thread writes its own `<stem>.t<N>.binpack`.

use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sfbinpack::chess::{
    color::Color as SfColor,
    coords::Square as SfSquare,
    piece::Piece as SfPiece,
    piecetype::PieceType as SfPieceType,
    position::Position as SfPosition,
    r#move::{Move as SfMove, MoveType as SfMoveType},
};
use sfbinpack::{CompressedTrainingDataEntryWriter, TrainingDataEntry};

use crate::board::{Board, Color, Piece};
use crate::book::{Book, Rng};
use crate::eval::MATE_SCORE;
use crate::movegen::MoveGen;
use crate::moves::{Move, MoveFlag, MoveList};
use crate::search::{SearchLimits, SearchTables, Searcher};
use crate::syzygy::{SyzygyTb, WdlResult};
use crate::tt::TranspositionTable;

pub struct Config {
    /// Output path; worker `N` writes `<stem>.t<N>.binpack`.
    pub out:          String,
    pub games:        u64,
    pub threads:      usize,
    /// Base soft node budget per move.
    pub nodes:        u64,
    pub book:         Option<String>,
    pub book_uniform: bool,
    pub random_plies: u32,
    pub syzygy:       Option<String>,
    pub seed:         u64,
    /// Transposition table per worker, MB.
    pub hash_mb:      usize,
}

/// |score| at or above this for `WIN_PLIES` plies in a row: decisive.
const WIN_SCORE: i32 = 2500;
const WIN_PLIES: u32 = 4;
/// |score| at or below this for `DRAW_PLIES` plies in a row, from
/// `DRAW_MIN_PLY` on: drawn.
const DRAW_SCORE: i32 = 10;
const DRAW_PLIES: u32 = 8;
const DRAW_MIN_PLY: usize = 80;
/// Hard game length cap (counted as a draw).
const MAX_GAME_PLY: usize = 600;
/// An opening whose search score is further than this from equal is thrown
/// away -- a lopsided start teaches little.
const MAX_OPENING_SCORE: i32 = 1000;
/// Stored scores are clamped to this. Mate / TB-win scores land above the
/// trainer's |score| <= 10000 filter, so those entries are skipped there.
const SCORE_CLAMP: i32 = 32_000;
/// Give up after this many rejected openings in a row (book exhausted).
const MAX_REJECTS: u32 = 10_000;

#[derive(Default)]
struct Stats {
    games:          AtomicU64,
    positions:      AtomicU64,
    /// Every search, including opening balance checks of rejected games.
    searches:       AtomicU64,
    nodes:          AtomicU64,
    white_wins:     AtomicU64,
    draws:          AtomicU64,
    black_wins:     AtomicU64,
    dup_rejects:    AtomicU64,
    unbal_rejects:  AtomicU64,
    /// Post-opening positions sampled for the uniqueness estimate
    /// (1 in `SAMPLE_MOD` by hash), and how many of them were new.
    sampled:        AtomicU64,
    sampled_unique: AtomicU64,
}

const SAMPLE_MOD: u64 = 64;

struct Shared {
    cfg:         Config,
    book:        Option<Book>,
    tb:          Arc<Option<SyzygyTb>>,
    /// Game slots handed out so far (`< cfg.games` means more to play).
    claimed:     AtomicU64,
    /// Zobrist keys of opening end positions already used.
    openings:    Mutex<HashSet<u64>>,
    /// Sampled post-opening position keys (uniqueness measurement).
    seen:        Mutex<HashSet<u64>>,
    stats:       Stats,
}

pub fn run(cfg: Config) {
    let book = match cfg.book.as_deref() {
        Some(path) => match Book::load(path) {
            Ok(b) => {
                eprintln!("datagen: book {path} ({} entries)", b.len());
                Some(b)
            }
            Err(e) => {
                eprintln!("datagen: failed to load book '{path}': {e}");
                return;
            }
        },
        None => None,
    };
    if book.is_none() && cfg.random_plies == 0 {
        eprintln!("datagen: no book and --random-plies 0 -- every game would be identical");
        return;
    }

    let tb = match cfg.syzygy.as_deref() {
        Some(path) => {
            let mut tb = SyzygyTb::new();
            let n: usize = path.split(';').map(|p| p.trim()).filter(|p| !p.is_empty())
                .map(|p| tb.add_directory(p)).sum();
            eprintln!("datagen: {n} Syzygy file(s), up to {} pieces", tb.max_pieces());
            if n > 0 { Some(tb) } else { None }
        }
        None => None,
    };

    let stem = cfg.out.strip_suffix(".binpack").unwrap_or(&cfg.out).to_string();
    let mut writers = Vec::new();
    for t in 0..cfg.threads {
        let path = format!("{stem}.t{t}.binpack");

        let file = match File::create_new(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("datagen: cannot create {path}: {e}");
                return;
            }
        };
        writers.push((path, file));
    }

    eprintln!(
        "datagen: {} games, {} threads, {}-{} nodes/move, random plies {}, book {}, seed {}",
        cfg.games, cfg.threads, cfg.nodes, cfg.nodes + cfg.nodes / 2, cfg.random_plies,
        if book.is_none() { "none" } else if cfg.book_uniform { "uniform" } else { "weighted" },
        cfg.seed,
    );

    let shared = Arc::new(Shared {
        cfg,
        book,
        tb: Arc::new(tb),
        claimed: AtomicU64::new(0),
        openings: Mutex::new(HashSet::new()),
        seen: Mutex::new(HashSet::new()),
        stats: Stats::default(),
    });

    let start = Instant::now();
    let done = AtomicU64::new(0);
    std::thread::scope(|s| {
        for (tid, (path, file)) in writers.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let done = &done;
            s.spawn(move || {
                worker(&shared, tid, file);
                eprintln!("datagen: thread {tid} finished -> {path}");
                done.fetch_add(1, Ordering::SeqCst);
            });
        }

        let mut last = Instant::now();
        while done.load(Ordering::SeqCst) < shared.cfg.threads as u64 {
            std::thread::sleep(Duration::from_millis(200));
            if last.elapsed() >= Duration::from_secs(10) {
                last = Instant::now();
                report(&shared.stats, start.elapsed());
            }
        }
    });
    report(&shared.stats, start.elapsed());
}

fn report(st: &Stats, elapsed: Duration) {
    let g = st.games.load(Ordering::Relaxed);
    let p = st.positions.load(Ordering::Relaxed);
    let searches = st.searches.load(Ordering::Relaxed).max(1);
    let nodes = st.nodes.load(Ordering::Relaxed);
    let secs = elapsed.as_secs_f64().max(1e-9);
    let sampled = st.sampled.load(Ordering::Relaxed);
    let unique = st.sampled_unique.load(Ordering::Relaxed);
    eprintln!(
        "datagen: {g} games  {p} positions  {:.0} pos/s  {} nodes/search  {:.0}k nps  W/D/L {}/{}/{}  rejected dup {} unbalanced {}  post-opening unique {:.2}% (sampled {sampled})  {:.0}s",
        p as f64 / secs,
        nodes / searches,
        nodes as f64 / secs / 1000.0,
        st.white_wins.load(Ordering::Relaxed),
        st.draws.load(Ordering::Relaxed),
        st.black_wins.load(Ordering::Relaxed),
        st.dup_rejects.load(Ordering::Relaxed),
        st.unbal_rejects.load(Ordering::Relaxed),
        if sampled == 0 { 100.0 } else { 100.0 * unique as f64 / sampled as f64 },
        secs,
    );
}

/// One played game: the moves from the start position, the search score
/// (side-to-move relative) of the position each was played from, and the
/// result from White's point of view.
struct Game {
    moves:        Vec<Move>,
    scores:       Vec<i32>,
    white_result: i16,
}

struct Engine {
    tt:     Arc<TranspositionTable>,
    tb:     Arc<Option<SyzygyTb>>,
    tables: Option<SearchTables>,
    /// Nodes and searches since the last `take_counts`.
    nodes:    Arc<AtomicU64>,
    searches: u64,
}

impl Engine {
    /// Search `board` with a soft node budget (hard cap 4x); returns the best
    /// move and its score.
    fn search(&mut self, board: &mut Board, soft_nodes: u64) -> (Move, i32) {
        let limits = SearchLimits {
            soft_nodes: Some(soft_nodes),
            nodes:      Some(soft_nodes * 4),
            quiet:      true,
            ..SearchLimits::default()
        };
        let mut searcher = Searcher::new(
            limits,
            Arc::clone(&self.tt),
            Arc::clone(&self.tb),
            Arc::new(AtomicU64::new(0)),
            Arc::clone(&self.nodes),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            self.tables.take().unwrap_or_default(),
            0,
        );
        let mv = searcher.search(board);
        let score = searcher.score();
        self.tables = Some(searcher.into_tables());
        self.searches += 1;
        (mv, score)
    }

    /// Fold this engine's node / search counts into the run totals.
    fn take_counts(&mut self, st: &Stats) {
        st.nodes.fetch_add(self.nodes.swap(0, Ordering::Relaxed), Ordering::Relaxed);
        st.searches.fetch_add(std::mem::take(&mut self.searches), Ordering::Relaxed);
    }

    /// Fresh state per game, so a game depends only on its own moves.
    fn new_game(&mut self) {
        self.tt.clear();
        self.tables = Some(SearchTables::default());
    }
}

fn worker(shared: &Shared, tid: usize, file: File) {
    let cfg = &shared.cfg;
    let mut rng = Rng::new(cfg.seed ^ (tid as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut engine = Engine {
        tt:     Arc::new(TranspositionTable::new(cfg.hash_mb)),
        tb:     Arc::clone(&shared.tb),
        tables: None,
        nodes:    Arc::new(AtomicU64::new(0)),
        searches: 0,
    };
    let mut writer = match CompressedTrainingDataEntryWriter::new(BufWriter::new(file)) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("datagen: thread {tid}: writer error: {e}");
            return;
        }
    };

    while shared.claimed.fetch_add(1, Ordering::SeqCst) < cfg.games {
        let mut rejects = 0;
        let game = loop {
            let game = play_game(shared, &mut engine, &mut rng);
            engine.take_counts(&shared.stats);
            if let Some(g) = game {
                break g;
            }
            rejects += 1;
            if rejects >= MAX_REJECTS {
                eprintln!("datagen: thread {tid}: {MAX_REJECTS} openings rejected in a row, stopping");
                return;
            }
        };
        if let Err(e) = write_game(&mut writer, &game) {
            eprintln!("datagen: thread {tid}: write error: {e}");
            return;
        }
        let st = &shared.stats;
        st.games.fetch_add(1, Ordering::Relaxed);
        st.positions.fetch_add(game.moves.len() as u64, Ordering::Relaxed);
        match game.white_result {
            1 => st.white_wins.fetch_add(1, Ordering::Relaxed),
            -1 => st.black_wins.fetch_add(1, Ordering::Relaxed),
            _ => st.draws.fetch_add(1, Ordering::Relaxed),
        };
    }

}

/// Build the opening move list: book moves, then random plies. `None` if the
/// game would end inside the opening.
fn pick_opening(shared: &Shared, rng: &mut Rng) -> Option<(Vec<Move>, Board)> {
    let mut board = Board::start_pos();
    let mut moves = Vec::new();
    if let Some(book) = &shared.book {
        loop {
            let mv = if shared.cfg.book_uniform {
                let list = book.moves(&board);
                if list.is_empty() { None } else { Some(list[(rng.next() % list.len() as u64) as usize].0) }
            } else {
                book.pick(&board, rng.next())
            };
            let Some(mv) = mv else { break };
            board.make_move(mv);
            moves.push(mv);
        }
    }
    for _ in 0..shared.cfg.random_plies {
        let mut list = MoveList::new();
        MoveGen::generate_all(&board, &mut list);
        if list.is_empty() {
            return None;
        }
        let mv = list[(rng.next() % list.len() as u64) as usize];
        board.make_move(mv);
        moves.push(mv);
    }
    Some((moves, board))
}

fn play_game(shared: &Shared, engine: &mut Engine, rng: &mut Rng) -> Option<Game> {
    let cfg = &shared.cfg;
    let (opening, mut end) = pick_opening(shared, rng)?;
    let jitter = |rng: &mut Rng| cfg.nodes + rng.next() % (cfg.nodes / 2 + 1);

    if !shared.openings.lock().unwrap().insert(end.hash) {
        shared.stats.dup_rejects.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if game_over(&end, shared).is_some() {
        return None;
    }
    engine.new_game();
    let (_, s) = engine.search(&mut end, jitter(rng));
    if s.abs() > MAX_OPENING_SCORE {
        shared.stats.unbal_rejects.fetch_add(1, Ordering::Relaxed);
        return None;
    }

    engine.new_game();
    let mut board = Board::start_pos();
    let mut moves = Vec::new();
    let mut scores = Vec::new();
    let mut win_run = 0u32;
    let mut win_sign = 0i32;
    let mut draw_run = 0u32;

    let white_result = loop {
        if let Some(r) = game_over(&board, shared) {
            break r;
        }
        let ply = moves.len();
        if ply >= MAX_GAME_PLY {
            break 0;
        }

        let (best, score) = engine.search(&mut board, jitter(rng));
        if best.is_null() {
            break 0;
        }
        let mv = if ply < opening.len() { opening[ply] } else { best };

        let white_score = if board.side == Color::White { score } else { -score };
        if white_score.abs() >= WIN_SCORE {
            let sign = white_score.signum();
            win_run = if sign == win_sign { win_run + 1 } else { 1 };
            win_sign = sign;
        } else {
            win_run = 0;
        }
        draw_run = if ply >= DRAW_MIN_PLY && score.abs() <= DRAW_SCORE { draw_run + 1 } else { 0 };

        if ply >= opening.len() {
            sample_position(shared, board.hash);
        }
        moves.push(mv);
        scores.push(score);
        board.make_move(mv);

        if win_run >= WIN_PLIES || white_score.abs() >= MATE_SCORE - 1000 && ply >= opening.len() {
            break win_sign as i16;
        }
        if draw_run >= DRAW_PLIES {
            break 0;
        }
    };

    Some(Game { moves, scores, white_result })
}

/// Result from White's point of view if the game is over here: no legal
/// moves, 50-move rule, threefold repetition, insufficient material, or a
/// tablebase verdict.
fn game_over(board: &Board, shared: &Shared) -> Option<i16> {
    let mut list = MoveList::new();
    MoveGen::generate_all(board, &mut list);
    let stm_white = board.side == Color::White;
    if list.is_empty() {
        if board.in_check() {
            return Some(if stm_white { -1 } else { 1 });
        }
        return Some(0);
    }
    if board.halfmove >= 100 || Searcher::is_material_draw(board) || is_threefold(board) {
        return Some(0);
    }
    if let Some(tb) = shared.tb.as_ref()
        && let Some(wdl) = tb.probe_wdl(board)
    {
        let stm = match wdl {
            WdlResult::Win => 1,
            WdlResult::Loss => -1,
            _ => 0,
        };
        return Some(if stm_white { stm } else { -stm });
    }
    None
}

/// The current position has occurred twice before (same side to move,
/// within the reversible stretch).
fn is_threefold(board: &Board) -> bool {
    let n = board.history.len();
    let lookback = (board.halfmove as usize).min(n);
    let mut hits = 0;
    for k in (2..=lookback).step_by(2) {
        if board.history[n - k].1.hash == board.hash {
            hits += 1;
            if hits >= 2 {
                return true;
            }
        }
    }
    false
}

fn sample_position(shared: &Shared, key: u64) {
    if !key.is_multiple_of(SAMPLE_MOD) {
        return;
    }
    shared.stats.sampled.fetch_add(1, Ordering::Relaxed);
    if shared.seen.lock().unwrap().insert(key) {
        shared.stats.sampled_unique.fetch_add(1, Ordering::Relaxed);
    }
}

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Write one game as a single binpack chain. Positions are derived with
/// sfbinpack's own `after_move`, so consecutive entries are always
/// recognised as continuations.
fn write_game<W: std::io::Write>(
    writer: &mut CompressedTrainingDataEntryWriter<W>,
    game: &Game,
) -> Result<(), String> {
    let mut board = Board::start_pos();
    let mut pos = SfPosition::from_fen(STARTPOS).map_err(|e| format!("{e:?}"))?;
    for (ply, (&mv, &score)) in game.moves.iter().zip(&game.scores).enumerate() {
        let stm_white = board.side == Color::White;
        let sf_mv = to_sf_move(&board, mv);
        let entry = TrainingDataEntry {
            pos,
            mv: sf_mv,
            score: score.clamp(-SCORE_CLAMP, SCORE_CLAMP) as i16,
            ply: ply as u16,
            result: if stm_white { game.white_result } else { -game.white_result },
        };
        writer.write_entry(&entry).map_err(|e| e.to_string())?;
        pos = pos.after_move(sf_mv);
        board.make_move(mv);
    }
    Ok(())
}

/// Engine move -> sfbinpack move. Castling is king-takes-rook there; en
/// passant and promotions carry their own move type.
fn to_sf_move(board: &Board, mv: Move) -> SfMove {
    let from = SfSquare::new(mv.from() as u32);
    let to = SfSquare::new(mv.to() as u32);
    let color = if board.side == Color::White { SfColor::White } else { SfColor::Black };
    match mv.flags() {
        MoveFlag::CastleKing | MoveFlag::CastleQueen => {
            let rank_base = mv.from() & !7;
            let rook = if mv.flags() == MoveFlag::CastleKing { rank_base + 7 } else { rank_base };
            SfMove::new(from, SfSquare::new(rook as u32), SfMoveType::Castle, SfPiece::none())
        }
        MoveFlag::EnPassant => SfMove::new(from, to, SfMoveType::EnPassant, SfPiece::none()),
        _ => match mv.promo_piece() {
            Some(p) => {
                let pt = match p {
                    Piece::Knight => SfPieceType::Knight,
                    Piece::Bishop => SfPieceType::Bishop,
                    Piece::Rook => SfPieceType::Rook,
                    _ => SfPieceType::Queen,
                };
                SfMove::new(from, to, SfMoveType::Promotion, SfPiece::new(pt, color))
            }
            None => SfMove::new(from, to, SfMoveType::Normal, SfPiece::none()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sfbinpack::CompressedTrainingDataEntryReader;

    /// Same position: board, side and castling must match exactly. The
    /// engine sets the en passant square after every double push, sfbinpack
    /// only when a capture is possible, so ep must match only when sfbinpack
    /// has one.
    fn assert_same(sf: &str, engine: &str, what: &str) {
        let a: Vec<&str> = sf.split_whitespace().collect();
        let b: Vec<&str> = engine.split_whitespace().collect();
        assert_eq!(a[..3], b[..3], "{what}: {sf} vs {engine}");
        if a[3] != "-" {
            assert_eq!(a[3], b[3], "{what}: {sf} vs {engine}");
        }
    }

    /// Play a short real game set, write it, read it back with the trainer's
    /// reader, and check every position, move and result against the engine.
    #[test]
    fn binpack_round_trip() {
        crate::nnue::load_embedded();
        let dir = std::env::temp_dir().join(format!("erebus-datagen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.binpack");
        let _ = std::fs::remove_file(&path);

        let shared = Shared {
            cfg: Config {
                out: String::new(), games: 0, threads: 1, nodes: 300, book: None,
                book_uniform: false, random_plies: 8, syzygy: None, seed: 7, hash_mb: 2,
            },
            book: None,
            tb: Arc::new(None),
            claimed: AtomicU64::new(0),
            openings: Mutex::new(HashSet::new()),
            seen: Mutex::new(HashSet::new()),
            stats: Stats::default(),
        };
        let mut engine = Engine {
            tt: Arc::new(TranspositionTable::new(2)), tb: Arc::new(None), tables: None,
            nodes: Arc::new(AtomicU64::new(0)), searches: 0,
        };
        let mut rng = Rng::new(99);
        let mut games = Vec::new();
        while games.len() < 3 {
            if let Some(g) = play_game(&shared, &mut engine, &mut rng) {
                games.push(g);
            }
        }
        {
            let mut w = CompressedTrainingDataEntryWriter::new(File::create(&path).unwrap()).unwrap();
            for g in &games {
                write_game(&mut w, g).unwrap();
            }
        }

        let mut reader = CompressedTrainingDataEntryReader::new(File::open(&path).unwrap()).unwrap();
        for g in &games {
            let mut board = Board::start_pos();
            for (ply, &mv) in g.moves.iter().enumerate() {
                assert!(reader.has_next());
                let e = reader.next();
                assert_same(&e.pos.fen().unwrap(), &board.to_fen(), &format!("ply {ply}"));
                let uci = e.mv.as_uci();
                assert_eq!(board.find_uci_move(&uci).map(|m| m.to_uci()), Some(mv.to_uci()), "{uci}");
                assert_eq!(e.ply as usize, ply);
                let stm_res = if board.side == Color::White { g.white_result } else { -g.white_result };
                assert_eq!(e.result, stm_res);
                assert_eq!(e.score as i32, g.scores[ply].clamp(-SCORE_CLAMP, SCORE_CLAMP));
                board.make_move(mv);
            }
        }
        assert!(!reader.has_next());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn castling_and_promotion_convert() {
        let b = Board::from_fen("r3k2r/1P6/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let conv = |u: &str| to_sf_move(&b, b.find_uci_move(u).unwrap()).as_uci();

        let pos = SfPosition::from_fen(&b.to_fen()).unwrap();
        for u in ["e1g1", "e1c1", "b7a8q", "b7b8n"] {
            let sf = to_sf_move(&b, b.find_uci_move(u).unwrap());
            let after = pos.after_move(sf).fen().unwrap();
            let mut eb = b.clone();
            eb.make_move(eb.find_uci_move(u).unwrap());
            assert_same(&after, &eb.to_fen(), &format!("{u} ({})", conv(u)));
        }
    }
}
