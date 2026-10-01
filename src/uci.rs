use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::board::Board;
use crate::book::{Book, Rng};
use crate::movegen::MoveGen;
use crate::search::{
    SearchLimits, SearchTables, Searcher, SEE_MARGIN_MAX, SEE_NOISY_MARGIN,
    SEE_NOISY_MARGIN_DEFAULT, SEE_PRUNE_DEPTH, SEE_PRUNE_DEPTH_DEFAULT, SEE_PRUNE_DEPTH_MAX,
    SEE_QUIET_MARGIN, SEE_QUIET_MARGIN_DEFAULT,
};
use crate::syzygy::SyzygyTb;
use crate::tt::TranspositionTable;

const ENGINE_NAME:   &str = concat!("Erebus ", env!("CARGO_PKG_VERSION"));
const ENGINE_AUTHOR: &str = "Joshua Shriver";

const MOVES_TO_GO_DEFAULT: u64 = 40;
const SOFT_FACTOR:         f64 = 0.6;
const HARD_FACTOR:         f64 = 2.5;

const ALLOC_CAP_DIV:       u64 = 5;
const HARD_CAP_DIV:        u64 = 3;

const MOVE_OVERHEAD_DEFAULT: u64 = 30;
const MOVE_OVERHEAD_MAX:     u64 = 5000;

pub const MAX_THREADS: usize = 512;

struct SmpState {
    tt:             Arc<TranspositionTable>,
    tb:             Arc<Option<SyzygyTb>>,
    board:          Arc<Mutex<Board>>,
    stop:           Arc<AtomicBool>,
    ponderhit:      Arc<AtomicBool>,

    tbhits:         Arc<AtomicU64>,

    nodes_global:   Arc<AtomicU64>,

    ponder_stop:    Arc<AtomicBool>,

    ponder_epoch:   Arc<AtomicU64>,

    abort_silent:   Arc<AtomicBool>,
    ponder_threads: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>,

    tables_pool:    Vec<Arc<Mutex<Option<SearchTables>>>>,
    threads:        Vec<std::thread::JoinHandle<()>>,
}

impl SmpState {
    fn new(tt_mb: usize) -> Self {
        SmpState {
            tt:            Arc::new(TranspositionTable::new(tt_mb)),
            tb:            Arc::new(None),

            board:         Arc::new(Mutex::new(Board::start_pos())),
            stop:          Arc::new(AtomicBool::new(false)),
            ponderhit:     Arc::new(AtomicBool::new(false)),
            tbhits:        Arc::new(AtomicU64::new(0)),
            nodes_global:  Arc::new(AtomicU64::new(0)),
            ponder_stop:   Arc::new(AtomicBool::new(false)),
            ponder_epoch:  Arc::new(AtomicU64::new(0)),
            abort_silent:  Arc::new(AtomicBool::new(false)),
            ponder_threads: Arc::new(Mutex::new(Vec::new())),
            tables_pool:   Vec::new(),
            threads:       Vec::new(),
        }
    }

    /// Ensure the persistent-tables pool has a slot for every thread id in
    /// `0..n`. New slots start empty; the tables themselves are allocated
    /// lazily by the first search that takes the slot.
    fn ensure_tables_pool(&mut self, n: usize) {
        while self.tables_pool.len() < n {
            self.tables_pool.push(Arc::new(Mutex::new(None)));
        }
    }

    fn stop_and_wait(&mut self) {
        self.abort_silent.store(true, Ordering::SeqCst);
        self.stop.store(true, Ordering::SeqCst);
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
    }

    /// Halt the autonomous ponder search (if one is running) and reap its
    /// threads. Called at the top of every command (except `isready`) so any
    /// command implicitly ends pondering; cheap no-op when nothing is
    /// pondering. The TT entries
    /// the ponder produced stay -- that warmth is the whole point.
    fn stop_ponder(&self) {

        self.ponder_epoch.fetch_add(1, Ordering::SeqCst);
        let handles: Vec<_> = {
            let mut g = self.ponder_threads.lock().unwrap();
            self.ponder_stop.store(true, Ordering::SeqCst);
            g.drain(..).collect()
        };
        for h in handles {
            let _ = h.join();
        }
    }

    fn is_running(&self) -> bool {
        !self.threads.is_empty()
    }
}

/// `initial_threads` / `initial_hash_mb` come from the `--threads` /
/// `--hash` CLI flags (see `main.rs::parse_flag_value`); both are already
/// clamped to their valid UCI ranges by the caller. They only set the
/// starting values -- "setoption name Threads/Hash value <n>" can still
/// change them later in the session.
pub fn uci_loop(syzygy_path: Option<String>, initial_threads: usize, initial_hash_mb: usize) {
    let stdin = io::stdin();
    let mut tt_mb          = initial_hash_mb;
    let mut num_threads    = initial_threads;
    let mut ponder_enabled = true;
    let mut move_overhead  = MOVE_OVERHEAD_DEFAULT;
    let mut own_book       = false;
    let mut book: Option<Book> = None;
    let mut book_rng       = Rng::from_time();

    let mut position_at: Option<Instant> = None;

    let mut smp = SmpState::new(tt_mb);

    if let Some(ref path) = syzygy_path {
        load_syzygy(&mut smp, path);
    }

    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let received_at = Instant::now();
        let line = line.trim().to_string();
        if line.is_empty() { continue; }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        let prev_position_at = position_at.take();

        if tokens[0] != "isready" {
            smp.stop_ponder();
        }
        match tokens[0] {
            "uci" => cmd_uci(),

            "isready" => println!("readyok"),

            "ucinewgame" => {
                smp.stop_and_wait();
                *smp.board.lock().unwrap() = Board::start_pos();
                smp.tt.clear();

                for slot in &smp.tables_pool {
                    *slot.lock().unwrap() = None;
                }
            }

            "setoption" => {

                if let Some(name_idx) = tokens.iter().position(|&t| t.eq_ignore_ascii_case("name")) {
                    let value_idx = tokens.iter().position(|&t| t.eq_ignore_ascii_case("value"));
                    let name_end  = value_idx.unwrap_or(tokens.len());
                    let name      = tokens[(name_idx + 1)..name_end].join(" ");
                    let value     = value_idx
                        .map(|vi| tokens[(vi + 1)..].join(" "))
                        .unwrap_or_default();

                    match name.to_ascii_lowercase().as_str() {
                        "hash" => {
                            if let Ok(v) = value.parse::<usize>() {
                                let new_mb = v.clamp(1, 1024);
                                if new_mb != tt_mb {
                                    tt_mb = new_mb;
                                    smp.stop_and_wait();
                                    smp.tt = Arc::new(TranspositionTable::new(tt_mb));
                                }
                            }
                        }
                        "threads" => {
                            if let Ok(v) = value.parse::<usize>() {
                                num_threads = v.clamp(1, MAX_THREADS);
                            }
                        }
                        "ponder" => {
                            ponder_enabled = value.eq_ignore_ascii_case("true");
                        }

                        "moveoverhead" | "move overhead" => {
                            if let Ok(v) = value.parse::<u64>() {
                                move_overhead = v.min(MOVE_OVERHEAD_MAX);
                            }
                        }

                        "seeprunedepth" => {
                            if let Ok(v) = value.parse::<i32>() {
                                SEE_PRUNE_DEPTH.store(v.clamp(0, SEE_PRUNE_DEPTH_MAX), Ordering::Relaxed);
                            }
                        }
                        "seenoisymargin" => {
                            if let Ok(v) = value.parse::<i32>() {
                                SEE_NOISY_MARGIN.store(v.clamp(0, SEE_MARGIN_MAX), Ordering::Relaxed);
                            }
                        }
                        "seequietmargin" => {
                            if let Ok(v) = value.parse::<i32>() {
                                SEE_QUIET_MARGIN.store(v.clamp(0, SEE_MARGIN_MAX), Ordering::Relaxed);
                            }
                        }
                        "ownbook" => {
                            own_book = value.eq_ignore_ascii_case("true");
                        }
                        "bookfile" => {
                            book = load_book(&value);
                        }
                        "syzygypath" => {
                            smp.stop_and_wait();
                            load_syzygy(&mut smp, &value);
                        }
                        "evalfile" => {

                            smp.stop_and_wait();
                            load_eval_file(&value);
                            smp.board.lock().unwrap().reload_net();
                        }
                        _ => {}
                    }
                }
            }

            "position" => {
                if smp.is_running() { smp.stop_and_wait(); }
                cmd_position(&mut smp.board.lock().unwrap(), &tokens);
                position_at = Some(received_at);
            }

            "go" => {
                smp.stop_and_wait();
                smp.stop    .store(false, Ordering::SeqCst);
                smp.abort_silent.store(false, Ordering::SeqCst);
                smp.ponderhit.store(false, Ordering::SeqCst);
                smp.tbhits  .store(0, Ordering::SeqCst);
                smp.nodes_global.store(0, Ordering::SeqCst);
                smp.tt.new_generation();

                if own_book
                    && !tokens.contains(&"ponder")
                    && !tokens.contains(&"infinite")
                    && let Some(ref bk) = book
                    && let Some(mv) = bk.pick(&smp.board.lock().unwrap(), book_rng.next())
                {
                    println!("info string book move");
                    println!("bestmove {}", mv);
                    continue;
                }

                let is_ponder = tokens.contains(&"ponder") && ponder_enabled;
                let mut limits = parse_limits(
                    &tokens, &smp.board.lock().unwrap(), is_ponder, move_overhead,
                );
                limits.start = Some(prev_position_at.unwrap_or(received_at));

                let helper_count = num_threads.saturating_sub(1);
                smp.ensure_tables_pool(num_threads.max(1));

                for tid in 1..=helper_count {

                    let helper_board = smp.board.lock().unwrap().clone();

                    let helper_limits = SearchLimits {
                        max_depth:        limits.max_depth,
                        move_time:        None,
                        soft_time:        None,
                        nodes:            None,
                        soft_nodes:       None,
                        infinite:         true,
                        ponder:           false,
                        ponder_move_time: None,
                        ponder_soft_time: None,
                        start:            None,
                        quiet:            false,
                    };

                    let tt_arc    = Arc::clone(&smp.tt);
                    let tb_arc    = Arc::clone(&smp.tb);
                    let tbh_arc   = Arc::clone(&smp.tbhits);
                    let ng_arc    = Arc::clone(&smp.nodes_global);
                    let stop_arc  = Arc::clone(&smp.stop);
                    let phit_arc  = Arc::clone(&smp.ponderhit);
                    let tbl_slot  = Arc::clone(&smp.tables_pool[tid]);

                    let handle = std::thread::spawn(move || {
                        let mut board    = helper_board;
                        let tables       = tbl_slot.lock().unwrap().take().unwrap_or_default();
                        let mut searcher = Searcher::new(
                            helper_limits,
                            tt_arc,
                            tb_arc,
                            tbh_arc,
                            ng_arc,
                            stop_arc,
                            phit_arc,
                            tables,
                            tid,
                        );
                        searcher.search(&mut board);
                        *tbl_slot.lock().unwrap() = Some(searcher.into_tables());
                    });
                    smp.threads.push(handle);
                }

                let ponder_out = ponder_enabled;
                let board_arc = Arc::clone(&smp.board);
                let tt_arc    = Arc::clone(&smp.tt);
                let tb_arc    = Arc::clone(&smp.tb);
                let tbh_arc   = Arc::clone(&smp.tbhits);
                let ng_arc    = Arc::clone(&smp.nodes_global);
                let stop_arc  = Arc::clone(&smp.stop);
                let phit_arc  = Arc::clone(&smp.ponderhit);
                let tbl_slot  = Arc::clone(&smp.tables_pool[0]);

                let pd_tt      = Arc::clone(&smp.tt);
                let pd_tb      = Arc::clone(&smp.tb);
                let pd_tbh     = Arc::clone(&smp.tbhits);
                let pd_ng      = Arc::clone(&smp.nodes_global);
                let pd_pstop   = Arc::clone(&smp.ponder_stop);
                let pd_threads = Arc::clone(&smp.ponder_threads);
                let pd_epoch   = Arc::clone(&smp.ponder_epoch);
                let pd_epoch_at = smp.ponder_epoch.load(Ordering::SeqCst);
                let silent_arc = Arc::clone(&smp.abort_silent);
                let pd_nthreads = num_threads.max(1);

                let handle = std::thread::spawn(move || {
                    let mut board    = board_arc.lock().unwrap();
                    let tables       = tbl_slot.lock().unwrap().take().unwrap_or_default();
                    let mut searcher = Searcher::new(
                        limits,
                        tt_arc,
                        tb_arc,
                        tbh_arc,
                        ng_arc,
                        Arc::clone(&stop_arc),
                        Arc::clone(&phit_arc),
                        tables,
                        0,
                    );
                    let best = searcher.search(&mut board);
                    let pmove = searcher.ponder_move();

                    stop_arc.store(true, Ordering::SeqCst);

                    let was_pondering = searcher.limits.ponder;
                    let hit     = phit_arc.load(Ordering::Relaxed);
                    let silent  = silent_arc.load(Ordering::SeqCst);

                    let emitted_move = !was_pondering || hit || !silent;
                    if emitted_move {
                        if best.is_null() {
                            let mut moves = crate::moves::MoveList::new();
                            MoveGen::generate_all(&board, &mut moves);
                            let mv = if moves.is_empty() {
                                "(none)".to_string()
                            } else {
                                moves[0].to_string()
                            };
                            println!("bestmove {}", mv);
                        } else {

                            if ponder_out && !pmove.is_null() {
                                println!("bestmove {} ponder {}", best, pmove);
                            } else {
                                println!("bestmove {}", best);
                            }
                        }
                    }

                    *tbl_slot.lock().unwrap() = Some(searcher.into_tables());

                    if emitted_move
                        && ponder_out
                        && !was_pondering
                        && !best.is_null()
                        && !pmove.is_null()
                    {

                        let mut pb = board.clone();
                        drop(board);
                        let mut ok = false;
                        if let Some(m) = pb.find_uci_move(&best.to_uci()) {
                            pb.make_move(m);
                            if let Some(r) = pb.find_uci_move(&pmove.to_uci()) {
                                pb.make_move(r);
                                ok = true;
                            }
                        }

                        let mut guard = pd_threads.lock().unwrap();
                        let still_current =
                            pd_epoch.load(Ordering::SeqCst) == pd_epoch_at;
                        if still_current && ok {
                            pd_pstop.store(false, Ordering::SeqCst);
                            let mut handles = Vec::with_capacity(pd_nthreads);

                            for tid in 0..pd_nthreads {
                                let mut pb = pb.clone();
                                let ptt    = Arc::clone(&pd_tt);
                                let ptb    = Arc::clone(&pd_tb);
                                let ptbh   = Arc::clone(&pd_tbh);
                                let png    = Arc::clone(&pd_ng);
                                let pstop  = Arc::clone(&pd_pstop);
                                handles.push(std::thread::spawn(move || {
                                    let plimits = SearchLimits {
                                        infinite: true,
                                        ..SearchLimits::default()
                                    };
                                    let mut ps = Searcher::new(
                                        plimits,
                                        ptt, ptb, ptbh, png,
                                        pstop,
                                        Arc::new(AtomicBool::new(false)),
                                        SearchTables::default(),
                                        tid + 1,
                                    );
                                    ps.search(&mut pb);
                                }));
                            }
                            *guard = handles;
                        }
                    }
                });
                smp.threads.push(handle);
            }

            "ponderhit" => {
                smp.ponderhit.store(true, Ordering::SeqCst);
            }

            "stop" => {
                smp.stop.store(true, Ordering::SeqCst);
                for h in smp.threads.drain(..) {
                    let _ = h.join();
                }

                smp.stop_ponder();
            }

            "quit" => {
                smp.stop.store(true, Ordering::SeqCst);
                for h in smp.threads.drain(..) {
                    let _ = h.join();
                }
                smp.stop_ponder();
                break;
            }

            "d"     => smp.board.lock().unwrap().print(),
            "fen"   => println!("{}", smp.board.lock().unwrap().to_fen()),
            "perft" => cmd_perft(&mut smp.board.lock().unwrap(), &tokens),
            "eval"  => cmd_eval(&mut smp.board.lock().unwrap()),
            _ => {}
        }
    }
}

fn load_syzygy(smp: &mut SmpState, path: &str) {
    if path.is_empty() || path == "<empty>" {
        smp.tb = Arc::new(None);
        eprintln!("info string Syzygy tablebases cleared");
        return;
    }

    let mut tb = SyzygyTb::new();

    let mut total = 0usize;
    for p in path.split(';') {
        let p = p.trim();
        if !p.is_empty() {
            let n = tb.add_directory(p);
            eprintln!("info string Loaded {} Syzygy file(s) from {}", n, p);
            total += n;
        }
    }

    if total == 0 {
        eprintln!("info string Warning: no Syzygy files found in '{}'", path);
        smp.tb = Arc::new(None);
    } else {
        eprintln!(
            "info string Syzygy tablebases loaded: {} file(s), up to {} pieces",
            total, tb.max_pieces()
        );
        smp.tb = Arc::new(Some(tb));
    }
}

fn load_book(path: &str) -> Option<Book> {
    if path.is_empty() || path == "<empty>" {
        eprintln!("info string Opening book cleared");
        return None;
    }
    match Book::load(path) {
        Ok(b) => {
            eprintln!("info string Opening book loaded: {} ({} entries)", path, b.len());
            Some(b)
        }
        Err(e) => {
            eprintln!("info string Failed to load BookFile '{}': {}", path, e);
            None
        }
    }
}

fn load_eval_file(path: &str) {
    if path.is_empty() || path == "<empty>" {
        eprintln!("info string EvalFile ignored: no classical fallback exists, keeping current net");
        return;
    }

    match crate::nnue::load(path) {
        Ok(()) => eprintln!("info string NNUE eval file loaded: {}", path),
        Err(e) => eprintln!("info string Failed to load EvalFile '{}': {} -- keeping previous net", path, e),
    }
}

fn cmd_uci() {
    println!("id name {}", ENGINE_NAME);
    println!("id author {}", ENGINE_AUTHOR);
    println!("option name Hash        type spin   default 16   min 1    max 1024");
    println!("option name Threads     type spin   default 1    min 1    max {}", MAX_THREADS);
    println!("option name Ponder      type check  default true");
    println!("option name Move Overhead type spin default {MOVE_OVERHEAD_DEFAULT} min 0 max {MOVE_OVERHEAD_MAX}");
    println!("option name SyzygyPath  type string default <empty>");
    println!("option name EvalFile    type string default <embedded>");
    println!("option name OwnBook     type check  default false");
    println!("option name BookFile    type string default <empty>");
    println!("option name SeePruneDepth type spin default {SEE_PRUNE_DEPTH_DEFAULT} min 0 max {SEE_PRUNE_DEPTH_MAX}");
    println!("option name SeeNoisyMargin type spin default {SEE_NOISY_MARGIN_DEFAULT} min 0 max {SEE_MARGIN_MAX}");
    println!("option name SeeQuietMargin type spin default {SEE_QUIET_MARGIN_DEFAULT} min 0 max {SEE_MARGIN_MAX}");
    println!("uciok");
}

fn cmd_position(board: &mut Board, tokens: &[&str]) {
    if tokens.len() < 2 { return; }
    let moves_start;
    match tokens[1] {
        "startpos" => { *board = Board::start_pos(); moves_start = 2; }
        "fen" => {
            let fen_end = tokens.iter().position(|&t| t == "moves").unwrap_or(tokens.len());
            let fen = tokens[2..fen_end].join(" ");
            *board = match Board::from_fen(&fen) {
                Some(b) => b,
                None    => { eprintln!("Invalid FEN: {}", fen); return; }
            };
            moves_start = fen_end;
        }
        _ => return,
    }
    if moves_start < tokens.len() && tokens[moves_start] == "moves" {
        board.apply_moves(&tokens[(moves_start + 1)..]);
    }
}

fn parse_limits(tokens: &[&str], board: &Board, is_ponder: bool, overhead: u64) -> SearchLimits {
    let mut limits    = SearchLimits::default();
    let mut wtime:     Option<u64> = None;
    let mut btime:     Option<u64> = None;
    let mut winc:      u64 = 0;
    let mut binc:      u64 = 0;
    let mut movestogo: Option<u64> = None;
    let mut movetime:  Option<u64> = None;

    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "depth"     => { if let Some(&d) = tokens.get(i+1) { limits.max_depth = d.parse().unwrap_or(64); } i += 2; }
            "nodes"     => { if let Some(&n) = tokens.get(i+1) { limits.nodes     = n.parse().ok(); } i += 2; }
            "movetime"  => { if let Some(&t) = tokens.get(i+1) { movetime         = t.parse().ok(); } i += 2; }
            "wtime"     => { if let Some(&t) = tokens.get(i+1) { wtime            = t.parse().ok(); } i += 2; }
            "btime"     => { if let Some(&t) = tokens.get(i+1) { btime            = t.parse().ok(); } i += 2; }
            "winc"      => { if let Some(&t) = tokens.get(i+1) { winc             = t.parse().unwrap_or(0); } i += 2; }
            "binc"      => { if let Some(&t) = tokens.get(i+1) { binc             = t.parse().unwrap_or(0); } i += 2; }
            "movestogo" => { if let Some(&m) = tokens.get(i+1) { movestogo        = m.parse().ok(); } i += 2; }
            "infinite"  => { limits.infinite = true; i += 1; }
            "ponder"    => { i += 1; }
            _           => { i += 1; }
        }
    }

    let (soft, hard) = if let Some(mt) = movetime {
        let ms = mt.saturating_sub(overhead).max(1);
        (Some(Duration::from_millis(ms)), Some(Duration::from_millis(ms)))
    } else if !limits.infinite && limits.nodes.is_none() {
        use crate::board::Color;
        let (time_left, inc) = if board.side == Color::White { (wtime, winc) } else { (btime, binc) };
        if let Some(t) = time_left {
            let t       = t.saturating_sub(overhead);
            let mtg     = movestogo.unwrap_or(MOVES_TO_GO_DEFAULT).max(1);

            let bank    = (t + inc * (mtg - 1)).saturating_sub(overhead * (mtg - 1));
            let alloc   = bank / mtg;

            let (alloc_cap, hard_cap) = if mtg < ALLOC_CAP_DIV {
                (t / 2, t / 2)
            } else {
                (t / ALLOC_CAP_DIV, t / HARD_CAP_DIV)
            };
            let alloc   = alloc.min(alloc_cap).max(1);
            let soft_ms = (alloc as f64 * SOFT_FACTOR) as u64;
            let hard_ms = ((alloc as f64 * HARD_FACTOR) as u64).min(hard_cap).max(soft_ms);
            (Some(Duration::from_millis(soft_ms.max(1))),
             Some(Duration::from_millis(hard_ms.max(1))))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    if is_ponder {
        limits.ponder           = true;
        limits.infinite         = true;
        limits.ponder_soft_time = soft;
        limits.ponder_move_time = hard;
    } else {
        limits.soft_time = soft;
        limits.move_time = hard;
    }

    limits
}

fn cmd_perft(board: &mut Board, tokens: &[&str]) {
    let depth: u32 = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let results = MoveGen::perft_divide(board, depth);
    let mut total = 0u64;
    for (mv, nodes) in &results {
        println!("{}: {}", mv, nodes);
        total += nodes;
    }
    println!("\nNodes searched: {}", total);
}

fn cmd_eval(board: &mut Board) {
    let cp = crate::eval::evaluate(board);
    println!("eval (nnue, stm-relative): {cp} cp");
}
