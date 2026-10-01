# Erebus

A UCI chess engine written in Rust: bitboard move generation, an NNUE
evaluation, and a modern alpha-beta search with Lazy SMP.

---

## Building

```bash
cargo build --release
./target/release/erebus
```

Requires Rust 1.88 or newer (edition 2024). The build uses
`-C target-cpu=native` (see `.cargo/config.toml`), so the binary is tuned for
the machine that builds it and may not run on an older CPU. Build on the
machine you run on, or override the target:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

The network (`nets/net.nnue`) is embedded into the executable, so the binary
is self-contained.

---

## Evaluation

NNUE only; there is no handcrafted evaluation.

- **Architecture:** `(768 -> 1024) x 2 -> 1` with a SCReLU activation. The 768
  inputs are color x piece x square, from each side's perspective.
- **Quantisation:** QA = 255, QB = 64, eval scale 400, the standard
  [bullet](https://github.com/jw1912/bullet) scheme.
- **Incremental accumulators:** two i16 accumulators (White's and Black's
  perspective) live on the board. A move toggles at most 4 features, and
  unmake replays those toggles inverted.
- **SIMD:** AVX2 on x86-64 and NEON on aarch64, with a scalar fallback. The
  SCReLU and output-weight multiply are fused into one `madd_epi16`. Tests
  check that the SIMD kernels are bit-identical to the scalar reference.

See [`docs/TRAINER_HANDOFF.md`](docs/TRAINER_HANDOFF.md) for the net file
format and how to train a replacement.

---

## Search

| Area | Features |
|---|---|
| **Framework** | Iterative deepening, principal variation search, aspiration windows, triangular PV table |
| **Transposition table** | Lock-free (XOR-verified) shared table, depth-preferred replacement with per-search generation aging, mate scores stored ply-adjusted |
| **Pruning** | Reverse futility, razoring, adaptive null-move, futility, late-move, history, SEE, and mate-distance pruning |
| **Reductions / extensions** | Log-log late-move reductions, internal iterative reductions, check extension |
| **Move ordering** | TT move, promotions, MVV-LVA captures, two killers per ply, counter-move, threat-aware history (`[stm][from][to][from attacked][to attacked]`) with gravity and malus |
| **Draws** | Repetition (first recurrence inside the tree), 50-move rule, insufficient material |
| **Quiescence** | Captures and promotions, SEE filter, TT probe, full evasions when in check |
| **Parallelism** | Lazy SMP: helper threads with staggered start depths share the TT |
| **Endgames** | Syzygy WDL/DTZ probing via `shakmaty-syzygy` |

History and counter-move tables carry over from one move of a game to the
next (decayed at the start of each search). `ucinewgame` clears them.

---

## UCI

### Options

| Option | Type | Default | Notes |
|---|---|---|---|
| `Hash` | spin | 16 | TT size in MB (1–1024) |
| `Threads` | spin | 1 | Search threads (1–512) |
| `Ponder` | check | true | See [Pondering](#pondering) |
| `Move Overhead` | spin | 30 | ms reserved per move for GUI / network lag (0–5000); raise it if the engine still loses on time. `MoveOverhead` is accepted too |
| `SyzygyPath` | string | `<empty>` | One or more directories, `;`-separated |
| `EvalFile` | string | `<embedded>` | Diagnostic: load a different net of the same architecture from disk |
| `OwnBook` | check | `false` | Play moves from `BookFile` instantly while in book (not during `go ponder` / `go infinite`) |
| `BookFile` | string | `<empty>` | Polyglot `.bin` opening book; moves are picked at random, weighted by the book |
| `SeePruneDepth` | spin | 3 | Tuning: max depth for SEE pruning (0 disables it) |
| `SeeNoisyMargin` | spin | 120 | Tuning: SEE pruning margin for captures/promotions, cp per ply (depth ≤ 3) |
| `SeeQuietMargin` | spin | 60 | Tuning: SEE pruning margin for quiet moves, cp per ply; 1000 disables it |

### Commands

All standard commands are supported: `uci`, `isready`, `ucinewgame`,
`position startpos|fen ... [moves ...]`, `go`, `stop`, `ponderhit`,
`setoption`, `quit`.

`go` accepts `wtime` / `btime` / `winc` / `binc` / `movestogo`, `movetime`,
`depth`, `nodes`, `infinite` and `ponder`.

Debug commands (non-standard):

| Command | Output |
|---|---|
| `d` | Print the board |
| `fen` | Print the current FEN |
| `eval` | Static NNUE evaluation, side-to-move relative |
| `perft N` | Perft divide to depth N |

### Info output

Each completed iteration prints `depth`, `score` (`cp` or `mate`), `nodes`,
`nps`, `hashfull`, `tbhits`, `time` and `pv`. Nodes, nps and tbhits are totals
across all search threads.

### Pondering

`bestmove` includes `ponder <move>` when the PV predicts a reply.

- **Standard `go ponder`:** the engine searches until `ponderhit` (switching
  to the normal time limits from that moment) or `stop` (the ponder missed).
  `bestmove` is printed in both cases, as UCI requires.
- **Autonomous pondering:** some controllers never send `go ponder`, for
  example ICS bridges that only relay `position` / `go`. For those, when
  `Ponder` is enabled the engine keeps searching the predicted position on all
  threads after it moves. The next command (other than `isready`) stops it,
  and the warmed TT carries into the real search.

`setoption name Ponder value false` turns off both kinds of pondering.

### Time management

| Parameter | Value |
|---|---|
| Allocation | `(time_left + (increment − overhead) × (movestogo − 1)) / movestogo` (movestogo default 40), capped at `time_left / 5` |
| Soft limit | `0.6 × allocation`, scaled 0.8–1.2× by how far the search score is from the static eval; checked between iterations |
| Hard limit | `2.5 × allocation`, capped at `time_left / 3`; aborts mid-iteration |
| Overhead | `Move Overhead` (default 30 ms) is taken off `time_left` for this move and off the increment for every move after it |

`time_left` here is the clock after the overhead is taken off. When
`movestogo` is below 5, both caps are `time_left / 2` instead. The clock
runs from when `position` (or `go`, if something came in between) arrives,
so the engine's own setup time counts against the move.

`movetime` sets both limits. `infinite`, `depth` and `nodes` searches run
without a clock.

---

## Command-line flags

| Flag | Effect |
|---|---|
| `--threads N` | Starting `Threads` value |
| `--hash MB` | Starting `Hash` value |
| `--syzygy-path PATH` | Load tablebases at startup (same as `SyzygyPath`) |
| `--eval-file PATH` | Load a net from disk instead of the embedded one (diagnostic) |
| `--probe-fen "FEN"` | Print the NNUE eval of one position and exit |
| `--probe FILE [--probe-threads N]` | Batch-evaluate a FEN-per-line file and report throughput |
| `--probe-symmetry "FEN" "MIRROR"` | Evaluate a position and its color-flipped mirror |

Flags also accept the `--flag=value` form.

### Training data generation

```
erebus --datagen out.binpack --games 100000 --threads 4 --nodes 5000
```

Plays self-play games from the start position and writes them as Stockfish
binpacks (one `out.t<N>.binpack` per thread, never overwriting an existing
file). Each game is one binpack chain: book moves, a few random plies, then
searched moves. Every position is searched, so book and random plies get real
scores; the stored move is the one played.

| Flag | Default | Effect |
|---|---|---|
| `--games N` | 10000 | Games to play |
| `--nodes N` | 5000 | Soft node budget per move, jittered to N–1.5N |
| `--book FILE` | `books/TCEC26.bin` | Polyglot opening book (`none` to disable) |
| `--book-uniform` | off | Pick book moves uniformly instead of by weight |
| `--random-plies N` | 4 | Random legal moves after leaving the book |
| `--seed N` | clock | RNG seed (thread N uses a derived seed) |
| `--hash MB` | 16 | Transposition table per thread |
| `--syzygy-path PATH` | | End games on a tablebase result |
| `--eval-file PATH` | | Generate with a different net |

Repeated openings are rejected, as are openings scoring beyond ±1000.
Adjudication: win at |score| ≥ 2500 for 4 plies; draw from ply 80 at
|score| ≤ 10 for 8 plies; plus mate, stalemate, repetition, 50-move rule,
insufficient material and tablebases. Scores are clamped to ±32000, so mate
and TB scores fall outside the trainer's |score| ≤ 10000 filter.

---

## Source layout

| File | Purpose |
|---|---|
| `main.rs` | Entry point, CLI flags |
| `uci.rs` | UCI loop, time allocation, Lazy SMP and ponder thread management |
| `search.rs` | Iterative deepening, PVS, quiescence, move ordering, pruning |
| `nnue.rs` | Net loading, accumulators, SIMD forward pass |
| `eval.rs` | Evaluation entry point and score constants |
| `board.rs` | Position state, FEN, make/unmake with incremental NNUE updates |
| `movegen.rs` | Legal move generation (pin/check-aware) and perft |
| `moves.rs` | Move encoding, flags, `MoveList` |
| `bitboard.rs` | Masks, magic bitboards, between/line tables |
| `see.rs` | Static exchange evaluation |
| `tt.rs` | Transposition table |
| `syzygy.rs` | Tablebase probing via `shakmaty-syzygy` |
| `book.rs` | Polyglot opening book probing (`OwnBook`, datagen) |
| `datagen.rs` | Self-play training data generation (binpacks via `sfbinpack`) |
| `probe.rs` | Standalone NNUE probe modes |
| `zobrist.rs` | Compile-time Zobrist keys |

---

## Testing

```bash
cargo test --release
```

The suite covers perft on the standard positions, the legal generator against
a make/unmake reference (move set and order), SIMD-vs-scalar NNUE
equivalence, net sanity checks, and basic tactics.

## License

GPL-3.0; see [LICENSE](LICENSE).
