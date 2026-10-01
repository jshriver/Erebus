//! nnue: incremental NNUE evaluation.
//!
//! ============================== ARCHITECTURE ==============================
//! A single perspective-pair feature transformer with one hidden layer and
//! a scalar output.
//!
//!     (768 -> HL) x 2  ->  1
//!
//!   * 768 inputs      = 2 colors * 6 piece types * 64 squares
//!   * HL              = hidden / accumulator width per perspective
//!   * SCReLU          = clipped-then-squared activation, clamp to [0, QA]
//!   * QA = 255, QB = 64, SCALE = 400  (bullet's standard quantisation)
//!
//! Two fixed-perspective accumulators are maintained on `Board`: a
//! White-perspective accumulator and a Black-perspective accumulator.
//! Whichever color is on move supplies the "stm" half of the network input,
//! the other the "ntm" half. Which accumulator IS White's / Black's never
//! depends on whose turn it is, so `Board::make_move` updates them
//! incrementally:
//!
//!   - Any move changes at most 4 (piece, color, square) features, so we
//!     toggle just those in/out of both accumulators -- O(1) per move.
//!   - `unmake_move` replays the same toggles inverted.
//!
//! ============================== NET SOURCE ==============================
//! The engine ships with a network embedded into the executable via
//! `include_bytes!` (see `EMBEDDED_NET`), loaded once at startup in
//! `main()`. `nets/net.nnue` is a `bullet` export for exactly this
//! architecture: `(768 x HL) i16` feature-transformer weights (feature-major
//! -- each feature's HL weights contiguous), `HL i16` feature-transformer
//! biases, `2*HL i16` output weights (first HL = stm perspective, next HL =
//! ntm perspective), `1 i16` output bias, then trailing `"bullet"` padding
//! which is ignored.
//!
//! `load` (runtime, from a file path) exists as a diagnostic override,
//! reachable via `--eval-file` or `setoption name EvalFile value <path>`.
//! Callers of `load` must re-snapshot any live board afterward (see
//! `Board::reload_net`).

use std::fs;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use crate::board::{Board, Color, Piece};

/// Hidden layer / accumulator width per perspective.
pub const HL: usize = 1024;

/// Number of raw board features: 2 colors * 6 pieces * 64 squares.
const INPUT_SIZE: usize = 768;

const QA: i64 = 255;
const QB: i64 = 64;
const SCALE: i64 = 400;

const EMBEDDED_NET: &[u8] = include_bytes!("../nets/net.nnue");

pub struct Net {
    /// Feature-transformer weights, feature-major: `l0w[feature * HL + h]`.
    l0w: Vec<i16>,
    /// Feature-transformer biases: `l0b[h]`.
    l0b: Vec<i16>,
    /// Output weights: `l1w[0..HL]` applied to the stm accumulator,
    /// `l1w[HL..2*HL]` applied to the ntm accumulator.
    l1w: Vec<i16>,
    /// Scalar output bias (raw quantised i16 units).
    l1b: i32,
}

/// The active net, behind an `Arc` so a `Board` can take a cheap owning
/// snapshot once (at construction / after an EvalFile swap) and then read
/// it with zero synchronisation on every eval and every move. The `RwLock`
/// is only touched when installing a net or taking a snapshot, never in the
/// search hot loop.
static NET: RwLock<Option<Arc<Net>>> = RwLock::new(None);

/// Install the network embedded in the executable as the active net.
/// Must be called once at process startup, before any `Board` is
/// constructed. Panics on failure -- a corrupt embedded net means the
/// binary was built wrong, and there is no classical eval to fall back to.
pub fn load_embedded() {
    let net = parse_net(EMBEDDED_NET)
        .unwrap_or_else(|e| panic!("embedded NNUE net failed to parse: {e}"));
    *NET.write().unwrap() = Some(Arc::new(net));
}

/// Diagnostic / experimentation override: load a net of the *same*
/// architecture from a file on disk at runtime (`--eval-file` /
/// `setoption name EvalFile`). Does NOT refresh any live board's
/// accumulators -- the caller (the `EvalFile` setoption handler in `uci.rs`)
/// must call `Board::reload_net` so the board picks up the new snapshot.
pub fn load(path: &str) -> Result<(), String> {
    let buf = fs::read(path).map_err(|e| format!("reading '{path}': {e}"))?;
    let net = parse_net(&buf)?;
    *NET.write().unwrap() = Some(Arc::new(net));
    Ok(())
}

/// Take an owning snapshot of the current net (one `RwLock` read + one
/// refcount bump). `None` if no net has been installed yet.
pub fn current_net() -> Option<Arc<Net>> {
    NET.read().unwrap().clone()
}

/// Acquire a read lock on the net for a caller-controlled scope. Used only
/// by the standalone `probe` batch modes, which hold it once per worker
/// thread across a whole chunk; the search never calls this.
pub fn read_lock() -> RwLockReadGuard<'static, Option<Arc<Net>>> {
    NET.read().unwrap()
}

// ================= Loading =================

fn read_i16_block(buf: &[u8], offset: &mut usize, n: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(i16::from_le_bytes([buf[*offset], buf[*offset + 1]]));
        *offset += 2;
    }
    out
}

/// Parse a `bullet` export for the `(768 -> HL) x 2 -> 1` SCReLU net.
///
/// Layout (little-endian i16 throughout), matching `chal.c`'s `net.h`:
///   l0w : INPUT_SIZE * HL   feature-major
///   l0b : HL
///   l1w : 2 * HL            (stm half, then ntm half)
///   l1b : 1
///   ...  trailing "bullet" padding, ignored.
fn parse_net(buf: &[u8]) -> Result<Net, String> {
    let n_l0w = INPUT_SIZE * HL;
    let n_l0b = HL;
    let n_l1w = 2 * HL;
    let expected = (n_l0w + n_l0b + n_l1w + 1) * 2;

    if buf.len() < expected {
        return Err(format!(
            "net data is {} bytes, expected at least {expected} for a (768 -> {HL}) x 2 -> 1 \
             i16 net. Either HL is wrong for this file, or this is not a bullet export for \
             this architecture.",
            buf.len()
        ));
    }

    let mut off = 0usize;
    let l0w = read_i16_block(buf, &mut off, n_l0w);
    let l0b = read_i16_block(buf, &mut off, n_l0b);
    let l1w = read_i16_block(buf, &mut off, n_l1w);
    let l1b = read_i16_block(buf, &mut off, 1)[0] as i32;

    let max_l1w = l1w.iter().map(|&x| (x as i32).abs()).max().unwrap_or(0);
    if QA as i32 * max_l1w > i16::MAX as i32 {
        return Err(format!(
            "l1w |max| = {max_l1w}: QA({QA}) * {max_l1w} = {} exceeds i16 range, so the fused \
             i16 SCReLU would truncate. Retrain with tighter output-weight quantisation \
             (keep |l1w| <= {}).",
            QA as i32 * max_l1w,
            i16::MAX as i32 / QA as i32,
        ));
    }

    let nz = l0w.iter().filter(|&&x| x != 0).count();
    let nz_frac = nz as f64 / l0w.len().max(1) as f64;
    if nz_frac < 0.01 {
        return Err(format!(
            "feature-transformer weights are {:.3}% non-zero ({nz}/{} values) -- this is not a \
             trained net. Every position would evaluate to the same constant.",
            nz_frac * 100.0,
            l0w.len(),
        ));
    }

    Ok(Net { l0w, l0b, l1w, l1b })
}

/// L0 feature index for one (piece, color, square) as seen from
/// `perspective`.
///
///   White perspective: index = 384*color        + 64*piece + sq
///   Black perspective: index = 384*(color ^ 1)   + 64*piece + (sq ^ 56)
#[inline(always)]
fn feature_index(perspective: Color, piece: Piece, piece_color: Color, sq: u8) -> usize {
    let (rel_color, rel_sq) = match perspective {
        Color::White => (piece_color as usize, sq as usize),
        Color::Black => ((piece_color as usize) ^ 1, (sq ^ 56) as usize),
    };
    384 * rel_color + 64 * (piece as usize) + rel_sq
}

#[inline(always)]
fn weight_row(net: &Net, feature: usize) -> &[i16] {
    let base = feature * HL;
    &net.l0w[base..base + HL]
}

/// One perspective's accumulator: feature-transformer bias plus the sum of
/// the weight rows of every active feature. Kept in **i16** -- half the
/// memory traffic of i32 per toggle and per eval load, and it lets the
/// forward pass fuse SCReLU square + output-weight multiply into a single
/// `madd_epi16`. A bullet-quantised net keeps every hidden unit well inside
/// i16 range (32 pieces * max|l0w|); `refresh_one` debug-asserts this.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Accumulator {
    pub v: [i16; HL],
}

impl Accumulator {
    pub fn zeroed() -> Self {
        Accumulator { v: [0; HL] }
    }
}

/// Both perspectives, fixed by color (NOT by side-to-move).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DualAccumulator {
    pub white: Accumulator,
    pub black: Accumulator,
}

impl DualAccumulator {
    pub fn zeroed() -> Self {
        DualAccumulator { white: Accumulator::zeroed(), black: Accumulator::zeroed() }
    }
}

const QA_I16: i16 = QA as i16;

/// Clamped-square-times-weight for one lane, matching the SIMD `mullo_epi16`
/// (`c*w` truncates to i16 -- unreachable for a `parse_net`-validated net).
#[inline(always)]
fn screlu_w(acc: i16, w: i16) -> i64 {
    let c = acc.clamp(0, QA_I16);
    let cw = c.wrapping_mul(w);
    (c as i64) * (cw as i64)
}

#[inline]
fn acc_add(acc: &mut [i16], row: &[i16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return acc_addsub_avx2(acc, row, false) }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { return acc_addsub_avx2(acc, row, false) }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe { return acc_addsub_neon(acc, row, false) }
    #[allow(unreachable_code)]
    acc_addsub_scalar(acc, row, false)
}

#[cfg(test)]
#[inline]
fn acc_sub(acc: &mut [i16], row: &[i16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return acc_addsub_avx2(acc, row, true) }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { return acc_addsub_avx2(acc, row, true) }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe { return acc_addsub_neon(acc, row, true) }
    #[allow(unreachable_code)]
    acc_addsub_scalar(acc, row, true)
}

#[inline]
fn acc_set(acc: &mut [i16], bias: &[i16]) {
    let n = acc.len().min(bias.len());
    acc[..n].copy_from_slice(&bias[..n]);
}

/// `dst = src` with every `(row, add)` applied, in one fused pass: a lazy
/// accumulator step reads the parent and writes the child exactly once.
#[inline]
fn acc_update(dst: &mut [i16], src: &[i16], rows: &[(&[i16], bool)]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return acc_update_avx2(dst, src, rows) }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { return acc_update_avx2(dst, src, rows) }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe { return acc_update_neon(dst, src, rows) }
    #[allow(unreachable_code)]
    acc_update_scalar(dst, src, rows)
}

#[inline]
fn screlu_dot(acc: &[i16], w: &[i16]) -> i64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return screlu_dot_avx2(acc, w) }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { return screlu_dot_avx2(acc, w) }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe { return screlu_dot_neon(acc, w) }
    #[allow(unreachable_code)]
    screlu_dot_scalar(acc, w)
}

fn acc_addsub_scalar(acc: &mut [i16], row: &[i16], sub: bool) {
    let n = acc.len().min(row.len());
    if sub {
        for h in 0..n { acc[h] = acc[h].wrapping_sub(row[h]); }
    } else {
        for h in 0..n { acc[h] = acc[h].wrapping_add(row[h]); }
    }
}

/// Shortest length among `dst`, `src` and every row: the span all kernels
/// may touch.
#[inline]
fn update_len(dst: &[i16], src: &[i16], rows: &[(&[i16], bool)]) -> usize {
    rows.iter().fold(dst.len().min(src.len()), |n, (row, _)| n.min(row.len()))
}

fn acc_update_scalar(dst: &mut [i16], src: &[i16], rows: &[(&[i16], bool)]) {
    let n = update_len(dst, src, rows);
    dst[..n].copy_from_slice(&src[..n]);
    for &(row, add) in rows {
        acc_addsub_scalar(&mut dst[..n], &row[..n], !add);
    }
}

fn screlu_dot_scalar(acc: &[i16], w: &[i16]) -> i64 {
    let n = acc.len().min(w.len());
    let mut sum: i64 = 0;
    for h in 0..n {
        sum += screlu_w(acc[h], w[h]);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn acc_addsub_avx2(acc: &mut [i16], row: &[i16], sub: bool) {
    use std::arch::x86_64::*;
    let n = acc.len().min(row.len());
    let mut h = 0;
    while h + 16 <= n {
        let a = _mm256_loadu_si256(acc.as_ptr().add(h) as *const __m256i);
        let r = _mm256_loadu_si256(row.as_ptr().add(h) as *const __m256i);
        let s = if sub { _mm256_sub_epi16(a, r) } else { _mm256_add_epi16(a, r) };
        _mm256_storeu_si256(acc.as_mut_ptr().add(h) as *mut __m256i, s);
        h += 16;
    }
    if sub {
        while h < n { acc[h] = acc[h].wrapping_sub(row[h]); h += 1; }
    } else {
        while h < n { acc[h] = acc[h].wrapping_add(row[h]); h += 1; }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn acc_update_avx2(dst: &mut [i16], src: &[i16], rows: &[(&[i16], bool)]) {
    use std::arch::x86_64::*;
    let n = update_len(dst, src, rows);
    let mut h = 0;
    while h + 16 <= n {
        let mut v = _mm256_loadu_si256(src.as_ptr().add(h) as *const __m256i);
        for &(row, add) in rows {
            let r = _mm256_loadu_si256(row.as_ptr().add(h) as *const __m256i);
            v = if add { _mm256_add_epi16(v, r) } else { _mm256_sub_epi16(v, r) };
        }
        _mm256_storeu_si256(dst.as_mut_ptr().add(h) as *mut __m256i, v);
        h += 16;
    }
    while h < n {
        let mut v = src[h];
        for &(row, add) in rows {
            v = if add { v.wrapping_add(row[h]) } else { v.wrapping_sub(row[h]) };
        }
        dst[h] = v;
        h += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn screlu_dot_avx2(acc: &[i16], w: &[i16]) -> i64 {
    use std::arch::x86_64::*;
    let n = acc.len().min(w.len());
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA_I16);
    let mut sum = _mm256_setzero_si256();
    let mut h = 0;
    while h + 16 <= n {
        let a = _mm256_loadu_si256(acc.as_ptr().add(h) as *const __m256i);
        let c = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
        let wv = _mm256_loadu_si256(w.as_ptr().add(h) as *const __m256i);
        let cw = _mm256_mullo_epi16(c, wv);
        let prod = _mm256_madd_epi16(c, cw);

        sum = _mm256_add_epi64(sum, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod)));
        sum = _mm256_add_epi64(sum, _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(prod)));
        h += 16;
    }
    let lo = _mm256_castsi256_si128(sum);
    let hi = _mm256_extracti128_si256::<1>(sum);
    let s = _mm_add_epi64(lo, hi);
    let s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
    let mut total = _mm_cvtsi128_si64(s);
    while h < n {
        total += screlu_w(acc[h], w[h]);
        h += 1;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn acc_addsub_neon(acc: &mut [i16], row: &[i16], sub: bool) {
    use std::arch::aarch64::*;
    let n = acc.len().min(row.len());
    let mut h = 0;
    while h + 8 <= n {
        let a = vld1q_s16(acc.as_ptr().add(h));
        let r = vld1q_s16(row.as_ptr().add(h));
        let s = if sub { vsubq_s16(a, r) } else { vaddq_s16(a, r) };
        vst1q_s16(acc.as_mut_ptr().add(h), s);
        h += 8;
    }
    if sub {
        while h < n { acc[h] = acc[h].wrapping_sub(row[h]); h += 1; }
    } else {
        while h < n { acc[h] = acc[h].wrapping_add(row[h]); h += 1; }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn acc_update_neon(dst: &mut [i16], src: &[i16], rows: &[(&[i16], bool)]) {
    use std::arch::aarch64::*;
    let n = update_len(dst, src, rows);
    let mut h = 0;
    while h + 8 <= n {
        let mut v = vld1q_s16(src.as_ptr().add(h));
        for &(row, add) in rows {
            let r = vld1q_s16(row.as_ptr().add(h));
            v = if add { vaddq_s16(v, r) } else { vsubq_s16(v, r) };
        }
        vst1q_s16(dst.as_mut_ptr().add(h), v);
        h += 8;
    }
    while h < n {
        let mut v = src[h];
        for &(row, add) in rows {
            v = if add { v.wrapping_add(row[h]) } else { v.wrapping_sub(row[h]) };
        }
        dst[h] = v;
        h += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn screlu_dot_neon(acc: &[i16], w: &[i16]) -> i64 {
    use std::arch::aarch64::*;
    let n = acc.len().min(w.len());
    let zero = vdupq_n_s16(0);
    let qa = vdupq_n_s16(QA_I16);
    let mut sum = vdupq_n_s64(0);
    let mut h = 0;
    while h + 8 <= n {
        let a = vld1q_s16(acc.as_ptr().add(h));
        let c = vminq_s16(vmaxq_s16(a, zero), qa);
        let wv = vld1q_s16(w.as_ptr().add(h));
        let cw = vmulq_s16(c, wv);

        let p_lo = vmull_s16(vget_low_s16(c), vget_low_s16(cw));
        let p_hi = vmull_s16(vget_high_s16(c), vget_high_s16(cw));

        sum = vaddq_s64(sum, vaddl_s32(vget_low_s32(p_lo), vget_high_s32(p_lo)));
        sum = vaddq_s64(sum, vaddl_s32(vget_low_s32(p_hi), vget_high_s32(p_hi)));
        h += 8;
    }
    let mut total = vgetq_lane_s64(sum, 0) + vgetq_lane_s64(sum, 1);
    while h < n {
        total += screlu_w(acc[h], w[h]);
        h += 1;
    }
    total
}

/// Full from-scratch build of one perspective's accumulator from the board.
fn refresh_one(
    net: &Net,
    sq_piece: &[Option<(Piece, Color)>; 64],
    perspective: Color,
    acc: &mut Accumulator,
) {
    acc_set(&mut acc.v, &net.l0b);
    for (sq, slot) in sq_piece.iter().enumerate() {
        let Some((piece, color)) = *slot else { continue };
        let f = feature_index(perspective, piece, color, sq as u8);
        acc_add(&mut acc.v, weight_row(net, f));
    }

    #[cfg(debug_assertions)]
    {
        let mut wide = [0i32; HL];
        for h in 0..HL { wide[h] = net.l0b[h] as i32; }
        for (sq, slot) in sq_piece.iter().enumerate() {
            if let Some((piece, color)) = *slot {
                let f = feature_index(perspective, piece, color, sq as u8);
                let row = weight_row(net, f);
                for h in 0..HL { wide[h] += row[h] as i32; }
            }
        }
        for h in 0..HL {
            debug_assert!(
                wide[h] > i16::MIN as i32 + 256 && wide[h] < i16::MAX as i32 - 256,
                "NNUE hidden unit {h} = {} is within 256 of the i16 boundary -- this net is too \
                 hot for i16 accumulators", wide[h],
            );
            debug_assert_eq!(wide[h] as i16, acc.v[h], "i16 accumulator diverged at unit {h}");
        }
    }
}

/// Full from-scratch build of both perspectives' accumulators.
pub fn refresh_all(
    net: &Net,
    sq_piece: &[Option<(Piece, Color)>; 64],
    dual: &mut DualAccumulator,
) {
    refresh_one(net, sq_piece, Color::White, &mut dual.white);
    refresh_one(net, sq_piece, Color::Black, &mut dual.black);
}

/// One input-feature change: `(piece, color, square, added)`. A move makes
/// at most 4 (castling moves king + rook).
pub type FeatDelta = (Piece, Color, u8, bool);

/// `child = parent` with `feat` applied, for both perspectives.
fn update_from_parent(net: &Net, parent: &DualAccumulator, child: &mut DualAccumulator, feat: &[FeatDelta]) {
    for perspective in [Color::White, Color::Black] {
        let mut rows: [(&[i16], bool); 4] = [(&[], false); 4];
        for (slot, &(piece, color, sq, add)) in rows.iter_mut().zip(feat) {
            *slot = (weight_row(net, feature_index(perspective, piece, color, sq)), add);
        }
        let (src, dst) = match perspective {
            Color::White => (&parent.white.v, &mut child.white.v),
            Color::Black => (&parent.black.v, &mut child.black.v),
        };
        acc_update(dst, src, &rows[..feat.len()]);
    }
}

/// Entries in an `AccStack`. The search never goes deeper than its
/// `MAX_PLY` (128) below the root, and `Searcher::search` rebases the stack
/// at the root, so this is never exhausted mid-search.
const ACC_STACK_CAP: usize = 256;

/// Lazily-updated accumulators, one entry per `make_move` along the current
/// line. `make_move` only records the move's feature diff; `unmake_move`
/// steps back an entry. The accumulators are brought up to date on demand
/// (`current`) from the nearest computed ancestor, so positions that are
/// never evaluated -- TT cutoffs, TT-eval hits, pruned moves -- cost no
/// accumulator work at all.
#[derive(Clone)]
pub struct AccStack {
    accs:  Box<[DualAccumulator]>,
    /// `diffs[i]`: the feature changes taking entry `i - 1` to entry `i`.
    diffs: Box<[([FeatDelta; 4], u8)]>,
    /// Entry of the current position.
    top:   usize,
    /// Entries `0..=clean` hold computed accumulators; `clean < i <= top`
    /// are pending.
    clean: usize,
}

impl AccStack {
    pub fn new() -> Self {
        AccStack {
            accs:  vec![DualAccumulator::zeroed(); ACC_STACK_CAP].into_boxed_slice(),
            diffs: vec![([(Piece::Pawn, Color::White, 0, false); 4], 0); ACC_STACK_CAP].into_boxed_slice(),
            top:   0,
            clean: 0,
        }
    }

    /// Drop every entry and hand back entry 0 for the caller to fill with a
    /// full refresh of the current position.
    pub fn reset(&mut self) -> &mut DualAccumulator {
        self.top = 0;
        self.clean = 0;
        &mut self.accs[0]
    }

    /// Record a move's feature diff as a new pending entry. Returns `false`
    /// when the stack is full; the caller must then rebuild via `reset`.
    #[inline]
    pub fn push(&mut self, feat: &[FeatDelta]) -> bool {
        if self.top + 1 == ACC_STACK_CAP {
            return false;
        }
        self.top += 1;
        let slot = &mut self.diffs[self.top];
        slot.0[..feat.len()].copy_from_slice(feat);
        slot.1 = feat.len() as u8;
        true
    }

    /// Step back to the previous entry. Returns `false` when there is none
    /// (the stack was reset since the matching `push`); the caller must then
    /// rebuild via `reset`.
    #[inline]
    pub fn pop(&mut self) -> bool {
        if self.top == 0 {
            return false;
        }
        self.top -= 1;
        self.clean = self.clean.min(self.top);
        true
    }

    /// The current position's accumulators, applying any pending diffs.
    #[inline]
    pub fn current(&mut self, net: &Net) -> &DualAccumulator {
        while self.clean < self.top {
            let i = self.clean + 1;
            let (done, pending) = self.accs.split_at_mut(i);
            let (feat, n) = &self.diffs[i];
            update_from_parent(net, &done[i - 1], &mut pending[0], &feat[..*n as usize]);
            self.clean = i;
        }
        &self.accs[self.top]
    }
}

/// Forward pass from a pair of accumulators, relative to `stm`.
pub fn evaluate_with_net(net: &Net, acc: &DualAccumulator, stm: Color) -> i32 {
    let (us, them) = match stm {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    let sum: i64 = screlu_dot(&us.v, &net.l1w[..HL])
        + screlu_dot(&them.v, &net.l1w[HL..2 * HL]);

    let out = (sum / QA + net.l1b as i64) * SCALE / (QA * QB);
    out as i32
}

/// Engine-wide static evaluation entry point. Reads the board's own net
/// snapshot -- no synchronisation -- and brings the board's lazy
/// accumulators up to date first (hence `&mut`). Panics if no net is loaded,
/// which is structurally unreachable (main() installs one at startup, before
/// any `Board` is built).
pub fn evaluate(board: &mut Board) -> i32 {
    let stm = board.side;
    let (net, acc) = board.nnue_state().expect("NNUE net not loaded");
    evaluate_with_net(net, acc, stm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board_from(fen: &str) -> Board {
        Board::from_fen(fen).expect("valid fen")
    }

    #[test]
    fn embedded_net_parses() {

        assert!(parse_net(EMBEDDED_NET).is_ok());
    }

    /// The SIMD kernels must be bit-identical to the scalar reference.
    #[test]
    fn simd_matches_scalar() {

        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        for trial in 0..64 {

            let mut acc_ref = [0i16; HL];
            let mut acc_simd = [0i16; HL];
            for h in 0..HL {
                let v = ((next() % 4096) as i32 - 1024) as i16;
                acc_ref[h] = v;
                acc_simd[h] = v;
            }
            let row: Vec<i16> = (0..HL).map(|_| (next() % 512) as i16 - 256).collect();

            let w: Vec<i16> = (0..HL).map(|_| (next() % 255) as i16 - 127).collect();

            let sub = trial & 1 == 0;
            acc_addsub_scalar(&mut acc_ref, &row, sub);
            if sub { acc_sub(&mut acc_simd, &row) } else { acc_add(&mut acc_simd, &row) }
            assert_eq!(acc_ref, acc_simd, "acc_{} mismatch on trial {trial}", if sub {"sub"} else {"add"});

            let d_ref = screlu_dot_scalar(&acc_ref, &w);
            let d_simd = screlu_dot(&acc_ref, &w);
            assert_eq!(d_ref, d_simd, "screlu_dot mismatch on trial {trial}");
        }
    }

    /// End-to-end: the fused i16 forward pass equals a full-precision i64
    /// `c*c*w` reference. Holds bit-exactly for a `parse_net`-validated net
    /// (its l1w bound guarantees `c*w` never overflows i16).
    #[test]
    fn forward_pass_matches_scalar() {
        load_embedded();
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let mut b = board_from(fen);
            let stm = b.side;
            let (net, acc) = b.nnue_state().unwrap();
            let (us, them) = match stm {
                Color::White => (&acc.white, &acc.black),
                Color::Black => (&acc.black, &acc.white),
            };
            let hp = |acc: i16, w: i16| -> i64 {
                let c = acc.clamp(0, QA_I16) as i64;
                c * c * w as i64
            };
            let mut sum: i64 = 0;
            for h in 0..HL {
                sum += hp(us.v[h], net.l1w[h]);
                sum += hp(them.v[h], net.l1w[HL + h]);
            }
            let want = ((sum / QA + net.l1b as i64) * SCALE / (QA * QB)) as i32;
            assert_eq!(want, evaluate(&mut b), "forward pass mismatch on {fen}");
        }
    }

    /// The board's lazy accumulators must equal a from-scratch refresh.
    fn assert_acc_fresh(b: &mut Board) {
        let sq_piece = b.sq_piece;
        let (net, acc) = b.nnue_state().unwrap();
        let mut want = DualAccumulator::zeroed();
        refresh_all(net, &sq_piece, &mut want);
        assert!(*acc == want, "lazy accumulator differs from a full refresh");
    }

    /// Walk the move tree reading the accumulators only at some nodes, so
    /// pending chains of every length build up and are resolved both on the
    /// way down and after unmakes.
    fn lazy_walk(b: &mut Board, depth: u32, visited: &mut u64) {
        use crate::movegen::MoveGen;
        use crate::moves::MoveList;

        *visited += 1;
        if depth == 0 || visited.is_multiple_of(3) {
            assert_acc_fresh(b);
        }
        if depth == 0 {
            return;
        }
        let mut moves = MoveList::new();
        MoveGen::generate_all(b, &mut moves);
        for i in 0..moves.len() {
            b.make_move(moves[i]);
            lazy_walk(b, depth - 1, visited);
            b.unmake_move();
        }
        if visited.is_multiple_of(5) {
            assert_acc_fresh(b);
        }
    }

    #[test]
    fn lazy_accumulators_match_refresh() {
        load_embedded();
        for fen in [

            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",

            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ] {
            let mut b = board_from(fen);
            let mut visited = 0;
            lazy_walk(&mut b, 3, &mut visited);
            assert_acc_fresh(&mut b);
        }
    }

    /// A line longer than the stack rebuilds instead of overflowing, and
    /// unmaking back through the rebuild stays correct.
    #[test]
    fn lazy_accumulators_survive_a_full_stack() {
        load_embedded();
        let mut b = board_from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"];
        let plies = ACC_STACK_CAP + 41;
        for i in 0..plies {
            let mv = b.find_uci_move(shuffle[i % 4]).expect("legal shuffle move");
            b.make_move(mv);
            if i % 50 == 0 {
                assert_acc_fresh(&mut b);
            }
        }
        assert_acc_fresh(&mut b);
        for i in 0..plies {
            b.unmake_move();
            if i % 7 == 0 {
                assert_acc_fresh(&mut b);
            }
        }
        assert_acc_fresh(&mut b);
    }

    #[test]
    fn startpos_eval_is_small_and_symmetric_ish() {
        load_embedded();
        let mut start = board_from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let e = evaluate(&mut start);

        assert!(e.abs() < 150, "startpos eval {e} out of sane range");
    }

    #[test]
    fn massive_material_advantage_is_detected() {
        load_embedded();

        let mut w = board_from("4k3/8/8/8/8/8/8/QQQQK3 w - - 0 1");
        let e = evaluate(&mut w);
        assert!(e > 800, "QQQQ vs k should be winning for White, got {e}");

        let mut b = board_from("4k3/8/8/8/8/8/8/QQQQK3 b - - 0 1");
        let e = evaluate(&mut b);
        assert!(e < -800, "QQQQ vs k, Black to move, got {e}");
    }

    #[test]
    fn color_flip_symmetry() {
        load_embedded();
        let a = evaluate(&mut board_from("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"));
        let b = evaluate(&mut board_from("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 0 1"));

        assert!(a.abs() < 200 && b.abs() < 200, "symmetric position evals {a} / {b} not small");
    }
}
