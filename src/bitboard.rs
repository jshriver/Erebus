//! Bitboard primitives: masks, square constants, between/line tables, magic
//! slider attacks and the leaper attack tables.

/// A bitboard is a 64-bit integer where each bit represents a square.
/// Bit 0 = a1, Bit 7 = h1, Bit 56 = a8, Bit 63 = h8.
pub type Bitboard = u64;

pub const FILE_A: Bitboard = 0x0101010101010101;
pub const FILE_B: Bitboard = FILE_A << 1;
pub const FILE_G: Bitboard = FILE_A << 6;
pub const FILE_H: Bitboard = FILE_A << 7;

pub const RANK_1: Bitboard = 0xFF;
pub const RANK_3: Bitboard = RANK_1 << 16;
pub const RANK_6: Bitboard = RANK_1 << 40;
pub const RANK_8: Bitboard = RANK_1 << 56;

pub const NOT_FILE_A:  Bitboard = !FILE_A;
pub const NOT_FILE_H:  Bitboard = !FILE_H;
pub const NOT_FILE_AB: Bitboard = !(FILE_A | FILE_B);
pub const NOT_FILE_GH: Bitboard = !(FILE_G | FILE_H);

#[rustfmt::skip]
pub mod sq {
    pub const A1:u8=0;  pub const B1:u8=1;  pub const C1:u8=2;  pub const D1:u8=3;
    pub const E1:u8=4;  pub const F1:u8=5;  pub const G1:u8=6;  pub const H1:u8=7;
    pub const A8:u8=56; pub const B8:u8=57; pub const C8:u8=58; pub const D8:u8=59;
    pub const E8:u8=60; pub const F8:u8=61; pub const G8:u8=62; pub const H8:u8=63;
}

#[inline(always)] pub fn bit(sq: u8) -> Bitboard { 1u64 << sq }

use std::sync::LazyLock;

static BETWEEN: LazyLock<[[Bitboard; 64]; 64]> = LazyLock::new(|| build_ray_table(false));
static LINE:    LazyLock<[[Bitboard; 64]; 64]> = LazyLock::new(|| build_ray_table(true));

#[inline] pub fn between(a: u8, b: u8) -> Bitboard { BETWEEN[a as usize][b as usize] }
#[inline] pub fn line(a: u8, b: u8) -> Bitboard { LINE[a as usize][b as usize] }

fn build_ray_table(full_line: bool) -> [[Bitboard; 64]; 64] {
    let mut t = [[0u64; 64]; 64];
    for a in 0i32..64 {
        let (ar, af) = (a / 8, a % 8);
        for b in 0i32..64 {
            if a == b { continue; }
            let (br, bf) = (b / 8, b % 8);
            let aligned = ar == br || af == bf || (ar - br).abs() == (af - bf).abs();
            if !aligned { continue; }
            let dr = (br - ar).signum();
            let df = (bf - af).signum();

            let mut bb = 0u64;

            let (mut r, mut f) = (ar + dr, af + df);
            while (r, f) != (br, bf) {
                bb |= 1u64 << (r * 8 + f);
                r += dr;
                f += df;
            }
            if full_line {
                bb |= (1u64 << a) | (1u64 << b);
                for sign in [1i32, -1] {
                    let (mut r, mut f) = (ar + dr * sign, af + df * sign);
                    while (0..8).contains(&r) && (0..8).contains(&f) {
                        bb |= 1u64 << (r * 8 + f);
                        r += dr * sign;
                        f += df * sign;
                    }
                }
            }
            t[a as usize][b as usize] = bb;
        }
    }
    t
}

#[inline(always)]
pub fn pop_lsb(bb: &mut Bitboard) -> u8 {
    let sq = bb.trailing_zeros() as u8;
    *bb &= *bb - 1;
    sq
}

#[inline(always)] pub fn lsb(bb: Bitboard) -> u8 { bb.trailing_zeros() as u8 }

pub fn sq_to_str(sq: u8) -> String {
    format!("{}{}", (b'a' + (sq & 7)) as char, (b'1' + (sq >> 3)) as char)
}

pub fn str_to_sq(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() < 2 { return None; }
    let file = b[0].wrapping_sub(b'a');
    let rank = b[1].wrapping_sub(b'1');
    if file > 7 || rank > 7 { return None; }
    Some(rank * 8 + file)
}

#[rustfmt::skip]
const ROOK_MAGICS: [u64; 64] = [
    0x8a80104000800020,0x140002000100040, 0x2801880a0017001, 0x100081001000420,
    0x200020010080420, 0x3001c0002010008, 0x8480008002000100,0x2080088004402900,
    0x800098204000,    0x2024401000200040,0x100802000801000, 0x120800800801000,
    0x208808088000400, 0x2802200800400,   0x2200800100020080,0x801000060821100,
    0x80044006422000,  0x100808020004000, 0x12108a0010204200,0x140848010000802,
    0x481828014002800, 0x8094004002004100,0x4010040010010802,0x20008806104,
    0x100400080208000, 0x2040002120081000,0x21200680100081,  0x20100080080080,
    0x2000a00200410,   0x20080800400,     0x80088400100102,  0x80004600042881,
    0x4040008040800020,0x440003000200801, 0x4200011004500,   0x188020010100100,
    0x14800401802800,  0x2080040080800200,0x124080204001001, 0x200046502000484,
    0x480400080088020, 0x1000422010034000,0x30200100110040,  0x100021010009,
    0x2002080100110004,0x202008004008002, 0x20020004010100,  0x2048440040820001,
    0x101002200408200, 0x40802000401080,  0x4008142004410100,0x2060820c0120200,
    0x1001004080100,   0x20c020080040080, 0x2935610830022400,0x44440041009200,
    0x280001040802101, 0x2100190040002085,0x80c0084100102001,0x4024081001000421,
    0x20030a0244872,   0x12001008414402,  0x2006104900a0804, 0x1004081002402,
];

#[rustfmt::skip]
const BISHOP_MAGICS: [u64; 64] = [
    0x40040844404084,  0x2004208a004208,  0x10190041080202,  0x108060845042010,
    0x581104180800210, 0x2112080446200010,0x1080820820060210,0x3c0808410220200,
    0x4050404440404,   0x21001420088,     0x24d0080801082102,0x1020a0a020400,
    0x40308200402,     0x4011002100800,   0x401484104104005, 0x801010402020200,
    0x400210c3880100,  0x404022024108200, 0x810018200204102, 0x4002801a02003,
    0x85040820080400,  0x810102c808880400,0xe900410884800,   0x8002020480840102,
    0x220200865090201, 0x2010100a02021202,0x152048408022401, 0x20080002081110,
    0x4001001021004000,0x800040400a011002,0xe4004081011002,  0x1c004001012080,
    0x8004200962a00220,0x8422100208500202,0x2000402200300c08,0x8646020080080080,
    0x80020a0200100808,0x2010004880111000,0x623000a080011400,0x42008c0340209202,
    0x209188240001000, 0x400408a884001800,0x110400a6080400,  0x1840060a44020800,
    0x90080104000041,  0x201011000808101, 0x1a2208080504f080,0x8012020600211212,
    0x500861011240000, 0x180806108200800, 0x4000020e01040044,0x300000261044000a,
    0x802241102020002, 0x20906061210001,  0x5a84841004010310,0x4010801011c04,
    0xa010109502200,   0x4a02012000,      0x500201010098b028,0x8040002811040900,
    0x28000010020204,  0x6000020202d0240, 0x8918844842082200,0x4010011029020020,
];

#[rustfmt::skip]
const ROOK_SHIFTS: [u32; 64] = [
    52,53,53,53,53,53,53,52,
    53,54,54,54,54,54,54,53,
    53,54,54,54,54,54,54,53,
    53,54,54,54,54,54,54,53,
    53,54,54,54,54,54,54,53,
    53,54,54,54,54,54,54,53,
    53,54,54,54,54,54,54,53,
    52,53,53,53,53,53,53,52,
];

#[rustfmt::skip]
const BISHOP_SHIFTS: [u32; 64] = [
    58,59,59,59,59,59,59,58,
    59,59,59,59,59,59,59,59,
    59,59,57,57,57,57,59,59,
    59,59,57,55,55,57,59,59,
    59,59,57,55,55,57,59,59,
    59,59,57,57,57,57,59,59,
    59,59,59,59,59,59,59,59,
    58,59,59,59,59,59,59,58,
];

const ROOK_TABLE_SIZE:   usize = 0x19000;
const BISHOP_TABLE_SIZE: usize = 0x1480;

/// Process-wide slider attack tables, computed entirely at compile time: a
/// plain static with no lazy-init check on the hot path.
pub static MAGICS: Magics = Magics::new();

pub struct Magics {
    rook_masks:     [Bitboard; 64],
    bishop_masks:   [Bitboard; 64],
    rook_offsets:   [usize; 64],
    bishop_offsets: [usize; 64],
    rook_table:     [Bitboard; ROOK_TABLE_SIZE],
    bishop_table:   [Bitboard; BISHOP_TABLE_SIZE],
}

impl Magics {
    const fn new() -> Self {
        let mut m = Magics {
            rook_masks:     [0; 64],
            bishop_masks:   [0; 64],
            rook_offsets:   [0; 64],
            bishop_offsets: [0; 64],
            rook_table:     [0; ROOK_TABLE_SIZE],
            bishop_table:   [0; BISHOP_TABLE_SIZE],
        };

        let mut offset = 0usize;
        let mut sq = 0usize;
        while sq < 64 {
            let mask  = rook_mask(sq as u8);
            let shift = ROOK_SHIFTS[sq];
            m.rook_masks[sq]   = mask;
            m.rook_offsets[sq] = offset;
            let mut occ = 0u64;
            loop {
                let idx = magic_index(occ, ROOK_MAGICS[sq], shift);
                m.rook_table[offset + idx] = slow_rook(sq as u8, occ);
                occ = occ.wrapping_sub(mask) & mask;
                if occ == 0 { break; }
            }
            offset += 1usize << (64 - shift);
            sq += 1;
        }

        let mut offset = 0usize;
        let mut sq = 0usize;
        while sq < 64 {
            let mask  = bishop_mask(sq as u8);
            let shift = BISHOP_SHIFTS[sq];
            m.bishop_masks[sq]   = mask;
            m.bishop_offsets[sq] = offset;
            let mut occ = 0u64;
            loop {
                let idx = magic_index(occ, BISHOP_MAGICS[sq], shift);
                m.bishop_table[offset + idx] = slow_bishop(sq as u8, occ);
                occ = occ.wrapping_sub(mask) & mask;
                if occ == 0 { break; }
            }
            offset += 1usize << (64 - shift);
            sq += 1;
        }
        m
    }

    #[inline(always)]
    pub fn rook_attacks(&self, sq: u8, occ: Bitboard) -> Bitboard {
        let s   = sq as usize;
        let idx = magic_index(occ & self.rook_masks[s], ROOK_MAGICS[s], ROOK_SHIFTS[s]);
        self.rook_table[self.rook_offsets[s] + idx]
    }

    #[inline(always)]
    pub fn bishop_attacks(&self, sq: u8, occ: Bitboard) -> Bitboard {
        let s   = sq as usize;
        let idx = magic_index(occ & self.bishop_masks[s], BISHOP_MAGICS[s], BISHOP_SHIFTS[s]);
        self.bishop_table[self.bishop_offsets[s] + idx]
    }

    #[inline(always)]
    pub fn queen_attacks(&self, sq: u8, occ: Bitboard) -> Bitboard {
        self.rook_attacks(sq, occ) | self.bishop_attacks(sq, occ)
    }
}

#[inline(always)]
const fn magic_index(occ: Bitboard, magic: u64, shift: u32) -> usize {
    (occ.wrapping_mul(magic) >> shift) as usize
}

const fn rook_mask(sq: u8) -> Bitboard {
    let r = (sq >> 3) as i32;
    let f = (sq & 7) as i32;
    let mut mask = 0u64;
    let mut i = r + 1; while i < 7 { mask |= 1u64 << (i * 8 + f); i += 1; }
    let mut i = 1;     while i < r { mask |= 1u64 << (i * 8 + f); i += 1; }
    let mut i = f + 1; while i < 7 { mask |= 1u64 << (r * 8 + i); i += 1; }
    let mut i = 1;     while i < f { mask |= 1u64 << (r * 8 + i); i += 1; }
    mask
}

const fn bishop_mask(sq: u8) -> Bitboard {
    let r = (sq >> 3) as i32;
    let f = (sq & 7) as i32;
    let mut mask = 0u64;
    let (mut r2,mut f2)=(r+1,f+1); while r2<7&&f2<7 { mask|=1u64<<(r2*8+f2); r2+=1;f2+=1; }
    let (mut r2,mut f2)=(r+1,f-1); while r2<7&&f2>0 { mask|=1u64<<(r2*8+f2); r2+=1;f2-=1; }
    let (mut r2,mut f2)=(r-1,f+1); while r2>0&&f2<7 { mask|=1u64<<(r2*8+f2); r2-=1;f2+=1; }
    let (mut r2,mut f2)=(r-1,f-1); while r2>0&&f2>0 { mask|=1u64<<(r2*8+f2); r2-=1;f2-=1; }
    mask
}

const fn slow_ray(sq: u8, occ: Bitboard, dirs: [(i32, i32); 4]) -> Bitboard {
    let r = (sq >> 3) as i32;
    let f = (sq & 7) as i32;
    let mut attacks = 0u64;
    let mut d = 0;
    while d < 4 {
        let (dr, df) = dirs[d];
        let (mut r2, mut f2) = (r + dr, f + df);
        while r2 >= 0 && r2 < 8 && f2 >= 0 && f2 < 8 {
            let s = 1u64 << (r2 * 8 + f2);
            attacks |= s;
            if occ & s != 0 { break; }
            r2 += dr; f2 += df;
        }
        d += 1;
    }
    attacks
}

const fn slow_rook(sq: u8, occ: Bitboard) -> Bitboard {
    slow_ray(sq, occ, [(1, 0), (-1, 0), (0, 1), (0, -1)])
}

const fn slow_bishop(sq: u8, occ: Bitboard) -> Bitboard {
    slow_ray(sq, occ, [(1, 1), (1, -1), (-1, 1), (-1, -1)])
}

pub static KNIGHT_ATTACKS: [Bitboard; 64] = knight_attacks_table();
pub static KING_ATTACKS:   [Bitboard; 64] = king_attacks_table();

const fn knight_attacks_table() -> [Bitboard; 64] {
    let mut t = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let b = 1u64 << sq;
        t[sq] =
            ((b << 17) & NOT_FILE_A)  | ((b << 15) & NOT_FILE_H)
          | ((b << 10) & NOT_FILE_AB) | ((b <<  6) & NOT_FILE_GH)
          | ((b >> 17) & NOT_FILE_H)  | ((b >> 15) & NOT_FILE_A)
          | ((b >> 10) & NOT_FILE_GH) | ((b >>  6) & NOT_FILE_AB);
        sq += 1;
    }
    t
}

const fn king_attacks_table() -> [Bitboard; 64] {
    let mut t = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let b = 1u64 << sq;
        t[sq] =
            ((b << 1) & NOT_FILE_A) | ((b >> 1) & NOT_FILE_H)
          | (b << 8)  | (b >> 8)
          | ((b << 9) & NOT_FILE_A) | ((b >> 9) & NOT_FILE_H)
          | ((b << 7) & NOT_FILE_H) | ((b >> 7) & NOT_FILE_A);
        sq += 1;
    }
    t
}
