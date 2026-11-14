// Block Game, Enes Altun, 2025, MIT License
// Antialiasing ideas after Tonny Espeset's Cacodemon (https://www.shadertoy.com/view/NXtXzj).

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

struct GameUniform {
    camera_height: f32,
    camera_angle: f32,
    camera_scale: f32,
    volume: f32,
    sample_offset: u32,
    samples_to_generate: u32,
    sample_rate: f32,
    // space held
    action: f32,
};
@group(1) @binding(1) var<uniform> u_game: GameUniform;

const TAU: f32 = 6.2831853;

// g2: mouse (0), fonts (1,2), audio (3), game state (4)
struct MouseUniform {
    position: vec2<f32>,
    click_position: vec2<f32>,
    wheel: vec2<f32>,
    buttons: vec2<u32>,
};
@group(2) @binding(0) var<uniform> u_mouse: MouseUniform;
struct FontUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    grid_size: vec2<f32>,
};
@group(2) @binding(1) var<uniform> font_texture_uniform: FontUniforms;
@group(2) @binding(2) var t_font_texture_atlas: texture_2d<f32>;
@group(2) @binding(3) var<storage, read_write> audio_buffer: array<f32>;
// game state (atomic buffer)
@group(2) @binding(4) var<storage, read_write> atm: array<atomic<u32>>;

@group(3) @binding(0) var t0: texture_2d<f32>; @group(3) @binding(1) var s0: sampler;
@group(3) @binding(2) var t1: texture_2d<f32>; @group(3) @binding(3) var s1: sampler;
@group(3) @binding(4) var t2: texture_2d<f32>; @group(3) @binding(5) var s2: sampler;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>;
fn G(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&atm[i])); }
fn S(i: u32, v: f32) { atomicStore(&atm[i], bitcast<u32>(v)); }

const FONT_SPACING: f32 = 2.0;

// render single character
fn ch(pp: vec2<f32>, pos: vec2<f32>, code: u32, size: f32) -> f32 {
    let char_size_pixels = vec2<f32>(size, size);
    let relative_pos = pp - pos;

    // Check bounds
    if (relative_pos.x < 0.0 || relative_pos.x >= char_size_pixels.x ||
        relative_pos.y < 0.0 || relative_pos.y >= char_size_pixels.y) {
        return 0.0;
    }

    // Calculate UV coordinates within the character cell
    let local_uv = relative_pos / char_size_pixels;

    // calc char pos in atlas grid (16x16)
    let grid_x = code % 16u;
    let grid_y = code / 16u;


    let padding = 0.05;
    let padded_uv = local_uv * (1.0 - 2.0 * padding) + vec2<f32>(padding);

    // atlas UV coords
    let cell_size_uv = vec2<f32>(1.0 / 16.0, 1.0 / 16.0);
    let cell_offset = vec2<f32>(f32(grid_x), f32(grid_y)) * cell_size_uv;
    let final_uv = cell_offset + padded_uv * cell_size_uv;

    // sample font atlas with textureLoad
    let atlas_coord = vec2<i32>(
        i32(final_uv.x * font_texture_uniform.atlas_size.x),
        i32(final_uv.y * font_texture_uniform.atlas_size.y)
    );
    let sample = textureLoad(t_font_texture_atlas, atlas_coord, 0);

    // red channel font data + anti-alias
    let font_alpha = sample.r * 0.8;
    return smoothstep(0.1, 0.9, font_alpha);
}

// char spacing
fn adv(size: f32) -> f32 {
    return size * (1.0 / FONT_SPACING);
}

// render number
fn num(pp: vec2<f32>, pos: vec2<f32>, number: u32, size: f32) -> f32 {
    let char_advance = adv(size);
    var alpha = 0.0;
    var temp_num = number;
    var digit_count = 0u;

    // Count digits
    if (temp_num == 0u) {
        digit_count = 1u;
    } else {
        var count_temp = temp_num;
        while (count_temp > 0u) {
            count_temp = count_temp / 10u;
            digit_count++;
        }
    }

    // Render digits from right to left
    temp_num = number;
    for (var i = 0u; i < digit_count; i++) {
        let digit = temp_num % 10u;
        let digit_char_code = 48u + digit;
        let digit_pos = pos + vec2<f32>(f32(digit_count - 1u - i) * char_advance, 0.0);
        let char_alpha = ch(pp, digit_pos, digit_char_code, size);
        alpha = max(alpha, char_alpha);
        temp_num = temp_num / 10u;
    }

    return alpha;
}

// a word of up to 16 ASCII codes
fn word(pp: vec2<f32>, pos: vec2<f32>, c: array<u32, 16>, n: u32, size: f32) -> f32 {
    var a = 0.0;
    for (var i = 0u; i < n; i++) {
        a = max(a, ch(pp, pos + vec2<f32>(f32(i) * adv(size), 0.0), c[i], size));
    }
    return a;
}

// the same word centred on x; glyphs sit in the middle of their cells
fn centered(pp: vec2<f32>, x: f32, y: f32, c: array<u32, 16>, n: u32, size: f32) -> f32 {
    return word(pp, vec2<f32>(x - (f32(n - 1u) * adv(size) + size) * 0.5, y), c, n, size);
}

fn digits(n: u32) -> u32 {
    var c = 1u;
    var v = n / 10u;
    while (v > 0u) { v /= 10u; c++; }
    return c;
}

// BLOCK TOWER, CLICK TO START, PERFECT MATCH =, MORE POINTS, SCORE:, GAME OVER, CLICK TO RESTART, PERFECT!
const W_TITLE = array<u32, 16>(66u, 76u, 79u, 67u, 75u, 32u, 84u, 79u, 87u, 69u, 82u, 32u, 32u, 32u, 32u, 32u);
const W_START = array<u32, 16>(67u, 76u, 73u, 67u, 75u, 32u, 84u, 79u, 32u, 83u, 84u, 65u, 82u, 84u, 32u, 32u);
const W_MATCH = array<u32, 16>(80u, 69u, 82u, 70u, 69u, 67u, 84u, 32u, 77u, 65u, 84u, 67u, 72u, 32u, 61u, 32u);
const W_POINTS = array<u32, 16>(77u, 79u, 82u, 69u, 32u, 80u, 79u, 73u, 78u, 84u, 83u, 32u, 32u, 32u, 32u, 32u);
const W_SCORE = array<u32, 16>(83u, 67u, 79u, 82u, 69u, 58u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);
const W_OVER = array<u32, 16>(71u, 65u, 77u, 69u, 32u, 79u, 86u, 69u, 82u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);
const W_RESTART = array<u32, 16>(67u, 76u, 73u, 67u, 75u, 32u, 84u, 79u, 32u, 82u, 69u, 83u, 84u, 65u, 82u, 84u);
const W_PERFECT = array<u32, 16>(80u, 69u, 82u, 70u, 69u, 67u, 84u, 33u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);

// one unit of screen size: 1 at the 600x800 reference window, following the smaller side
fn ui_scale(ss: vec2<f32>) -> f32 { return min(ss.x / 600.0, ss.y / 800.0); }

// state: 0 game, 1 score, 2 current block, 3 click latch, 4 camera y, 8 perfect time, 9 place ev, 10 perfect ev,
// 11 game over ev, 12 last block, 13 pan, 14 next offcut, 15 last drop time, 16 built
// game: 0 menu, 1 play, 2 over
// blocks: 50 * (x, y, z, sx, sy, sz, r, g, b, perfect)
const BD = 100u;
const BS = 10u;
// offcuts: 8 * (x, y, z, sx, sy, sz, t0, side * (block + 1))
const DB = 700u;
const ND = 8u;
const BH: f32 = 0.6;

struct Block { p: vec3<f32>, s: vec3<f32>, c: vec3<f32>, perf: f32, };

fn gs() -> u32 { return u32(G(0u)); }
fn ss_(s: u32) { S(0u, f32(s)); }
fn gsc() -> u32 { return u32(G(1u)); }
fn ssc(s: u32) { S(1u, f32(s)); }
fn gcb() -> u32 { return u32(G(2u)); }
fn scb(b: u32) { S(2u, f32(b)); }
fn gcy() -> f32 { return G(4u); }
// events store their start sample as float bits (+EV0 keeps them normal floats), so ages stay sample-exact
const EV0: u32 = 16777216u;
fn atime() -> f32 { return bitcast<f32>(u_game.sample_offset + EV0); }

// block colours, every third one metallic
fn bcol(id: u32) -> vec3<f32> {
    let h = fract(f32(id) * .618034);
    if (h < .33) { return vec3(.8, .2 + h * 1.8, .1); }
    if (h < .66) { return vec3(.1 + (.66 - h) * 2.1, .8, .2); }
    return vec3(.2, .1 + (h - .66) * 2.1, .9);
}

fn gb(id: u32) -> Block {
    if (id >= 50u) { return Block(vec3(0.), vec3(0.), vec3(0.), 0.); }
    let i = BD + id * BS;
    return Block(vec3(G(i), G(i + 1u), G(i + 2u)), vec3(G(i + 3u), G(i + 4u), G(i + 5u)), vec3(G(i + 6u), G(i + 7u), G(i + 8u)), G(i + 9u));
}
fn sb(id: u32, b: Block) {
    if (id < 50u) {
        let i = BD + id * BS;
        S(i, b.p.x); S(i + 1u, b.p.y); S(i + 2u, b.p.z);
        S(i + 3u, b.s.x); S(i + 4u, b.s.y); S(i + 5u, b.s.z);
        S(i + 6u, b.c.x); S(i + 7u, b.c.y); S(i + 8u, b.c.z);
        S(i + 9u, b.perf);
    }
}
// the part that falls off: centre, size, which way it tips
fn offcut(c: vec3<f32>, s: vec3<f32>, side: f32, id: u32) {
    let k = u32(G(14u)) % ND; let i = DB + k * 8u;
    S(i, c.x); S(i + 1u, c.y); S(i + 2u, c.z); S(i + 3u, s.x); S(i + 4u, s.y); S(i + 5u, s.z);
    S(i + 6u, u_time.time); S(i + 7u, side * f32(id + 1u));
    S(14u, f32(k + 1u));
}

fn updcam() {
    let cb = gcb();
    if (cb > 0u) { S(4u, mix(gcy(), max(f32(cb) * BH - 2.4, 0.), .1)); }
}

fn gmbp() -> vec3<f32> {
    let cb = gcb();
    if (gs() != 1u || cb == 0u) { return vec3(0., -100., 0.); }
    return vec3(sin(u_time.time * 4.) * 2.5, f32(cb) * BH, 0.);
}

fn clear_tower() {
    for (var i = 1u; i < 31u; i++) { sb(i, Block(vec3(0.), vec3(0.), vec3(0.), 0.)); }
    for (var k = 0u; k < ND; k++) { S(DB + k * 8u + 6u, -99.); }
}

fn init() {
    if (G(16u) < .5) {
        sb(0u, Block(vec3(0., 0., 0.), vec3(4., BH, 4.), vec3(.8, .6, .4), 0.));
        clear_tower();
        ss_(0u); ssc(0u); scb(1u); S(3u, 0.); S(4u, 0.); S(8u, -999.); S(16u, 1.);
    }
}

fn upd() {
    let mc = (u_mouse.buttons.x & 1u) != 0u || u_game.action > 0.5;
    let state = gs();
    updcam();
    let wc = G(3u) > .5;
    if (mc && !wc) {
        S(3u, 1.);
        if (state == 0u) {
            ss_(1u); ssc(0u); scb(1u); S(4u, 0.);
        } else if (state == 1u) {
            let cb = gcb();
            if (cb < 30u) {
                let mp = gmbp();
                let pb = gb(cb - 1u);
                var nb = Block(vec3(mp.x, f32(cb) * BH, mp.z), vec3(0., BH, pb.s.z), vec3(0.), 0.);
                let ox = abs(mp.x - pb.p.x);
                nb.s.x = max(pb.s.x - ox, .2);
                nb.perf = select(0., 1., ox < .05);
                if (nb.perf > .5) { S(8u, u_time.time); S(10u, atime()); }
                nb.p.x = select(pb.p.x - (pb.s.x - nb.s.x) * .5, pb.p.x + (pb.s.x - nb.s.x) * .5, mp.x > pb.p.x);
                nb.p.z = pb.p.z;
                nb.c = bcol(cb);
                let side = select(-1., 1., mp.x > pb.p.x);
                S(15u, u_time.time);
                if (nb.s.x < .5) {
                    // missed: the whole block falls
                    offcut(vec3(mp.x, mp.y + BH * .5, mp.z), vec3(pb.s.x, BH, pb.s.z), side, cb);
                    ss_(2u); S(11u, atime());
                } else {
                    if (nb.perf < .5) {
                        let cw = pb.s.x - nb.s.x;
                        offcut(vec3(nb.p.x + side * (nb.s.x + cw) * .5, nb.p.y + BH * .5, nb.p.z), vec3(cw, BH, nb.s.z), side, cb);
                    }
                    sb(cb, nb); scb(cb + 1u); ssc(gsc() + select(10u, 20u, nb.perf > .5));
                    S(9u, atime()); S(12u, f32(cb)); S(13u, clamp(mp.x / 2.5, -1., 1.));
                }
            }
        } else if (state == 2u) {
            ss_(0u); ssc(0u); scb(1u); S(4u, 0.);
            clear_tower();
        }
    } else if (!mc) { S(3u, 0.); }
}

// audio
fn hsh(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u; v *= 0x7feb352du;
    v ^= v >> 15u; v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}
fn wn(n: u32, sd: u32) -> f32 { return f32(hsh(n * 0x9e3779b9u ^ sd) >> 8u) / 8388608.0 - 1.0; }
fn vnz(n: u32, m: u32, sd: u32) -> f32 {
    let i = n / m;
    var f = f32(n % m) / f32(m);
    f = f * f * (3.0 - 2.0 * f);
    return mix(wn(i, sd), wn(i + 1u, sd), f);
}
fn eage(k: u32, n: u32) -> f32 {
    let e = bitcast<u32>(G(k));
    if (G(k) <= 0. || n + EV0 < e) { return -1.; }
    return f32(n + EV0 - e) / u_game.sample_rate;
}
fn pan2(p: f32) -> vec2<f32> {
    let a = (p + 1.) * .785398;
    return vec2(cos(a), sin(a)) * 1.414;
}
fn lp12(f: f32, fc: f32, res: f32) -> f32 {
    let w = f / fc;
    let q = 0.5 + res * 2.0;
    let d = 1.0 - w * w;
    return inverseSqrt(d * d + w * w / (q * q));
}

// band-limited saw at phase `ph` (cycles) through a 12 dB low-pass
fn saw(ph: f32, f: f32, fc: f32) -> f32 {
    let th = TAU * fract(ph);
    let c2 = 2.0 * cos(th);
    var s0 = 0.0;
    var s1 = sin(th);
    var s = 0.0;
    for (var h = 1; h <= 24; h++) {
        let hz = f * f32(h);
        if (hz > 12000.0) { break; }
        s += s1 / f32(h) * lp12(hz, fc, 0.4);
        let s2 = c2 * s1 - s0;
        s0 = s1;
        s1 = s2;
    }
    return s;
}

// FM bell: a 1:1 body and a short 14:1 tine
fn bell(a: f32, f: f32, dec: f32) -> f32 {
    if (a < 0.) { return 0.; }
    let w = TAU * fract(f * a);
    let m = 1.2 * exp(-a / .3) * sin(w) + .5 * exp(-a / .025) * sin(TAU * fract(f * 14. * a));
    return sin(w + m) * exp(-a * dec) * smoothstep(0., .002, a);
}

// sound effects `dl` seconds late, for the echo
fn sfx(n: u32, dl: f32) -> vec2<f32> {
    var s = vec2(0.);

    // place: a wooden thock plus a bell climbing the pentatonic scale with the tower, panned where the block landed
    let dp = eage(9u, n) - dl;
    if (dp >= 0. && dp < 1.2) {
        let k = u32(max(G(12), 0.));
        var deg = array<f32, 5>(0., 2., 4., 7., 9.);
        let f = 261.63 * exp2((deg[k % 5u] + 12. * f32(min(k / 5u, 2u))) / 12.);
        let thock = sin(TAU * fract(90. * dp + 90. * .02 * (1. - exp(-dp / .02)))) * exp(-dp / .06) + wn(n, 3u) * exp(-dp / .004) * .3;
        s += pan2(G(13) * .6) * (bell(dp, f, 3.5) * .22 + thock * .25 * smoothstep(0., .001, dp));
    }

    // perfect: a bright bell arpeggio bouncing left and right, with a shimmer on top
    let dq = eage(10u, n) - dl;
    if (dq >= 0. && dq < 1.4) {
        var notes = array<f32, 4>(523.25, 659.25, 783.99, 1046.5);
        for (var j = 0u; j < 4u; j++) {
            let side = select(vec2(1., .55), vec2(.55, 1.), j % 2u == 1u);
            s += side * bell(dq - f32(j) * .06, notes[j], 4.) * .16;
        }
        s += vec2(wn(n, 5u) - wn(n - 1u, 5u), wn(n, 7u) - wn(n - 1u, 7u)) * .04 * exp(-dq / .25) * smoothstep(0., .01, dq);
    }

    // game over: a detuned saw falling in three notes, the last one sagging, over a low rumble
    let dg = eage(11u, n) - dl;
    if (dg >= 0. && dg < 1.8) {
        var gn = array<f32, 3>(392., 311.13, 261.63);
        for (var j = 0u; j < 3u; j++) {
            let a = dg - f32(j) * .16;
            if (a >= 0.) {
                let f = gn[j];
                let ph = select(f * a, f * (a - .1 * a * a), j == 2u);
                let fc = f * (3. + 4. * exp(-a / .2));
                s += vec2(saw(ph * .996, f, fc), saw(ph * 1.004 + .3, f, fc)) * exp(-a * select(5., 2.5, j == 2u)) * smoothstep(0., .004, a) * .14;
            }
        }
        s += vec2(vnz(n, 90u, 9u)) * exp(-dg / .5) * .25 * smoothstep(0., .02, dg);
    }
    return s;
}

// dry effects plus a ping-pong echo
fn game_sound(n: u32) -> vec2<f32> {
    let s = sfx(n, 0.) + sfx(n, .18) * vec2(.25, .08) + sfx(n, .36) * vec2(.05, .12);
    return tanh(s);
}

@compute @workgroup_size(16, 16, 1)
fn sim(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (all(gid.xy == vec2(0u))) { init(); upd(); }
}

// ---------------- 3D ----------------
// y up, blocks stand on a round stone plinth (top at y = 0) in a meadow
const GY: f32 = -0.35;
const PR: f32 = 3.3;

fn h21(p: v2) -> f32 { return fract(sin(dot(p, v2(127.1, 311.7))) * 43758.5453); }
fn h22(p: v2) -> v2 { return fract(sin(v2(dot(p, v2(127.1, 311.7)), dot(p, v2(269.5, 183.3)))) * 43758.5453); }
fn vn(p: v2) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + v2(1.0, 0.0)), u.x), mix(h21(i + v2(0.0, 1.0)), h21(i + v2(1.0, 1.0)), u.x), u.y);
}
fn fbm(p: v2) -> f32 { return vn(p) * 0.55 + vn(p * 2.1 + 3.7) * 0.3 + vn(p * 4.3 + 9.1) * 0.15; }
fn gyr(p: v3) -> f32 { return dot(sin(p), cos(p.yzx)); }
fn lum(c: v3) -> f32 { return dot(c, v3(0.2126, 0.7152, 0.0722)); }

// the tower, read once per workgroup into shared memory (atomic loads are slow)
// WB: per box (min, perfect) and (max, colour index); the sliding block is the last box while playing
// WD: per offcut (centre, start time) and (size, side); WM: (game, boxes, camera y, perfect time), WN: (drop time, top block, -, -)
var<workgroup> WB: array<v4, 64>;
var<workgroup> WD: array<v4, 16>;
var<workgroup> WM: v4;
var<workgroup> WN: v4;
fn wload(li: u32) {
    let st = gs(); let cb = gcb();
    let nb = select(1u, min(cb, 31u), st != 0u) + select(0u, 1u, st == 1u);
    if (li < 32u && li < nb) {
        var b = gb(li);
        if (st == 1u && li == cb) { let pb = gb(cb - 1u); b = Block(gmbp(), v3(pb.s.x, BH, pb.s.z), bcol(cb), 0.0); }
        WB[li * 2u] = v4(b.p - v3(b.s.x * 0.5, 0.0, b.s.z * 0.5), b.perf);
        WB[li * 2u + 1u] = v4(b.p + v3(b.s.x * 0.5, b.s.y, b.s.z * 0.5), f32(li));
    } else if (li >= 32u && li < 32u + ND) {
        let i = DB + (li - 32u) * 8u;
        WD[(li - 32u) * 2u] = v4(G(i), G(i + 1u), G(i + 2u), G(i + 6u));
        WD[(li - 32u) * 2u + 1u] = v4(G(i + 3u), G(i + 4u), G(i + 5u), G(i + 7u));
    } else if (li == 64u) {
        WM = v4(f32(st), f32(nb), gcy(), G(8u));
        WN = v4(G(15u), f32(cb), 0.0, 0.0);
    }
}

var<private> NOW: f32;
// time of day follows tower height
var<private> SUN: v3; var<private> SUNC: v3; var<private> HOR: v3; var<private> ZEN: v3; var<private> FOGC: v3;
var<private> AMB: f32; var<private> NIGHT: f32;
fn dusk3(d: v3, k: v3, n: v3, ph: f32) -> v3 { return mix(mix(d, k, smoothstep(0.0, 0.5, ph)), n, smoothstep(0.5, 1.0, ph)); }
fn daytime(ph: f32) {
    SUN = normalize(dusk3(v3(-0.45, 0.62, 0.62), v3(-0.75, 0.22, 0.5), v3(0.4, 0.6, 0.5), ph));
    SUNC = dusk3(v3(1.0, 0.9, 0.76) * 3.0, v3(1.0, 0.52, 0.28) * 2.6, v3(0.45, 0.58, 1.0) * 0.5, ph);
    HOR = dusk3(v3(0.95, 0.78, 0.6), v3(1.0, 0.52, 0.36), v3(0.05, 0.07, 0.15), ph);
    ZEN = dusk3(v3(0.32, 0.55, 0.9), v3(0.24, 0.24, 0.5), v3(0.01, 0.015, 0.05), ph);
    FOGC = dusk3(v3(0.75, 0.72, 0.7), v3(0.8, 0.55, 0.45), v3(0.04, 0.05, 0.1), ph);
    AMB = mix(mix(1.0, 0.75, smoothstep(0.0, 0.5, ph)), 0.5, smoothstep(0.5, 1.0, ph));
    NIGHT = smoothstep(0.6, 1.0, ph);
}
fn sun() -> v3 { return SUN; }

// offcut pose: falls, drifts outward, tips over, gone after 3 s
struct Pose { c: v3, h: v3, a: f32, live: bool };
fn pose(k: u32) -> Pose {
    let a0 = WD[k * 2u]; let a1 = WD[k * 2u + 1u];
    let age = NOW - a0.w;
    if (a0.w < -50.0 || age < 0.0 || age > 3.0) { return Pose(v3(0.0), v3(0.0), 0.0, false); }
    let sd = sign(a1.w);
    let c = a0.xyz + v3(sd * 1.3 * age, -4.9 * age * age, 0.0);
    return Pose(c, a1.xyz * 0.5, -sd * 2.6 * age, c.y > GY - 1.0);
}
fn rz(p: v3, a: f32) -> v3 { let c = cos(a); let s = sin(a); return v3(c * p.x + s * p.y, -s * p.x + c * p.y, p.z); }

// slab test: entry distance and the face normal
fn ibox(ro: v3, rd: v3, lo: v3, hi: v3) -> v4 {
    let inv = 1.0 / rd;
    let a = (lo - ro) * inv; let b = (hi - ro) * inv;
    let tn = min(a, b); let tf = max(a, b);
    let t0 = max(max(tn.x, tn.y), tn.z); let t1 = min(min(tf.x, tf.y), tf.z);
    if (t0 > t1 || t1 < 0.0) { return v4(-1.0); }
    var n = v3(0.0);
    if (t0 == tn.x) { n.x = -sign(rd.x); } else if (t0 == tn.y) { n.y = -sign(rd.y); } else { n.z = -sign(rd.z); }
    return v4(t0, n);
}
fn isph(ro: v3, rd: v3, c: v3, r: f32) -> f32 {
    let oc = ro - c; let b = dot(oc, rd); let h = b * b - dot(oc, oc) + r * r;
    if (h < 0.0) { return -1.0; }
    return -b - sqrt(h);
}
// vertical cylinder: distance and normal
fn icyl(ro: v3, rd: v3, c: v2, r: f32, y0: f32, y1: f32) -> v4 {
    let o = ro.xz - c; let d = rd.xz;
    let a = dot(d, d); let b = dot(o, d); let h = b * b - a * (dot(o, o) - r * r);
    if (h < 0.0) { return v4(-1.0); }
    let t = (-b - sqrt(h)) / a;
    let y = ro.y + rd.y * t;
    if (t > 0.0 && y > y0 && y < y1) { return v4(t, normalize(v3(o.x + d.x * t, 0.0, o.y + d.y * t))); }
    // caps
    let tc = (select(y0, y1, rd.y < 0.0) - ro.y) / rd.y;
    let q = o + d * tc;
    if (tc > 0.0 && dot(q, q) < r * r) { return v4(tc, 0.0, select(-1.0, 1.0, rd.y < 0.0), 0.0); }
    return v4(-1.0);
}

// meadow trees around the plinth: two crown spheres and a trunk each
const NT = 11;
fn tree(i: i32) -> v4 {
    let h = h22(v2(f32(i), 3.7));
    let a = f32(i) / f32(NT) * TAU + h.x * 0.5;
    let r = 6.6 + 3.2 * h.y;
    return v4(cos(a) * r, 1.3 + 0.8 * h21(v2(f32(i), 9.1)), sin(a) * r, 0.9 + 0.5 * h.x);
}

// materials
const M_SKY = 0.0; const M_GRASS = 1.0; const M_STONE = 2.0; const M_BLOCK = 3.0; const M_LEAF = 4.0; const M_BARK = 5.0; const M_OFF = 6.0;
struct Hit { t: f32, n: v3, m: f32, id: f32 };

fn trace(ro: v3, rd: v3) -> Hit {
    var h = Hit(1e9, v3(0.0, 1.0, 0.0), M_SKY, 0.0);
    if (rd.y < 0.0) { h = Hit((GY - ro.y) / rd.y, v3(0.0, 1.0, 0.0), M_GRASS, 0.0); }
    let pl = icyl(ro, rd, v2(0.0), PR, GY, 0.0);
    if (pl.x > 0.0 && pl.x < h.t) { h = Hit(pl.x, pl.yzw, M_STONE, 0.0); }
    let nb = u32(WM.y);
    for (var i = 0u; i < nb; i++) {
        let b = ibox(ro, rd, WB[i * 2u].xyz, WB[i * 2u + 1u].xyz);
        if (b.x > 0.0 && b.x < h.t) { h = Hit(b.x, b.yzw, M_BLOCK, f32(i)); }
    }
    for (var k = 0u; k < ND; k++) {
        let ps = pose(k);
        if (!ps.live) { continue; }
        let lo = rz(ro - ps.c, ps.a); let ld = rz(rd, ps.a);
        let b = ibox(lo, ld, -ps.h, ps.h);
        if (b.x > 0.0 && b.x < h.t) { h = Hit(b.x, rz(b.yzw, -ps.a), M_OFF, f32(k)); }
    }
    for (var i = 0; i < NT; i++) {
        let tr = tree(i);
        let tk = icyl(ro, rd, tr.xz, 0.11, GY, tr.y);
        if (tk.x > 0.0 && tk.x < h.t) { h = Hit(tk.x, tk.yzw, M_BARK, f32(i)); }
        let c1 = v3(tr.x, tr.y + tr.w * 0.55, tr.z);
        let s1 = isph(ro, rd, c1, tr.w);
        if (s1 > 0.0 && s1 < h.t) { h = Hit(s1, normalize(ro + rd * s1 - c1), M_LEAF, f32(i)); }
        let c2 = c1 + v3(0.25, tr.w * 0.75, -0.15);
        let s2 = isph(ro, rd, c2, tr.w * 0.7);
        if (s2 > 0.0 && s2 < h.t) { h = Hit(s2, normalize(ro + rd * s2 - c2), M_LEAF, f32(i)); }
    }
    return h;
}

// distance to the boxes and offcuts, for soft shadows and occlusion
fn sdbox(p: v3, b: v3) -> f32 { let q = abs(p) - b; return length(max(q, v3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0); }
fn bdist(p: v3) -> f32 {
    var d = 1e5;
    let nb = u32(WM.y);
    for (var i = 0u; i < nb; i++) {
        let lo = WB[i * 2u].xyz; let hi = WB[i * 2u + 1u].xyz;
        d = min(d, sdbox(p - (lo + hi) * 0.5, (hi - lo) * 0.5) - 0.02);
    }
    for (var k = 0u; k < ND; k++) {
        let ps = pose(k);
        if (ps.live) { d = min(d, sdbox(rz(p - ps.c, ps.a), ps.h)); }
    }
    return d;
}
fn sphshadow(ro: v3, rd: v3, c: v3, r: f32) -> f32 {
    let oc = ro - c; let b = dot(oc, rd); let cc = dot(oc, oc) - r * r;
    let h = b * b - cc;
    if (b > 0.0) { return 1.0; }
    let d = sqrt(max(0.0, r * r - h)) - r;
    let t = -b - sqrt(max(h, 0.0));
    return select(clamp(6.0 * d / max(-b, 1e-4), 0.0, 1.0), 0.0, h > 0.0 && t > 0.0);
}
// tower shadows marched in its bounding column, trees analytic
fn shadow(p: v3, l: v3) -> f32 {
    var res = 1.0;
    let top = max(WB[(u32(WM.y) - 1u) * 2u + 1u].y, 1.0) + 0.5;
    let cy = icyl(p, l, v2(0.0), 4.6, GY - 2.0, top);
    if (cy.x > -0.5 || length(p.xz) < 4.6) {
        var t = 0.03;
        for (var i = 0; i < 40; i++) {
            let q = p + l * t;
            if (q.y > top || length(q.xz) > 6.0) { break; }
            let d = bdist(q);
            res = min(res, 9.0 * d / t);
            if (res < 0.01) { return 0.0; }
            t += clamp(d, 0.02, 0.5);
        }
    }
    for (var i = 0; i < NT; i++) {
        let tr = tree(i);
        let c1 = v3(tr.x, tr.y + tr.w * 0.55, tr.z);
        res = min(res, sphshadow(p, l, c1, tr.w));
        res = min(res, sphshadow(p, l, c1 + v3(0.25, tr.w * 0.75, -0.15), tr.w * 0.7));
    }
    return clamp(res, 0.0, 1.0);
}
fn occl(p: v3, n: v3) -> f32 {
    if (length(p.xz) > 5.5 && p.y < 0.5) { return 1.0; }
    var o = 0.0; var w = 1.0;
    for (var i = 0; i < 3; i++) {
        let hh = 0.06 + 0.16 * f32(i * i);
        o += w * max(0.0, hh - bdist(p + n * hh)) / hh;
        w *= 0.6;
    }
    return clamp(1.0 - o * 0.6, 0.2, 1.0);
}

fn sky(d: v3) -> v3 {
    var c = mix(HOR, ZEN, smoothstep(-0.02, 0.45, d.y));
    let s = max(dot(d, sun()), 0.0);
    c += mix(v3(1.0, 0.85, 0.6), v3(0.8, 0.85, 1.0), NIGHT) * (pow(s, 600.0) * mix(20.0, 6.0, NIGHT) + pow(s, 12.0) * 0.25 * (1.0 - NIGHT));
    // stars
    if (NIGHT > 0.0 && d.y > 0.0) {
        let sp = d.xz / (d.y + 0.3) * 60.0;
        let sh = h22(floor(sp));
        c += v3(0.9, 0.92, 1.0) * step(0.985, sh.x) * smoothstep(0.35, 0.0, length(fract(sp) - 0.5)) * NIGHT * (0.6 + 0.4 * sin(NOW * 3.0 + sh.y * 30.0));
    }
    // soft clouds on a dome
    if (d.y > 0.0) {
        let cp = d.xz / (d.y + 0.15) * 1.4 + v2(NOW * 0.01, 0.0);
        let cl = smoothstep(0.5, 0.8, fbm(cp));
        let cc = mix(v3(1.0, 0.96, 0.92), v3(0.85, 0.82, 0.86), fbm(cp * 2.0 + 4.0)) * mix(v3(1.0), HOR * 1.2, 0.4) * mix(1.0, 0.12, NIGHT);
        c = mix(c, cc, cl * smoothstep(0.0, 0.25, d.y) * 0.8);
    }
    return c;
}
fn hills(d: v3) -> f32 { let a = atan2(d.z, d.x); return 0.035 + 0.05 * fbm(v2(a * 2.5, 1.0)) + 0.02 * fbm(v2(a * 9.0, 4.0)); }

struct Mat { alb: v3, rough: f32, em: v3, coat: f32, wrap: f32 };

fn material(p: v3, n: v3, h: Hit) -> Mat {
    var m = Mat(v3(0.5), 0.8, v3(0.0), 0.0, 0.0);
    if (h.m == M_GRASS) {
        let g = fbm(p.xz * 0.6);
        m.alb = mix(v3(0.17, 0.3, 0.08), v3(0.32, 0.44, 0.13), g) * (0.85 + 0.25 * vn(p.xz * 12.0));
        // a ring of paving around the plinth
        let r = length(p.xz);
        let pave = smoothstep(PR + 1.1, PR + 0.8, r + 0.2 * vn(p.xz * 3.0));
        let cell = h21(floor(p.xz * 2.4));
        m.alb = mix(m.alb, mix(v3(0.52, 0.48, 0.42), v3(0.64, 0.6, 0.52), cell) * (0.8 + 0.2 * vn(p.xz * 9.0)), pave);
        let fl = h22(floor(p.xz * 8.0));
        if (fl.x > 0.94 && pave < 0.1) { m.alb = select(v3(0.95, 0.92, 0.85), v3(0.85, 0.45, 0.8), fl.y > 0.6); }
        m.rough = 0.95; m.wrap = 0.3;
    } else if (h.m == M_STONE) {
        let row = floor(p.y / 0.12);
        let u = atan2(p.z, p.x) * PR / 0.32 + row * 0.5;
        let mortar = min(1.0 - smoothstep(0.82, 0.94, abs(fract(p.y / 0.12) - 0.5) * 2.0), smoothstep(0.0, 0.1, 1.0 - abs(fract(u) - 0.5) * 2.0));
        let s = mix(v3(0.6, 0.56, 0.5), v3(0.72, 0.68, 0.6), h21(v2(floor(u), row)));
        m.alb = select(mix(s * 0.55, s, mortar), mix(v3(0.62, 0.58, 0.52), v3(0.7, 0.66, 0.58), vn(p.xz * 5.0)), n.y > 0.5);
        m.rough = 0.85;
    } else if (h.m == M_BLOCK || h.m == M_OFF) {
        let i = u32(h.id);
        var c = bcol(i);
        if (h.m == M_OFF) { c = bcol(u32(abs(WD[i * 2u + 1u].w)) - 1u); }
        if (h.m == M_BLOCK && i == 0u) { c = v3(0.75, 0.6, 0.45); }
        // glaze pooling: slightly deeper colour in a soft noise
        m.alb = c * (0.82 + 0.18 * vn(p.xz * 3.0 + p.y * 4.0));
        m.rough = select(0.32, 0.18, i % 3u == 0u);
        m.coat = 1.0; m.wrap = 0.15;
        if (h.m == M_BLOCK) {
            let b = WB[i * 2u];
            if (b.w > 0.5) { m.em = v3(1.0, 0.8, 0.35) * (0.2 + 0.15 * sin(NOW * 6.0)); m.alb = mix(m.alb, v3(1.0, 0.82, 0.4), 0.35); }
            if (WM.x == 1.0 && i == u32(WM.y) - 1u) { m.em += c * (0.12 + 0.08 * sin(NOW * 8.0)); }
            m.em += c * NIGHT * (0.18 + 0.1 * h21(v2(h.id, 7.0)));
        }
    } else if (h.m == M_LEAF) {
        let k = 0.5 + 0.5 * gyr(p * 6.0);
        let tint = h21(v2(h.id, 1.3));
        m.alb = mix(v3(0.16, 0.32, 0.1), v3(0.34, 0.5, 0.15), k * (0.6 + 0.4 * tint));
        m.rough = 0.9; m.wrap = 0.55;
    } else if (h.m == M_BARK) { m.alb = v3(0.28, 0.18, 0.1); m.rough = 0.85; }
    return m;
}

// rounded box edges (normal tilt)
fn bevel(p: v3, n: v3, h: Hit) -> v3 {
    var lo: v3; var hi: v3; var q = p; var a = 0.0;
    if (h.m == M_BLOCK) { lo = WB[u32(h.id) * 2u].xyz; hi = WB[u32(h.id) * 2u + 1u].xyz; }
    else { let ps = pose(u32(h.id)); a = ps.a; q = rz(p - ps.c, a); lo = -ps.h; hi = ps.h; }
    let c = (lo + hi) * 0.5; let e = (hi - lo) * 0.5;
    let d = q - c;
    let ln = select(n, rz(n, a), h.m == M_OFF);
    let dist = e - abs(d);
    var bn = ln;
    let rb = 0.06;
    if (abs(ln.x) < 0.5 && dist.x < rb) { bn += v3(sign(d.x), 0.0, 0.0) * (1.0 - dist.x / rb); }
    if (abs(ln.y) < 0.5 && dist.y < rb) { bn += v3(0.0, sign(d.y), 0.0) * (1.0 - dist.y / rb); }
    if (abs(ln.z) < 0.5 && dist.z < rb) { bn += v3(0.0, 0.0, sign(d.z)) * (1.0 - dist.z / rb); }
    bn = normalize(bn);
    return select(bn, rz(bn, -a), h.m == M_OFF);
}

fn shade(ro: v3, rd: v3, h: Hit, foot: f32) -> v3 {
    if (h.m == M_SKY) {
        var c = sky(rd);
        if (rd.y < hills(rd)) { c = mix(v3(0.42, 0.55, 0.5) * mix(1.0, 0.1, NIGHT), sky(rd), 0.55); }
        return c;
    }
    let p = ro + rd * h.t;
    var n = h.n;
    if (h.m == M_BLOCK || h.m == M_OFF) { n = bevel(p, n, h); }
    if (h.m == M_LEAF) { n = normalize(n + 0.35 * v3(gyr(p * 5.0), gyr(p.yzx * 5.0 + 2.0), gyr(p.zxy * 5.0 + 4.0))); }
    let m = material(p, n, h);
    let l = sun(); let v = -rd;
    let nl = dot(n, l);
    var sh = 0.0;
    if (nl > -m.wrap) { sh = shadow(p + h.n * 0.01, l); }
    let ao = occl(p + h.n * 0.01, h.n);
    let dif = clamp((nl + m.wrap) / (1.0 + m.wrap), 0.0, 1.0);
    // highlight never sharper than the pixel holds
    let r = sqrt(m.rough * m.rough + min(foot * foot * 40.0, 0.3));
    let hv = normalize(l + v);
    let a2 = r * r * r * r; let nh = max(dot(n, hv), 0.0); let dd = nh * nh * (a2 - 1.0) + 1.0;
    let fr = 0.04 + 0.96 * pow(1.0 - max(dot(v, hv), 0.0), 5.0);
    var col = m.alb * SUNC * dif * sh + SUNC * a2 / (3.14159 * dd * dd) * fr * max(nl, 0.0) * sh * 0.25;
    col += m.alb * mix(HOR * 0.55, ZEN * 1.1, n.y * 0.5 + 0.5) * 0.85 * AMB * ao;
    col += m.alb * v3(0.3, 0.36, 0.15) * 0.22 * AMB * ao * clamp(-n.y * 0.5 + 0.5, 0.0, 1.0) * (1.0 - NIGHT * 0.8);
    // glaze: a clear coat mirroring the sky
    if (m.coat > 0.0) {
        let rf = reflect(rd, n);
        col += sky(rf) * (0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0)) * ao * select(0.25, 1.0, rf.y > 0.0) * 0.9;
    }
    return col + m.em;
}

fn cam(px: v2, D: v2, ro: ptr<function, v3>, rd: ptr<function, v3>) {
    var tg = v3(0.0, WM.z + 1.0 + u_game.camera_height, 0.0);
    // a small dip when a block lands
    let da = NOW - WN.x;
    if (da > 0.0 && da < 0.25) { tg.y += sin(da * 40.0) * (1.0 - da / 0.25) * 0.05; }
    let az = 0.785 + u_game.camera_angle; let el = 0.55;
    let dist = 15.0 * 65.0 / max(u_game.camera_scale, 1.0);
    let dir = v3(sin(az) * cos(el), sin(el), cos(az) * cos(el));
    let f = -dir; let rt = normalize(cross(f, v3(0.0, 1.0, 0.0))); let up = cross(rt, f);
    let uv = (px - 0.5 * D) / D.y * 2.0; let tf = 0.42;
    *ro = tg + dir * dist;
    *rd = normalize(f + (uv.x * rt - uv.y * up) * tf);
}

@compute @workgroup_size(16, 16, 1)
fn scene(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    wload(li);
    workgroupBarrier();
    let D = v2(textureDimensions(output));
    if (any(v2(gid.xy) >= D)) { return; }
    NOW = u_time.time;
    daytime(clamp(WM.z / 14.0, 0.0, 1.0));
    var ro: v3; var rd: v3;
    cam(v2(gid.xy) + 0.5, D, &ro, &rd);
    let h = trace(ro, rd);
    let foot = h.t * 0.84 / D.y;
    var col = shade(ro, rd, h, foot);

    // perfect: light ring + sparks
    let pa = NOW - WM.w;
    if (WM.w > 0.0 && pa > 0.0 && pa < 1.0 && WM.x == 1.0) {
        let k = u32(WN.y) - 1u;
        let lo = WB[k * 2u].xyz; let hi = WB[k * 2u + 1u].xyz;
        let yp = hi.y + 0.02;
        let tp = (yp - ro.y) / rd.y;
        if (tp > 0.0 && tp < h.t) {
            let q = ro + rd * tp;
            let c = (lo + hi) * 0.5; let e = (hi - lo) * 0.5;
            let dr = sdbox(v3(q.x - c.x, 0.0, q.z - c.z), v3(e.x, 1.0, e.z));
            let w = 0.06 + 0.25 * pa;
            col += v3(1.0, 0.85, 0.5) * exp(-pow((dr - pa * 2.2) / w, 2.0)) * (1.0 - pa) * 3.0;
        }
        for (var i = 0; i < 28; i++) {
            let hs = h22(v2(f32(i), WM.w));
            let a = hs.x * TAU;
            let sp = v3(cos(a) * e_or(lo, hi).x, 0.0, sin(a) * e_or(lo, hi).z);
            let p0 = (lo + hi) * 0.5 + v3(sp.x, hi.y - (lo.y + hi.y) * 0.5, sp.z);
            let vel = normalize(v3(cos(a), 0.0, sin(a))) * (1.2 + 1.5 * hs.y) + v3(0.0, 2.5 + 2.0 * hs.x, 0.0);
            let pp = p0 + vel * pa + v3(0.0, -4.9 * pa * pa, 0.0);
            let tt = dot(pp - ro, rd);
            if (tt < 0.0 || tt > h.t) { continue; }
            let dd = length(ro + rd * tt - pp);
            col += v3(1.0, 0.8, 0.4) * exp(-dd * dd / (0.0009 + foot * foot * 4.0)) * 4.0 * (1.0 - pa);
        }
    }

    // fireflies drift over the meadow after dusk
    if (NIGHT > 0.0) {
        for (var i = 0; i < 24; i++) {
            let hs = h22(v2(f32(i), 4.1));
            let a = hs.x * TAU + NOW * 0.05;
            let rr = 4.0 + 5.0 * hs.y;
            let fp = v3(cos(a) * rr + sin(NOW * 0.4 + f32(i)) * 0.5, GY + 0.4 + 1.2 * h21(v2(f32(i), 8.0)) + 0.2 * sin(NOW * 0.7 + f32(i)), sin(a) * rr);
            let tt = dot(fp - ro, rd);
            if (tt < 0.0 || tt > h.t) { continue; }
            let dd = length(ro + rd * tt - fp);
            let bl = pow(max(0.0, sin(NOW * (1.2 + hs.x) + f32(i) * 4.0)), 3.0) * NIGHT;
            col += v3(1.0, 0.85, 0.35) * bl * (exp(-dd * dd / (foot * foot * 9.0 + 1e-5)) * 2.5 + exp(-dd * 20.0) * 0.25);
        }
    }
    // haze with distance
    col = mix(col, FOGC, clamp(1.0 - exp(-max(h.t - 18.0, 0.0) * 0.035), 0.0, 0.8));
    textureStore(output, vec2<i32>(gid.xy), v4(max(col, v3(0.0)), h.t));
}
fn e_or(lo: v3, hi: v3) -> v3 { return (hi - lo) * 0.5; }

// bloom and blur chains (non-wrapping: taps clamped half a texel inside)
fn cuv(uv: v2, t: texture_2d<f32>) -> v2 { let d = v2(textureDimensions(t)); return clamp(uv, 0.5 / d, 1.0 - 0.5 / d); }
fn dsmp(t: texture_2d<f32>, sm: sampler, uv: v2) -> v3 {
    let sp = 1.0 / v2(textureDimensions(t));
    return (textureSampleLevel(t, sm, cuv(uv + v2(-sp.x, -sp.y), t), 0.0).xyz + textureSampleLevel(t, sm, cuv(uv + v2(sp.x, -sp.y), t), 0.0).xyz
          + textureSampleLevel(t, sm, cuv(uv + v2(-sp.x, sp.y), t), 0.0).xyz + textureSampleLevel(t, sm, cuv(uv + v2(sp.x, sp.y), t), 0.0).xyz) * 0.25;
}
fn usmp(uv: v2) -> v3 {
    let lp = 1.0 / v2(textureDimensions(t1));
    var b = v3(0.0);
    for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
        b += textureSampleLevel(t1, s1, cuv(uv + v2(f32(i), f32(j)) * lp, t1), 0.0).xyz * f32((2 - abs(i)) * (2 - abs(j)));
    } }
    return textureSampleLevel(t0, s0, cuv(uv, t0), 0.0).xyz + b / 16.0;
}
fn pxuv(id: vec3<u32>) -> v2 { return (v2(id.xy) + 0.5) / v2(textureDimensions(output)); }
@compute @workgroup_size(16, 16, 1)
fn bd1(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(output))) { return; }
    let c = dsmp(t0, s0, pxuv(id)); let m = max(c.x, max(c.y, c.z));
    textureStore(output, id.xy, v4(c * max(m - 1.0, 0.0) / max(m, 1e-4), 1.0));
}
@compute @workgroup_size(16, 16, 1)
fn bd2(@builtin(global_invocation_id) id: vec3<u32>) { if (any(id.xy >= textureDimensions(output))) { return; } textureStore(output, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd3(@builtin(global_invocation_id) id: vec3<u32>) { if (any(id.xy >= textureDimensions(output))) { return; } textureStore(output, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd4(@builtin(global_invocation_id) id: vec3<u32>) { if (any(id.xy >= textureDimensions(output))) { return; } textureStore(output, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu3(@builtin(global_invocation_id) id: vec3<u32>) { if (any(id.xy >= textureDimensions(output))) { return; } textureStore(output, id.xy, v4(usmp(pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu2(@builtin(global_invocation_id) id: vec3<u32>) { if (any(id.xy >= textureDimensions(output))) { return; } textureStore(output, id.xy, v4(usmp(pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu1(@builtin(global_invocation_id) id: vec3<u32>) { if (any(id.xy >= textureDimensions(output))) { return; } textureStore(output, id.xy, v4(usmp(pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn db1(@builtin(global_invocation_id) id: vec3<u32>) { if (any(id.xy >= textureDimensions(output))) { return; } textureStore(output, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn db2(@builtin(global_invocation_id) id: vec3<u32>) { if (any(id.xy >= textureDimensions(output))) { return; } textureStore(output, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn du1(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(output))) { return; }
    let uv = pxuv(id); let lp = 1.0 / v2(textureDimensions(t1));
    var b = v3(0.0);
    for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
        b += textureSampleLevel(t1, s1, cuv(uv + v2(f32(i), f32(j)) * lp, t1), 0.0).xyz * f32((2 - abs(i)) * (2 - abs(j)));
    } }
    textureStore(output, id.xy, v4(mix(textureSampleLevel(t0, s0, cuv(uv, t0), 0.0).xyz, b / 16.0, 0.75), 1.0));
}

// text in ui_scale units: colour + coverage
fn txt(pp: vec2<f32>, ss: vec2<f32>) -> v4 {
    let state = gs();
    let u = ui_scale(ss);
    let cx = ss.x * .5;
    var a = 0.0; var tc = v3(1.0);
    let pt = G(8u);
    let dt = u_time.time - pt;
    if (dt < 2. && dt > 0. && pt > 0. && state == 1u) {
        let fade = 1. - dt / 2.;
        let size = 80. * u * (1. + sin(dt * 8.) * .2 * fade);
        let w = centered(pp, cx, ss.y * .3 - size * .5, W_PERFECT, 8u, size) * fade;
        if (w > a) { a = w; tc = v3(1.0, 0.85, 0.4); }
    }
    if (state == 0u) {
        var w = centered(pp, cx, ss.y * .12, W_TITLE, 11u, 64. * u); if (w > a) { a = w; tc = v3(1., 0.9, 0.5); }
        w = centered(pp, cx, ss.y * .25, W_START, 14u, 32. * u); if (w > a) { a = w; tc = v3(1.0, 0.55, 0.35); }
        w = centered(pp, cx, ss.y * .335, W_MATCH, 15u, 24. * u); if (w > a) { a = w; tc = v3(0.95); }
        w = centered(pp, cx, ss.y * .37, W_POINTS, 11u, 24. * u); if (w > a) { a = w; tc = v3(0.95); }
    } else if (state == 1u) {
        let size = 48. * u;
        let m = 40. * u;
        var w = word(pp, vec2(m, m), W_SCORE, 6u, size); if (w > a) { a = w; tc = v3(1.); }
        w = num(pp, vec2(m + 6. * adv(size), m), gsc(), size); if (w > a) { a = w; tc = v3(1.0, 0.85, 0.5); }
    } else if (state == 2u) {
        var w = centered(pp, cx, ss.y * .4, W_OVER, 9u, 60. * u); if (w > a) { a = w; tc = v3(1., .35, .3); }
        let size = 40. * u;
        let x0 = cx - (f32(5u + digits(gsc())) * adv(size) + size) * .5;
        w = word(pp, vec2(x0, ss.y * .5), W_SCORE, 6u, size); if (w > a) { a = w; tc = v3(1.); }
        w = num(pp, vec2(x0 + 6. * adv(size), ss.y * .5), gsc(), size); if (w > a) { a = w; tc = v3(1.); }
        w = centered(pp, cx, ss.y * .6, W_RESTART, 16u, 32. * u); if (w > a) { a = w; tc = v3(0.95); }
    }
    return v4(tc, a);
}

fn aces(x: v3) -> v3 { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), v3(0.0), v3(1.0)); }

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    // audio: one thread per sample, after the sim pass has moved the game on
    let ai = (wid.y * nw.x + wid.x) * 256u + li;
    if (ai < u_game.samples_to_generate) {
        let v = game_sound(u_game.sample_offset + ai) * u_game.volume;
        audio_buffer[ai * 2u] = v.x;
        audio_buffer[ai * 2u + 1u] = v.y;
    }

    let ss = vec2<f32>(textureDimensions(output));
    let pp = vec2<f32>(gid.xy);
    if (any(pp >= ss)) { return; }
    let uv = (pp + 0.5) / ss;
    let state = gs();

    // tilt-shift: the band around the top of the tower stays sharp
    let sc = textureLoad(t0, vec2<i32>(gid.xy), 0);
    let bl = textureSampleLevel(t2, s2, cuv(uv, t2), 0.0).xyz;
    let coc = smoothstep(0.16, 0.48, abs(uv.y - 0.5));
    var col = mix(sc.xyz, bl, coc);
    col += textureSampleLevel(t1, s1, cuv(uv, t1), 0.0).xyz * 0.35;

    // perfect flash
    let pt = G(8u); let dt = u_time.time - pt;
    if (dt < .3 && dt > 0. && pt > 0. && state == 1u) { col += v3(1.0, 0.9, 0.6) * (1.0 - dt / 0.3) * 0.25; }

    col = aces(col * 1.05);
    col = mix(v3(lum(col)), col, 1.08);
    col *= 0.74 + 0.26 * pow(16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.18);
    if (state == 0u) { col *= 0.8; }
    if (state == 2u) { col = mix(col, col * v3(1.0, 0.5, 0.45), 0.35); }

    let t = txt(pp, ss);
    let tsh = txt(pp - v2(3.0, 3.0) * ui_scale(ss), ss).w;
    col = mix(col, v3(0.0), tsh * 0.55);
    col = mix(col, t.rgb, t.w);

    col += (h21(pp + fract(u_time.time)) - 0.5) * 0.015;
    // the display expects linear light
    textureStore(output, vec2<i32>(gid.xy), v4(pow(clamp(col, v3(0.0), v3(1.0)), v3(2.2)), 1.0));
}
