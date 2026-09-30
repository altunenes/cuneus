// Bugberman, Enes Altun, 2026, CC0

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

struct Game { dir: u32, act: u32, vol: f32, so: u32, sn: u32, sr: f32, p0: f32, p1: f32 };
@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> gm: Game;

// g2: fonts (0,1) + audio (2). no mouse so fonts come first
struct Font { asz: v2, csz: v2, ssz: v2, gsz: v2 };
@group(2) @binding(0) var<uniform> fu: Font;
@group(2) @binding(1) var ft: texture_2d<f32>;
@group(2) @binding(2) var<storage, read_write> au: array<f32>;
@group(3) @binding(0) var<storage, read_write> g: array<f32>;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>; alias i2 = vec2<i32>; alias u3 = vec3<u32>;
const tau: f32 = 6.2831853;
const GW: i32 = 13; const GH: i32 = 11;
const TB: i32 = 100;  // tile type   [TB + y*GW + x]
const FB: i32 = 400;  // flame end   [FB + y*GW + x]
const BB: i32 = 600;  // bombs: NB * (x, y, etime, range)
const NB: i32 = 8;
const FUSE: f32 = 2.2; const FDUR: f32 = 0.55; const RAD: i32 = 1;
const MDLY: f32 = 0.12;
const EB: i32 = 700;  // enemies: NE * (alive, tx, ty, vx, vy, dir, last, _)
const NE: i32 = 8; const ES: i32 = 8; const EDLY: f32 = 0.18; const CHASE: f32 = 0.5;

// state: 0 ptx 1, 2 pty, 3 pvx, 4 pvy, 5 last, 6 crates, 7 actflag, 26 music start
// audio ev: 20 place 21 boom 22 die 23 win 24 step 25 kill   visual ev: 30..35 (same order)

fn hsh(a0: u32) -> u32 { var a = a0; a ^= a >> 16u; a *= 0x7feb352du; a ^= a >> 15u; a *= 0x846ca68bu; a ^= a >> 16u; return a; }
fn rng(x: i32, y: i32, s: u32) -> f32 { return f32(hsh(u32(x) * 73u + u32(y) * 131u + s * 977u + 12345u)) / 4294967295.0; }
// events store their start sample as float bits (+EV0 keeps them normal floats), so ages stay sample-exact
const EV0: u32 = 16777216u;
fn atime() -> f32 { return bitcast<f32>(gm.so + EV0); }

fn gt(x: i32, y: i32) -> u32 { if (x < 0 || y < 0 || x >= GW || y >= GH) { return 1u; } return u32(g[TB + y * GW + x]); }
fn st(x: i32, y: i32, v: u32) { if (x >= 0 && y >= 0 && x < GW && y < GH) { g[TB + y * GW + x] = f32(v); } }
fn sf(x: i32, y: i32, t: f32) { if (x >= 0 && y >= 0 && x < GW && y < GH) { g[FB + y * GW + x] = t; } }
fn gf(x: i32, y: i32) -> f32 { if (x < 0 || y < 0 || x >= GW || y >= GH) { return 0.0; } return g[FB + y * GW + x]; }
fn fa(x: i32, y: i32, now: f32) -> bool { return gf(x, y) > now; }
fn bi(x: i32, y: i32) -> i32 {
    for (var i = 0; i < NB; i++) {
        if (g[BB + i * 4 + 2] > 0.001 && i32(g[BB + i * 4]) == x && i32(g[BB + i * 4 + 1]) == y) { return i; }
    }
    return -1;
}
fn dvec(d: u32) -> i2 {
    if (d == 1u) { return i2(0, -1); }
    if (d == 2u) { return i2(0, 1); }
    if (d == 3u) { return i2(-1, 0); }
    return i2(1, 0);
}
// open for enemies: empty, no bomb, no live flame (so freed enemies don't walk into a lingering blast: for balancing)
fn opn(x: i32, y: i32) -> bool { return gt(x, y) == 0u && bi(x, y) < 0 && !fa(x, y, u_t.time); }
fn nliv() -> u32 { var c = 0u; for (var e = 0; e < NE; e++) { if (g[EB + e * ES] > 0.5) { c++; } } return c; }
fn opp(d: u32) -> u32 { if (d == 1u) { return 2u; } if (d == 2u) { return 1u; } if (d == 3u) { return 4u; } return 3u; }

// random open dir from a rotated offset, avoiding `av` when possible (0 = boxed)
fn pick(x: i32, y: i32, av: u32, r: f32) -> u32 {
    let s = u32(r * 4.0) % 4u;
    var fb = 0u;
    for (var k = 0u; k < 4u; k++) {
        let c = (s + k) % 4u + 1u;
        let d = dvec(c);
        if (opn(x + d.x, y + d.y)) { if (c != av) { return c; } fb = c; }
    }
    return fb;
}

// half chase the player (greedy on the longer axis), half wander turn at junctions, no backtrack
fn estep(e: i32, now: f32) {
    let b = EB + e * ES;
    if (now - g[b + 6] > EDLY) {
        let ex = i32(g[b + 1]); let ey = i32(g[b + 2]);
        let dir = u32(g[b + 5]);
        let dx = i32(g[1]) - ex; let dy = i32(g[2]) - ey;
        let r = rng(ex * GW + ey, e * 7 + 1, u32(now * 5.0));
        var nd = 0u;

        if (r < CHASE) {
            var w = 0u;
            if (abs(dx) >= abs(dy)) { if (dx != 0) { w = select(3u, 4u, dx > 0); } }
            else { if (dy != 0) { w = select(1u, 2u, dy > 0); } }
            if (w != 0u) { let d = dvec(w); if (opn(ex + d.x, ey + d.y)) { nd = w; } }
            if (nd == 0u) {
                var w2 = 0u;
                if (abs(dx) >= abs(dy)) { if (dy != 0) { w2 = select(1u, 2u, dy > 0); } }
                else { if (dx != 0) { w2 = select(3u, 4u, dx > 0); } }
                if (w2 != 0u) { let d = dvec(w2); if (opn(ex + d.x, ey + d.y)) { nd = w2; } }
            }
        }
        if (nd == 0u) {
            let d = dvec(dir);
            let tn = rng(ex + e, ey * 3 + 1, u32(now * 7.0));
            if (opn(ex + d.x, ey + d.y) && tn > 0.4) { nd = dir; }
            else { nd = pick(ex, ey, opp(dir), rng(ex * 5 + e, ey, u32(now * 11.0))); }
        }
        if (nd == 0u) { nd = pick(ex, ey, 0u, r); }

        if (nd != 0u) {
            let d = dvec(nd);
            if (opn(ex + d.x, ey + d.y)) { g[b + 1] = f32(ex + d.x); g[b + 2] = f32(ey + d.y); }
            g[b + 5] = f32(nd);
        }
        g[b + 6] = now;
    }
    g[b + 3] = mix(g[b + 3], g[b + 1], 0.3);
    g[b + 4] = mix(g[b + 4], g[b + 2], 0.3);
}

fn pbomb(x: i32, y: i32) {
    for (var i = 0; i < NB; i++) { if (g[BB + i * 4 + 2] > 0.001) { return; } }   // one at a time
    for (var i = 0; i < NB; i++) {
        if (g[BB + i * 4 + 2] <= 0.001) {
            g[BB + i * 4] = f32(x); g[BB + i * 4 + 1] = f32(y);
            g[BB + i * 4 + 2] = u_t.time + FUSE; g[BB + i * 4 + 3] = f32(RAD);
            g[20] = atime(); g[31] = u_t.time;
            return;
        }
    }
}

fn boom(idx: i32, now: f32) {
    let bx = i32(g[BB + idx * 4]); let by = i32(g[BB + idx * 4 + 1]); let rg = i32(g[BB + idx * 4 + 3]);
    g[BB + idx * 4 + 2] = 0.0;
    sf(bx, by, now + FDUR);
    g[21] = atime(); g[30] = now;
    var dirs = array<i2, 4>(i2(1, 0), i2(-1, 0), i2(0, 1), i2(0, -1));
    for (var di = 0; di < 4; di++) {
        let d = dirs[di];
        for (var r = 1; r <= rg; r++) {
            let cx = bx + d.x * r; let cy = by + d.y * r;
            let t = gt(cx, cy);
            if (t == 1u) { break; }                              // wall blocks
            sf(cx, cy, now + FDUR);
            if (t == 2u) { st(cx, cy, 0u); g[6] -= 1.0; break; } // crate burns
            let j = bi(cx, cy);
            if (j >= 0) { g[BB + j * 4 + 2] = now; }             // chain
        }
    }
}

fn newg() {
    let sd = u_t.frame;
    var cr = 0.0;
    for (var y = 0; y < GH; y++) {
        for (var x = 0; x < GW; x++) {
            var t = 0u;
            if (x == 0 || y == 0 || x == GW - 1 || y == GH - 1) { t = 1u; }   // border
            else if ((x % 2) == 0 && (y % 2) == 0) { t = 1u; }               // pillars
            else { if (!(x <= 2 && y <= 2) && rng(x, y, sd) < 0.55) { t = 2u; cr += 1.0; } }
            g[TB + y * GW + x] = f32(t);
            g[FB + y * GW + x] = 0.0;
        }
    }
    for (var i = 0; i < NB; i++) { g[BB + i * 4 + 2] = 0.0; }

    // enemies spread away from the player corner
    var sp = array<i2, 8>(
        i2(GW - 2, GH - 2), i2(GW - 2, 1), i2(1, GH - 2), i2(GW / 2, GH / 2),
        i2(GW - 2, GH / 2), i2(GW / 2, GH - 2), i2(GW / 2, 1), i2(4, GH - 2)
    );
    for (var e = 0; e < NE; e++) {
        let p = sp[e];
        st(p.x, p.y, 0u);
        if (gt(p.x + 1, p.y) == 2u) { st(p.x + 1, p.y, 0u); }
        if (gt(p.x - 1, p.y) == 2u) { st(p.x - 1, p.y, 0u); }
        if (gt(p.x, p.y + 1) == 2u) { st(p.x, p.y + 1, 0u); }
        if (gt(p.x, p.y - 1) == 2u) { st(p.x, p.y - 1, 0u); }
        let b = EB + e * ES;
        g[b] = 1.0; g[b + 1] = f32(p.x); g[b + 2] = f32(p.y);
        g[b + 3] = f32(p.x); g[b + 4] = f32(p.y);
        g[b + 5] = f32((e % 4) + 1); g[b + 6] = 0.0;
    }

    g[1] = 1.0; g[2] = 1.0; g[3] = 1.0; g[4] = 1.0; g[5] = 0.0; g[6] = cr; g[0] = 1.0;
    g[20] = 0.0; g[21] = 0.0; g[22] = 0.0; g[23] = 0.0; g[24] = 0.0; g[25] = 0.0;
    g[30] = 0.0; g[31] = 0.0; g[32] = 0.0; g[33] = 0.0; g[34] = 0.0; g[35] = 0.0;
    g[26] = atime();
}

fn init() { if (u_t.frame == 1u) { g[0] = 0.0; g[7] = 0.0; } }

fn upd() {
    let now = u_t.time;
    let act = gm.act != 0u;
    let edge = act && !(g[7] > 0.5);
    g[7] = select(0.0, 1.0, act);
    let s = u32(g[0]);

    if (s != 1u) { if (edge) { newg(); } return; }   // menu / win / over -> space

    // move (cadence owned here)
    let px = i32(g[1]); let py = i32(g[2]);
    if (gm.dir != 0u && now - g[5] > MDLY) {
        let d = dvec(gm.dir);
        let nx = px + d.x; let ny = py + d.y;
        if (gt(nx, ny) == 0u && bi(nx, ny) < 0) { g[1] = f32(nx); g[2] = f32(ny); g[5] = now; g[24] = atime(); g[34] = now; }
    }
    g[3] = mix(g[3], g[1], 0.35);
    g[4] = mix(g[4], g[2], 0.35);

    // drop on the action edge
    if (edge) { let cx = i32(g[1]); let cy = i32(g[2]); if (bi(cx, cy) < 0) { pbomb(cx, cy); } }

    // detonate
    for (var i = 0; i < NB; i++) { let et = g[BB + i * 4 + 2]; if (et > 0.001 && now >= et) { boom(i, now); } }

    // player death
    if (fa(i32(g[1]), i32(g[2]), now)) { g[0] = 3.0; g[22] = atime(); g[32] = now; }

    // enemies: burn, move, touch
    var liv = 0;
    for (var e = 0; e < NE; e++) {
        let b = EB + e * ES;
        if (g[b] < 0.5) { continue; }
        if (fa(i32(g[b + 1]), i32(g[b + 2]), now)) { g[b] = 0.0; g[25] = atime(); g[35] = now; continue; }
        estep(e, now);
        liv += 1;
        if (i32(g[b + 1]) == i32(g[1]) && i32(g[b + 2]) == i32(g[2])) { g[0] = 3.0; g[22] = atime(); g[32] = now; }
    }
    if (liv == 0 && u32(g[0]) == 1u) { g[0] = 2.0; g[23] = atime(); g[33] = now; }
}

// font (atlas renderer, from blockgame)
const FSP: f32 = 2.0;
fn ch(pp: v2, pos: v2, code: u32, sz: f32) -> f32 {
    let rp = pp - pos;
    if (rp.x < 0.0 || rp.x >= sz || rp.y < 0.0 || rp.y >= sz) { return 0.0; }
    let luv = rp / vec2(sz);
    let pad = 0.05;
    let puv = luv * (1.0 - 2.0 * pad) + vec2(pad);
    let cell = v2(1.0 / 16.0);
    let off = v2(f32(code % 16u), f32(code / 16u)) * cell;
    let uv = off + puv * cell;
    let ac = vec2<i32>(i32(uv.x * fu.asz.x), i32(uv.y * fu.asz.y));
    return smoothstep(0.1, 0.9, textureLoad(ft, ac, 0).r * 0.8);
}
fn adv(sz: f32) -> f32 { return sz * (1.0 / FSP); }
fn num(pp: v2, pos: v2, n0: u32, sz: f32) -> f32 {
    let ca = adv(sz);
    var a = 0.0; var dc = 0u;
    if (n0 == 0u) { dc = 1u; } else { var c = n0; while (c > 0u) { c = c / 10u; dc++; } }
    var n = n0;
    for (var i = 0u; i < dc; i++) {
        a = max(a, ch(pp, pos + v2(f32(dc - 1u - i) * ca, 0.0), 48u + n % 10u, sz));
        n = n / 10u;
    }
    return a;
}
fn word(pp: v2, pos: v2, c: array<u32, 16>, n: u32, sz: f32) -> f32 {
    let ca = adv(sz);
    var a = 0.0;
    for (var i = 0u; i < n; i++) { a = max(a, ch(pp, pos + v2(f32(i) * ca, 0.0), c[i], sz)); }
    return a;
}

// audio
fn nf(m: f32) -> f32 { return 440.0 * pow(2.0, (m - 69.0) / 12.0); }
fn wn(n: u32, sd: u32) -> f32 { return f32(hsh(n * 0x9e3779b9u ^ sd) >> 8u) / 8388608.0 - 1.0; }
fn bn(n: u32, m: u32, sd: u32) -> f32 { let i = n / m; var f = f32(n % m) / f32(m); f = f * f * (3.0 - 2.0 * f); return mix(wn(i, sd), wn(i + 1u, sd), f); }
fn eage(k: i32, n: u32) -> f32 { let e = bitcast<u32>(g[k]); if (g[k] <= 0.0 || n + EV0 < e) { return -1.0; } return f32(n + EV0 - e) / gm.sr; }
fn rel(a: f32, gate: f32, r: f32) -> f32 { return select(1.0, exp(-(a - gate) / r), a > gate); }
fn lp12(f: f32, fc: f32, res: f32) -> f32 { let w = f / fc; let q = 0.5 + res * 2.0; let d = 1.0 - w * w; return inverseSqrt(d * d + w * w / (q * q)); }

// band-limited saw at phase `ph` (cycles) through a 12 dB low-pass
fn saw(ph: f32, f: f32, fc: f32) -> f32 {
    let th = tau * fract(ph); let c2 = 2.0 * cos(th);
    var s0 = 0.0; var s1 = sin(th); var s = 0.0;
    for (var h = 1; h <= 24; h++) {
        let hz = f * f32(h); if (hz > 12000.0) { break; }
        s += s1 / f32(h) * lp12(hz, fc, 0.4);
        let s2 = c2 * s1 - s0; s0 = s1; s1 = s2;
    }
    return s;
}

// drums: integrated pitch sweeps, noise from the sample index
fn kick(a: f32) -> f32 {
    if (a < 0.0 || a > 1.2) { return 0.0; }
    let ph = 48.0 * a + 117.0 * 0.032 * (1.0 - exp(-a / 0.032));
    return tanh((sin(tau * fract(ph)) * exp(-a / 0.3) + sin(tau * fract(ph * 1.72)) * exp(-a / 0.035) * 0.25) * 1.8) * smoothstep(0.0, 0.0008, a);
}
fn clap(a: f32, n: u32) -> f32 {
    if (a < 0.0 || a > 0.8) { return 0.0; }
    var e = 0.0;
    for (var i = 0; i < 3; i++) { let x = a - f32(i) * 0.0105; if (x >= 0.0) { e += exp(-x / 0.0035); } }
    e += 0.6 * exp(-max(a - 0.021, 0.0) / 0.13) * step(0.021, a);
    return (bn(n, 2u, 7u) - bn(n, 9u, 7u)) * e * 0.8 + sin(tau * fract(185.0 * a + 1.2 * (1.0 - exp(-a / 0.02)))) * exp(-a / 0.05) * 0.35;
}
fn metal(a: f32) -> f32 {
    let fs = array<f32, 6>(205.3, 304.4, 369.6, 522.7, 540.0, 800.0);
    var m = 0.0; for (var i = 0; i < 6; i++) { m += select(-1.0, 1.0, fract(fs[i] * 2.0 * a) < 0.5); }
    return m;
}
fn hat(a: f32, n: u32, dc: f32) -> f32 {
    if (a < 0.0 || a > dc * 8.0) { return 0.0; }
    let hp = wn(n, 3u) - 0.6 * wn(n - 1u, 3u) - 0.4 * wn(n - 2u, 3u);
    return (hp * 0.6 + (metal(a) - metal(a - 1.0 / 44100.0)) / 12.0) * exp(-a / dc) * smoothstep(0.0, 0.0004, a);
}
// velocity, decay per 16th: offbeat 8ths, 16th ghosts, an open hat every other bar
fn hhit(st: i32) -> v2 {
    let p = st % 16;
    if (p == 14 && (st / 16) % 2 == 1) { return v2(0.75, 0.22); }
    if (p % 4 == 2) { return v2(0.8, 0.04); }
    if (p % 2 == 0) { return v2(0.4, 0.03); }
    if (wn(u32(st), 5u) < 0.0) { return v2(0.18, 0.025); }
    return v2(0.0);
}

// bass: sub octave + two detuned saws, filter snaps shut, soft drive
fn bassv(f: f32, a: f32, gate: f32) -> v2 {
    if (a < 0.0 || a > gate + 0.2) { return v2(0.0); }
    let env = smoothstep(0.0, 0.002, a) * (0.55 + 0.45 * exp(-a / 0.12)) * rel(a, gate, 0.02);
    let fc = f * (1.4 + 9.0 * exp(-a / 0.07));
    let st = v2(saw(f * 0.9962 * a, f, fc), saw(f * 1.0038 * a + 0.37, f, fc));
    return tanh((st * 0.45 + sin(tau * fract(f * 0.5 * a)) * 0.7) * 1.3) * env;
}

// BOMBERMAN BGM 1 (Atsushi Chikuma, 1987) - bass groove, 4 bars / 64 sixteenths @ Q=129, with a drum kit
fn bgm(n: u32) -> v2 {
    if (u32(g[0]) != 1u || g[26] <= 0.0) { return v2(0.0); }
    let e0 = bitcast<u32>(g[26]);
    if (n + EV0 < e0) { return v2(0.0); }
    let s16 = 60.0 / 129.0 / 4.0;
    let mt = f32((n + EV0 - e0) % u32(64.0 * s16 * gm.sr)) / gm.sr;
    let pos = mt / s16;
    var note = array<f32, 45>(
        47.,47.,59.,47.,50.,54.,56.,57.,57.,56.,
        45.,45.,57.,45.,49.,52.,54.,55.,54.,55.,45.,44.,45.,
        42.,42.,54.,42.,52.,51.,52.,42.,54.,42.,
        42.,42.,54.,42.,52.,51.,52.,42.,54.,42.,44.,46.
    );
    var dur = array<f32, 45>(
        1.,1.,1.,1.,2.,1.,1.,2.,2.,4.,
        1.,1.,1.,1.,2.,1.,1.,2.,1.,1.,1.,1.,2.,
        1.,1.,1.,1.,2.,1.,1.,2.,2.,4.,
        1.,1.,1.,1.,2.,1.,1.,2.,2.,2.,1.,1.
    );
    var acc = 0.0; var ci = 0u;
    for (var i = 0u; i < 45u; i++) {
        if (pos < acc + dur[i]) { ci = i; break; }
        acc += dur[i];
    }
    // current note plus the previous one's release tail (wraps to the last note of the loop)
    let pi = (ci + 44u) % 45u;
    var bass = bassv(nf(note[ci]), mt - acc * s16, dur[ci] * s16 * 0.9);
    bass += bassv(nf(note[pi]), mt - (acc - dur[pi]) * s16, dur[pi] * s16 * 0.9);

    let st = i32(pos);
    let ka = mt - f32(st / 4 * 4) * s16;
    let kk = kick(ka) + kick(ka + 4.0 * s16) * exp(-ka / 0.01) + (wn(n, 9u) - wn(n - 1u, 9u)) * 0.2 * exp(-ka / 0.0012);
    var cs = st / 8 * 8 + 4;
    if (st % 8 < 4) { cs -= 8; }
    let cl = clap(mt - f32(cs) * s16, n);
    let h0 = hhit(st); let h1 = hhit((st + 63) % 64);
    let ha = mt - f32(st) * s16;
    let hh = hat(ha, n, h0.y) * h0.x + hat(ha + s16, n, h1.y) * h1.x * select(1.0, exp(-ha / 0.008), h0.x > 0.0);
    let duck = 1.0 - 0.4 * exp(-ka * 12.0);
    return bass * 0.55 * duck + v2(kk * 0.5 + cl * 0.2) + hh * v2(0.07, 0.1);
}

// sfx: place / boom / die / win / step / kill, `dl` seconds late for the echo
fn sfx(n: u32, dl: f32) -> v2 {
    var s = v2(0.0);
    // place: FM blip falling a fifth
    let dp = eage(20, n) - dl;
    if (dp >= 0.0 && dp < 0.35) {
        let ph = 880.0 * dp + 440.0 * 0.03 * (1.0 - exp(-dp / 0.03));
        let m = 1.2 * exp(-dp / 0.04) * sin(tau * fract(ph * 2.0));
        s += v2(sin(tau * fract(ph) + m)) * exp(-dp / 0.07) * smoothstep(0.0, 0.002, dp) * 0.22;
    }
    // boom: sub drop, noise body, low rumble, first crack and sparse crackle
    let db = eage(21, n) - dl;
    if (db >= 0.0 && db < 1.6) {
        let ph = 38.0 * db + 82.0 * 0.06 * (1.0 - exp(-db / 0.06));
        let sub = sin(tau * fract(ph)) * exp(-db / 0.35);
        let body = (bn(n, 6u, 31u) - 0.5 * bn(n, 40u, 31u)) * exp(-db / 0.18);
        let rumble = bn(n, 90u, 33u) * exp(-db / 0.6);
        let crack = wn(n, 35u) * exp(-db / 0.02);
        let m = tanh((sub * 0.9 + body * 0.8 + rumble * 0.9 + crack * 0.5) * 1.5);
        let pop = select(0.0, 1.0, wn(n / 300u, 37u) > 0.7) * exp(-db / 0.3) * 0.15;
        s += v2(m) * 0.45 + v2(wn(n, 39u), wn(n, 41u)) * pop;
    }
    // die: detuned saw lead falling in three notes, the last one sagging
    let dd = eage(22, n) - dl;
    if (dd >= 0.0 && dd < 1.4) {
        var gn = array<f32, 3>(392.0, 311.13, 233.08);
        for (var j = 0u; j < 3u; j++) {
            let a = dd - f32(j) * 0.16;
            if (a >= 0.0) {
                let f = gn[j];
                var ph = f * a;
                if (j == 2u) { ph = f * (a - 0.12 * a * a); }
                let fc = f * (3.0 + 4.0 * exp(-a / 0.2));
                s += v2(saw(ph * 0.996, f, fc), saw(ph * 1.004 + 0.3, f, fc)) * exp(-a * select(6.0, 3.0, j == 2u)) * smoothstep(0.0, 0.004, a) * 0.16;
            }
        }
    }
    // win: FM bell arpeggio, bouncing left and right
    let dw = eage(23, n) - dl;
    if (dw >= 0.0 && dw < 1.6) {
        var wf = array<f32, 4>(523.25, 659.25, 783.99, 1046.5);
        for (var j = 0u; j < 4u; j++) {
            let a = dw - f32(j) * 0.12;
            if (a >= 0.0) {
                let w = tau * fract(wf[j] * a);
                let m = 1.4 * exp(-a / 0.4) * sin(w) + 0.6 * exp(-a / 0.03) * sin(tau * fract(wf[j] * 14.0 * a));
                let side = select(v2(1.0, 0.55), v2(0.55, 1.0), j % 2u == 1u);
                s += side * sin(w + m) * exp(-a * 3.0) * smoothstep(0.0, 0.002, a) * 0.16;
            }
        }
    }
    // step: a short woody tock
    let dk = eage(24, n) - dl;
    if (dk >= 0.0 && dk < 0.08) {
        let ph = 620.0 * dk + 380.0 * 0.006 * (1.0 - exp(-dk / 0.006));
        s += v2(sin(tau * fract(ph)) * exp(-dk / 0.012) + wn(n, 43u) * exp(-dk / 0.0015) * 0.3) * 0.07;
    }
    // kill: FM zap diving down
    let dx = eage(25, n) - dl;
    if (dx >= 0.0 && dx < 0.4) {
        let ph = 200.0 * dx + 1000.0 * 0.05 * (1.0 - exp(-dx / 0.05));
        let m = 2.5 * exp(-dx / 0.1) * sin(tau * fract(ph * 1.5));
        s += v2(sin(tau * fract(ph) + m) * exp(-dx / 0.12) + wn(n, 45u) * exp(-dx / 0.01) * 0.2) * 0.2;
    }
    return s;
}

fn snd(n: u32) -> v2 {
    var s = bgm(n) * 0.4 + sfx(n, 0.0);
    // ping-pong echo on the effects
    s += sfx(n, 0.19) * v2(0.28, 0.1) + sfx(n, 0.38) * v2(0.05, 0.14);
    return tanh(s);
}

// render
fn h2(p: v2) -> f32 { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
fn vn(p: v2) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(h2(i), h2(i + vec2(1.0, 0.0)), u.x), mix(h2(i + vec2(0.0, 1.0)), h2(i + vec2(1.0, 1.0)), u.x), u.y);
}
fn seg(p: v2, a: v2, b: v2) -> f32 { let pa = p - a; let ba = b - a; return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)); }

// Ferris
fn shell(n: v2) -> f32 {
    let rr = length(n);
    let ang = atan2(-n.y, n.x);
    let up = sin(ang);
    let a = 1.08; let b = 0.84;
    var R = a * b / sqrt(b * b * cos(ang) * cos(ang) + a * a * up * up);
    R += 0.07 * (1.0 - abs(fract(ang * 2.866) - 0.5) * 2.0) * smoothstep(0.15, 0.6, up);
    if (up < -0.05) { R = min(R, 0.74 / -up); }                                         
    return rr - R;
}

fn chela(n: v2) -> f32 {
    let arm = seg(n, v2(-0.5, 0.3), v2(-1.0, -0.28)) - 0.1;
    let hand = length((n - v2(-1.2, -0.46)) * v2(0.9, 1.0)) - 0.34;
    let notch = length(n - v2(-1.45, -0.78)) - 0.26;
    return min(arm, max(hand, -notch));
}

fn ferris(n: v2) -> v4 {
    let org = v3(0.95, 0.27, 0.04);
    let out = v3(0.4, 0.08, 0.0);
    var c = v3(0.0); var a = 0.0;

    let lg = min(min(seg(n, v2(-0.5, 0.5), v2(-0.6, 0.98)), seg(n, v2(-0.17, 0.6), v2(-0.22, 1.04))),
                 min(seg(n, v2(0.5, 0.5), v2(0.6, 0.98)), seg(n, v2(0.17, 0.6), v2(0.22, 1.04)))) - 0.05;
    if (lg < 0.0) { c = org * 0.7; a = 1.0; }

    // claws
    let cd = min(chela(n), chela(v2(-n.x, n.y)));
    if (cd < 0.06) { c = out; a = 1.0; }
    if (cd < 0.0) { c = org * 0.85; a = 1.0; }

    // shell
    let sd = shell(n);
    if (sd < 0.06) { c = out; a = 1.0; }
    if (sd < 0.0) {
        let lit = 0.66 + 0.34 * (-n.y) - 0.16 * length(n);
        let spec = smoothstep(0.55, 0.0, length(n - v2(-0.3, -0.5))) * 0.3;
        c = org * lit + spec;
        a = 1.0;
    }

    // face
    if (sd < -0.04) {
        let ck = min(length((n - v2(-0.45, 0.08)) * v2(1.0, 1.3)), length((n - v2(0.45, 0.08)) * v2(1.0, 1.3)));
        if (ck < 0.15) { c = mix(c, v3(1.0, 0.45, 0.35), 0.3); }                                          // cheeks
        let eye = min(length((n - v2(-0.27, -0.05)) * v2(1.15, 0.85)), length((n - v2(0.27, -0.05)) * v2(1.15, 0.85)));
        if (eye < 0.2) { c = v3(0.05); }                                                                  // eyes
        if (min(length(n - v2(-0.32, -0.12)), length(n - v2(0.22, -0.12))) < 0.06) { c = v3(0.95); }      // catchlights
        if (abs(n.y - (0.32 - 0.6 * n.x * n.x)) < 0.035 && abs(n.x) < 0.2) { c = out; }                   // smile
    }
    return v4(c, a);
}

// Bug (the enemies): rounded carapace, two antennae, angry eyes. tint varies per bug.
fn bug(n: v2, tint: v3) -> v4 {
    let lit = 0.65 + 0.35 * (-n.y);
    var c = v3(0.0); var a = 0.0;
    if (min(seg(n, v2(-0.2, -0.7), v2(-0.55, -1.15)), seg(n, v2(0.2, -0.7), v2(0.55, -1.15))) < 0.07) { c = tint * 0.55; a = 1.0; }
    if (min(length(n - v2(-0.55, -1.15)), length(n - v2(0.55, -1.15))) < 0.12) { c = tint * 0.8; a = 1.0; }
    let bd = length(n * v2(1.0, 1.12));
    if (bd < 1.0) { c = select(tint * lit, tint * 0.25, bd > 0.87); a = 1.0; if (abs(n.x) < 0.06) { c *= 0.55; } }
    if (bd < 1.0) {
        if (min(length((n - v2(-0.32, -0.22)) * v2(1.0, 1.2)), length((n - v2(0.32, -0.22)) * v2(1.0, 1.2))) < 0.2) { c = v3(0.95); }
        if (min(length(n - v2(-0.29, -0.17)), length(n - v2(0.29, -0.17))) < 0.08) { c = v3(0.05); }
    }
    return v4(c, a);
}

fn hud(pp: v2, ss: v2) -> v3 {
    let s = u32(g[0]);
    var tc = v3(0.0);
    var c = array<u32, 16>(32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);
    if (s == 0u) {
        c = array<u32, 16>(66u, 79u, 77u, 66u, 69u, 82u, 77u, 65u, 78u, 32u, 32u, 32u, 32u, 32u, 32u, 32u); // BOMBERMAN
        if (word(pp, v2(ss.x * 0.5 - 9.0 * adv(64.0) * 0.5, ss.y * 0.32), c, 9u, 64.0) > 0.01) { tc = v3(1.0, 0.85, 0.2); }
        c = array<u32, 16>(80u, 82u, 69u, 83u, 83u, 32u, 83u, 80u, 65u, 67u, 69u, 32u, 32u, 32u, 32u, 32u); // PRESS SPACE
        if (word(pp, v2(ss.x * 0.5 - 11.0 * adv(30.0) * 0.5, ss.y * 0.32 + 90.0), c, 11u, 30.0) > 0.01) { tc = v3(0.9, 0.4, 0.1); }
    } else if (s == 1u) {
        c = array<u32, 16>(69u, 78u, 69u, 77u, 73u, 69u, 83u, 58u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u); // ENEMIES:
        if (word(pp, v2(28.0, 24.0), c, 8u, 36.0) > 0.01) { tc = v3(1.0); }
        if (num(pp, v2(28.0 + 8.0 * adv(36.0), 24.0), nliv(), 36.0) > 0.01) { tc = v3(1.0, 0.5, 0.4); }
    } else if (s == 2u) {
        c = array<u32, 16>(89u, 79u, 85u, 32u, 87u, 73u, 78u, 33u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u); // YOU WIN!
        if (word(pp, v2(ss.x * 0.5 - 8.0 * adv(64.0) * 0.5, ss.y * 0.4), c, 8u, 64.0) > 0.01) { tc = v3(0.4, 1.0, 0.5); }
        c = array<u32, 16>(80u, 82u, 69u, 83u, 83u, 32u, 83u, 80u, 65u, 67u, 69u, 32u, 32u, 32u, 32u, 32u);
        if (word(pp, v2(ss.x * 0.5 - 11.0 * adv(28.0) * 0.5, ss.y * 0.4 + 80.0), c, 11u, 28.0) > 0.01) { tc = v3(0.8); }
    } else if (s == 3u) {
        c = array<u32, 16>(71u, 65u, 77u, 69u, 32u, 79u, 86u, 69u, 82u, 32u, 32u, 32u, 32u, 32u, 32u, 32u); // GAME OVER
        if (word(pp, v2(ss.x * 0.5 - 9.0 * adv(64.0) * 0.5, ss.y * 0.4), c, 9u, 64.0) > 0.01) { tc = v3(1.0, 0.25, 0.25); }
        c = array<u32, 16>(80u, 82u, 69u, 83u, 83u, 32u, 83u, 80u, 65u, 67u, 69u, 32u, 32u, 32u, 32u, 32u);
        if (word(pp, v2(ss.x * 0.5 - 11.0 * adv(28.0) * 0.5, ss.y * 0.4 + 80.0), c, 11u, 28.0) > 0.01) { tc = v3(0.8); }
    }
    return tc;
}

@compute @workgroup_size(8, 8, 1)
fn sim(@builtin(global_invocation_id) gid: u3) {
    if (all(gid.xy == vec2(0u))) {
        init(); upd();
    }
}

@compute @workgroup_size(8, 8, 1)
fn main_image(@builtin(global_invocation_id) gid: u3, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: u3, @builtin(num_workgroups) nw: u3) {
    // audio: one thread per sample, after the sim pass has moved the game on
    let ai = (wid.y * nw.x + wid.x) * 64u + li;
    if (ai < gm.sn) { let v = snd(gm.so + ai) * gm.vol; au[ai * 2u] = v.x; au[ai * 2u + 1u] = v.y; }

    let ss = v2(textureDimensions(out));
    let pp = v2(gid.xy);
    if (any(pp >= ss)) { return; }

    let now = u_t.time;
    let s = u32(g[0]);

    // explosion shake
    let bage = now - g[30];
    var shk = v2(0.0);
    if (g[30] > 0.0 && bage > 0.0 && bage < 0.35) { shk = v2(sin(now * 90.0), cos(now * 97.0)) * (1.0 - bage / 0.35) * 8.0; }

    // board layout (top-down, hud strip on top)
    let hh = 64.0;
    let tl = min(ss.x / f32(GW), (ss.y - hh) / f32(GH));
    let ox = (ss.x - tl * f32(GW)) * 0.5 + shk.x;
    let oy = hh + (ss.y - hh - tl * f32(GH)) * 0.5 + shk.y;
    let bp = (pp - v2(ox, oy)) / tl;
    let tx = i32(floor(bp.x)); let ty = i32(floor(bp.y));
    let lc = fract(bp);
    let inb = bp.x >= 0.0 && bp.x < f32(GW) && bp.y >= 0.0 && bp.y < f32(GH);

    var col = v3(0.06, 0.07, 0.10) + v3(0.02) * vn(pp * 0.02);

    if (inb && s != 0u) {
        col = select(v3(0.16, 0.42, 0.20), v3(0.20, 0.48, 0.24), ((tx + ty) % 2) == 0);   // floor

        let t = gt(tx, ty);
        if (t == 1u) {
            var b = v3(0.36, 0.39, 0.46) * mix(1.25, 0.65, lc.y);
            let e = step(0.06, lc.x) * step(lc.x, 0.94) * step(0.06, lc.y) * step(lc.y, 0.94);
            col = mix(b * 0.7, b, e);
        } else if (t == 2u) {
            var b = v3(0.62, 0.40, 0.18) * mix(1.2, 0.78, lc.y);
            b *= 0.82 + 0.18 * smoothstep(0.08, 0.2, abs(fract(lc.y * 3.0) - 0.5));
            let e = step(0.08, lc.x) * step(lc.x, 0.92) * step(0.08, lc.y) * step(lc.y, 0.92);
            col = mix(v3(0.30, 0.18, 0.07), b, e);
        }

        if (bi(tx, ty) >= 0) {
            let d = length(lc - 0.5);
            if (d < 0.32 * (1.0 + 0.06 * sin(now * 10.0))) {
                let n = (lc - 0.5) / 0.32;
                col = v3(0.07, 0.07, 0.10) * (0.5 + 0.5 * (-n.y));
                if (length(lc - v2(0.40, 0.38)) < 0.07) { col += v3(0.45); }
            }
            if (length(lc - v2(0.60, 0.18)) < 0.06) { col = mix(v3(1.0, 0.55, 0.1), v3(1.0, 1.0, 0.6), 0.5 + 0.5 * sin(now * 30.0)); }
        }
    }

    // enemies = bugs (under the player)
    if (s == 1u) {
        var pal = array<v3, 4>(v3(0.45, 0.65, 0.22), v3(0.62, 0.30, 0.72), v3(0.22, 0.60, 0.72), v3(0.74, 0.52, 0.18));
        for (var e = 0; e < NE; e++) {
            let b = EB + e * ES;
            if (g[b] < 0.5) { continue; }
            let cx = ox + (g[b + 3] + 0.5) * tl;
            let cy = oy + (g[b + 4] + 0.5) * tl;
            let r = tl * 0.34;
            if (abs(pp.x - cx) > r * 1.6 || abs(pp.y - cy) > r * 1.9) { continue; }   // bbox reject
            let bob = sin(now * 5.0 + f32(e)) * tl * 0.02;
            col = mix(col * 0.55, col, smoothstep(r * 0.6, r * 1.15, length((pp - v2(cx, cy + tl * 0.30)) / v2(1.0, 0.5))));
            let bgc = bug((pp - v2(cx, cy - bob)) / r, pal[e % 4]);
            col = mix(col, bgc.rgb, bgc.a);
        }
    }

    // player = Ferris
    if (s == 1u) {
        let cx = ox + (g[3] + 0.5) * tl;
        let cy = oy + (g[4] + 0.5) * tl;
        let r = tl * 0.40;
        if (!(abs(pp.x - cx) > r * 1.85 || abs(pp.y - cy) > r * 1.7)) {
            let bob = sin(now * 4.0) * tl * 0.02;
            col = mix(col * 0.55, col, smoothstep(r * 0.5, r * 1.1, length((pp - v2(cx, cy + tl * 0.32)) / v2(1.0, 0.5))));
            let fr = ferris((pp - v2(cx, cy - bob)) / r);
            col = mix(col, fr.rgb, fr.a);
        }
    }

    // flames on top (cover the player when caught)
    if (inb && s != 0u && fa(tx, ty, now)) {
        let life = clamp((gf(tx, ty) - now) / FDUR, 0.0, 1.0);
        let n = vn(lc * 4.0 + v2(f32(tx), f32(ty)) - v2(0.0, now * 7.0));
        let core = smoothstep(0.55, 0.0, length(lc - 0.5));
        let f = clamp(core * (0.5 + 0.7 * n) * (0.4 + life), 0.0, 1.2);
        let fc = mix(v3(1.7, 0.3, 0.05), v3(1.8, 1.6, 0.6), core);
        col = mix(col, fc, clamp(f, 0.0, 1.0)) + fc * f * 0.4;
    }

    if (s == 2u) { col = mix(col, v3(0.3, 0.7, 0.4), 0.18); }
    if (s == 3u) { col = mix(col, v3(0.6, 0.1, 0.1), 0.25); }
    if (g[30] > 0.0 && bage > 0.0 && bage < 0.12) { col += v3(1.0, 0.8, 0.4) * (1.0 - bage / 0.12) * 0.5; }   // flash

    let tc = hud(pp, ss);
    if (length(tc) > 0.0) { col = tc; }

    // tonemap + gamma + vignette
    col *= 1.15;
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    col = pow(col, v3(2.2));
    let uv = pp / ss;
    col += (h2(pp + fract(now)) - 0.5) * 0.04;                                       // film grain
    col *= 0.7 + 0.3 * pow(16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.15);   // vignette
    col = clamp(col, v3(0.0), v3(1.0));

    textureStore(out, vec2<i32>(gid.xy), v4(col, 1.0));
}
