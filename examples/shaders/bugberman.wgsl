// Bugberman, Enes Altun, 2026, CC0
// Marching and antialiasing tricks after Tonny Espeset's Cacodemon (https://www.shadertoy.com/view/NXtXzj).

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

// tsh: tilt-shift amount, blm: bloom
struct Game { dir: u32, act: u32, vol: f32, so: u32, sn: u32, sr: f32, tsh: f32, blm: f32 };
@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> gm: Game;

// g2: fonts (0,1), audio (2), game state (3)
struct Font { asz: v2, csz: v2, ssz: v2, gsz: v2 };
@group(2) @binding(0) var<uniform> fu: Font;
@group(2) @binding(1) var ft: texture_2d<f32>;
@group(2) @binding(2) var<storage, read_write> au: array<f32>;
@group(2) @binding(3) var<storage, read_write> atm: array<atomic<u32>>;

@group(3) @binding(0) var t0: texture_2d<f32>; @group(3) @binding(1) var s0: sampler;
@group(3) @binding(2) var t1: texture_2d<f32>; @group(3) @binding(3) var s1: sampler;
@group(3) @binding(4) var t2: texture_2d<f32>; @group(3) @binding(5) var s2: sampler;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>; alias i2 = vec2<i32>; alias u3 = vec3<u32>;
const tau: f32 = 6.2831853;
const GW: i32 = 13; const GH: i32 = 11;
// tile type   [TB + y*GW + x]  0 floor, 1 wall, 2 bush, 3 gate, 4 fire up, 5 bomb up
const TB: i32 = 100;
// flame end   [FB + y*GW + x]
const FB: i32 = 400;
// bombs: NB * (x, y, etime, range)
const BB: i32 = 600;
const NB: i32 = 8;
const FUSE: f32 = 2.2; const FDUR: f32 = 0.55;
const MDLY: f32 = 0.12;
// enemies: NE * (alive, tx, ty, vx, vy, dir, last, _)
const EB: i32 = 700;
const NE: i32 = 8; const ES: i32 = 8;

// state: 0 game, 1 ptx, 2 pty, 3 pvx, 4 pvy, 5 last move, 6 bushes, 7 action latch, 8 level, 9 flame range, 10 bomb cap,
// 11 facing, 12-13 last blast cell, 14 gate open, 15 level start, 16 built, 26 music start
// game: 0 menu, 1 play, 2 level clear, 3 over, 4 victory
// audio ev: 20 place 21 boom 22 die 23 win 24 step 25 kill 27 pickup   visual ev: 30..35, 37
fn G(i: i32) -> f32 { return bitcast<f32>(atomicLoad(&atm[u32(i)])); }
fn S(i: i32, v: f32) { atomicStore(&atm[u32(i)], bitcast<u32>(v)); }

fn hsh(a0: u32) -> u32 { var a = a0; a ^= a >> 16u; a *= 0x7feb352du; a ^= a >> 15u; a *= 0x846ca68bu; a ^= a >> 16u; return a; }
fn rng(x: i32, y: i32, s: u32) -> f32 { return f32(hsh(u32(x) * 73u + u32(y) * 131u + s * 977u + 12345u)) / 4294967295.0; }
// events store their start sample as float bits (+EV0 keeps them normal floats), so ages stay sample-exact
const EV0: u32 = 16777216u;
fn atime() -> f32 { return bitcast<f32>(gm.so + EV0); }

fn gt(x: i32, y: i32) -> u32 { if (x < 0 || y < 0 || x >= GW || y >= GH) { return 1u; } return u32(G(TB + y * GW + x)); }
fn st(x: i32, y: i32, v: u32) { if (x >= 0 && y >= 0 && x < GW && y < GH) { S(TB + y * GW + x, f32(v)); } }
fn sf(x: i32, y: i32, t: f32) { if (x >= 0 && y >= 0 && x < GW && y < GH) { S(FB + y * GW + x, t); } }
fn gf(x: i32, y: i32) -> f32 { if (x < 0 || y < 0 || x >= GW || y >= GH) { return 0.0; } return G(FB + y * GW + x); }
fn fa(x: i32, y: i32, now: f32) -> bool { return gf(x, y) > now; }
fn bi(x: i32, y: i32) -> i32 {
    for (var i = 0; i < NB; i++) {
        if (G(BB + i * 4 + 2) > 0.001 && i32(G(BB + i * 4)) == x && i32(G(BB + i * 4 + 1)) == y) { return i; }
    }
    return -1;
}
fn dvec(d: u32) -> i2 {
    if (d == 1u) { return i2(0, -1); }
    if (d == 2u) { return i2(0, 1); }
    if (d == 3u) { return i2(-1, 0); }
    return i2(1, 0);
}
fn lv() -> i32 { return clamp(i32(G(8)), 0, 2); }
fn edly() -> f32 { return select(select(0.15, 0.19, lv() == 1), 0.24, lv() == 0); }
fn chase() -> f32 { return select(select(0.65, 0.5, lv() == 1), 0.35, lv() == 0); }
fn walk(t: u32) -> bool { return t == 0u || t >= 3u; }
// open for enemies: empty, no bomb, no live flame (so freed enemies don't walk into a lingering blast: for balancing)
fn opn(x: i32, y: i32) -> bool { return gt(x, y) == 0u && bi(x, y) < 0 && !fa(x, y, u_t.time); }
fn nliv() -> u32 { var c = 0u; for (var e = 0; e < NE; e++) { if (G(EB + e * ES) > 0.5) { c++; } } return c; }
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
    if (now - G(b + 6) > edly()) {
        let ex = i32(G(b + 1)); let ey = i32(G(b + 2));
        let dir = u32(G(b + 5));
        let dx = i32(G(1)) - ex; let dy = i32(G(2)) - ey;
        let r = rng(ex * GW + ey, e * 7 + 1, u32(now * 5.0));
        var nd = 0u;

        if (r < chase()) {
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
            if (opn(ex + d.x, ey + d.y)) { S(b + 1, f32(ex + d.x)); S(b + 2, f32(ey + d.y)); }
            S(b + 5, f32(nd));
        }
        S(b + 6, now);
    }
    S(b + 3, mix(G(b + 3), G(b + 1), 0.3));
    S(b + 4, mix(G(b + 4), G(b + 2), 0.3));
}

fn pbomb(x: i32, y: i32) {
    var n = 0.0;
    for (var i = 0; i < NB; i++) { if (G(BB + i * 4 + 2) > 0.001) { n += 1.0; } }
    if (n >= G(10)) { return; }
    for (var i = 0; i < NB; i++) {
        if (G(BB + i * 4 + 2) <= 0.001) {
            S(BB + i * 4, f32(x)); S(BB + i * 4 + 1, f32(y));
            S(BB + i * 4 + 2, u_t.time + FUSE); S(BB + i * 4 + 3, G(9));
            S(20, atime()); S(31, u_t.time);
            return;
        }
    }
}

fn boom(idx: i32, now: f32) {
    let bx = i32(G(BB + idx * 4)); let by = i32(G(BB + idx * 4 + 1)); let rg = i32(G(BB + idx * 4 + 3));
    S(BB + idx * 4 + 2, 0.0);
    sf(bx, by, now + FDUR);
    S(21, atime()); S(30, now); S(12, f32(bx)); S(13, f32(by));
    var dirs = array<i2, 4>(i2(1, 0), i2(-1, 0), i2(0, 1), i2(0, -1));
    for (var di = 0; di < 4; di++) {
        let d = dirs[di];
        for (var r = 1; r <= rg; r++) {
            let cx = bx + d.x * r; let cy = by + d.y * r;
            let t = gt(cx, cy);
            // wall blocks
            if (t == 1u) { break; }
            sf(cx, cy, now + FDUR);
            // bush burns, sometimes leaving a power-up
            if (t == 2u) {
                let pr = rng(cx, cy, u32(now * 13.0) + 7u);
                st(cx, cy, select(select(0u, 5u, pr < 0.2), 4u, pr < 0.12)); S(6, G(6) - 1.0);
                break;
            }
            let j = bi(cx, cy);
            // chain
            if (j >= 0) { S(BB + j * 4 + 2, now); }
        }
    }
}

// level 0 meadow, 1 dusk, 2 night: denser bushes, more and faster bugs
fn newg(l: i32) {
    let sd = u_t.frame;
    S(8, f32(l));
    let dens = select(select(0.62, 0.55, l == 1), 0.45, l == 0);
    let ne = 4 + 2 * l;
    var cr = 0.0;
    for (var y = 0; y < GH; y++) {
        for (var x = 0; x < GW; x++) {
            var t = 0u;
            // border
            if (x == 0 || y == 0 || x == GW - 1 || y == GH - 1) { t = 1u; }
            // pillars
            else if ((x % 2) == 0 && (y % 2) == 0) { t = 1u; }
            else { if (!(x <= 2 && y <= 2) && rng(x, y, sd) < dens) { t = 2u; cr += 1.0; } }
            S(TB + y * GW + x, f32(t));
            S(FB + y * GW + x, 0.0);
        }
    }
    for (var i = 0; i < NB; i++) { S(BB + i * 4 + 2, 0.0); }

    // enemies spread away from the player corner
    var sp = array<i2, 8>(
        i2(GW - 2, GH - 2), i2(GW - 2, 1), i2(1, GH - 2), i2(GW / 2, GH / 2),
        i2(GW - 2, GH / 2), i2(GW / 2, GH - 2), i2(GW / 2, 1), i2(4, GH - 2)
    );
    for (var e = 0; e < NE; e++) {
        let b = EB + e * ES;
        if (e >= ne) { S(b, 0.0); continue; }
        let p = sp[e];
        st(p.x, p.y, 0u);
        if (gt(p.x + 1, p.y) == 2u) { st(p.x + 1, p.y, 0u); }
        if (gt(p.x - 1, p.y) == 2u) { st(p.x - 1, p.y, 0u); }
        if (gt(p.x, p.y + 1) == 2u) { st(p.x, p.y + 1, 0u); }
        if (gt(p.x, p.y - 1) == 2u) { st(p.x, p.y - 1, 0u); }
        S(b, 1.0); S(b + 1, f32(p.x)); S(b + 2, f32(p.y));
        S(b + 3, f32(p.x)); S(b + 4, f32(p.y));
        S(b + 5, f32((e % 4) + 1)); S(b + 6, 0.0);
    }

    S(1, 1.0); S(2, 1.0); S(3, 1.0); S(4, 1.0); S(5, 0.0); S(6, cr); S(0, 1.0);
    S(9, 1.0); S(10, 1.0); S(11, 2.0); S(12, 0.0); S(13, 0.0); S(14, 0.0); S(15, u_t.time);
    for (var k = 20; k <= 37; k++) { if (k != 26) { S(k, 0.0); } }
    S(26, atime());
}

fn init() { if (G(16) < 0.5) { newg(0); S(0, 0.0); S(7, 0.0); S(16, 1.0); } }

fn upd() {
    let now = u_t.time;
    let act = gm.act != 0u;
    let edge = act && !(G(7) > 0.5);
    S(7, select(0.0, 1.0, act));
    let s = u32(G(0));

    // menu / clear / over / victory -> space
    if (s != 1u) {
        if (edge) {
            if (s == 2u) { newg(lv() + 1); } else if (s == 3u) { newg(lv()); } else { newg(0); }
        }
        return;
    }

    // move (cadence owned here)
    let px = i32(G(1)); let py = i32(G(2));
    if (gm.dir != 0u && now - G(5) > MDLY) {
        let d = dvec(gm.dir);
        let nx = px + d.x; let ny = py + d.y;
        S(11, f32(gm.dir));
        if (walk(gt(nx, ny)) && bi(nx, ny) < 0) { S(1, f32(nx)); S(2, f32(ny)); S(5, now); S(24, atime()); S(34, now); }
    }
    S(3, mix(G(3), G(1), 0.35));
    S(4, mix(G(4), G(2), 0.35));

    // power-ups and the gate
    let cx = i32(G(1)); let cy = i32(G(2)); let ct = gt(cx, cy);
    if (ct == 4u) { S(9, min(G(9) + 1.0, 4.0)); st(cx, cy, 0u); S(27, atime()); S(37, now); }
    if (ct == 5u) { S(10, min(G(10) + 1.0, 4.0)); st(cx, cy, 0u); S(27, atime()); S(37, now); }
    if (ct == 3u) { S(0, select(2.0, 4.0, lv() >= 2)); S(23, atime()); S(33, now); return; }

    // drop on the action edge
    if (edge && bi(cx, cy) < 0) { pbomb(cx, cy); }

    // detonate
    for (var i = 0; i < NB; i++) { let et = G(BB + i * 4 + 2); if (et > 0.001 && now >= et) { boom(i, now); } }

    // player death
    if (fa(cx, cy, now)) { S(0, 3.0); S(22, atime()); S(32, now); }

    // enemies: burn, move, touch
    var liv = 0;
    for (var e = 0; e < NE; e++) {
        let b = EB + e * ES;
        if (G(b) < 0.5) { continue; }
        if (fa(i32(G(b + 1)), i32(G(b + 2)), now)) { S(b, 0.0); S(25, atime()); S(35, now); continue; }
        estep(e, now);
        liv += 1;
        if (i32(G(b + 1)) == cx && i32(G(b + 2)) == cy) { S(0, 3.0); S(22, atime()); S(32, now); }
    }
    // all bugs gone: the gate in the back wall opens
    if (liv == 0 && G(14) < 0.5) { S(14, 1.0); st(GW / 2, 0, 3u); S(23, atime()); S(33, now); }
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
fn eage(k: i32, n: u32) -> f32 { let e = bitcast<u32>(G(k)); if (G(k) <= 0.0 || n + EV0 < e) { return -1.0; } return f32(n + EV0 - e) / gm.sr; }
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
    if (u32(G(0)) != 1u || G(26) <= 0.0) { return v2(0.0); }
    let e0 = bitcast<u32>(G(26));
    if (n + EV0 < e0) { return v2(0.0); }
    // each level a little faster and two semitones up
    let s16 = 60.0 / (129.0 + 6.0 * f32(lv())) / 4.0;let tp = 2.0 * f32(lv());
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
    var bass = bassv(nf(note[ci] + tp), mt - acc * s16, dur[ci] * s16 * 0.9);
    bass += bassv(nf(note[pi] + tp), mt - (acc - dur[pi]) * s16, dur[pi] * s16 * 0.9);

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
    // pickup: two quick rising FM bells
    let du = eage(27, n) - dl;
    if (du >= 0.0 && du < 0.6) {
        for (var j = 0u; j < 2u; j++) {
            let a = du - f32(j) * 0.07;
            if (a >= 0.0) {
                let f = select(1318.5, 1760.0, j == 1u);
                let w = tau * fract(f * a);
                s += v2(sin(w + 1.2 * exp(-a / 0.1) * sin(w * 2.0))) * exp(-a * 7.0) * smoothstep(0.0, 0.002, a) * 0.13;
            }
        }
    }
    return s;
}

fn snd(n: u32) -> v2 {
    var s = bgm(n) * 0.4 + sfx(n, 0.0);
    // ping-pong echo on the effects
    s += sfx(n, 0.19) * v2(0.28, 0.1) + sfx(n, 0.38) * v2(0.05, 0.14);
    return tanh(s);
}

@compute @workgroup_size(16, 16, 1)
fn sim(@builtin(global_invocation_id) gid: u3) {
    if (all(gid.xy == vec2(0u))) { init(); upd(); }
}

// ---------------- 3D ----------------
// tile (x, y) spans [x, x+1] x [y, y+1] on xz, y up
fn h21(p: v2) -> f32 { return fract(sin(dot(p, v2(127.1, 311.7))) * 43758.5453); }
fn h22(p: v2) -> v2 { return fract(sin(v2(dot(p, v2(127.1, 311.7)), dot(p, v2(269.5, 183.3)))) * 43758.5453); }
fn vn(p: v2) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + v2(1.0, 0.0)), u.x), mix(h21(i + v2(0.0, 1.0)), h21(i + v2(1.0, 1.0)), u.x), u.y);
}
fn fbm(p: v2) -> f32 { return vn(p) * 0.55 + vn(p * 2.1 + 3.7) * 0.3 + vn(p * 4.3 + 9.1) * 0.15; }
fn gyr(p: v3) -> f32 { return dot(sin(p), cos(p.yzx)); }
fn lum(c: v3) -> f32 { return dot(c, v3(0.2126, 0.7152, 0.0722)); }
fn s2l(x: v3) -> v3 { return pow(x, v3(2.2)); }

fn smin(a: f32, b: f32, k: f32) -> f32 { let h = max(k - abs(a - b), 0.0) / k; return min(a, b) - h * h * k * 0.25; }
fn sdRBox(p: v3, b: v3, r: f32) -> f32 { let q = abs(p) - b + r; return length(max(q, v3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r; }
fn sdCyl(p: v3, r: f32, h: f32) -> f32 { let d = abs(v2(length(p.xz), p.y)) - v2(r, h); return min(max(d.x, d.y), 0.0) + length(max(d, v2(0.0))); }
// apex at the origin, base radius q.x at height q.y (< 0)
fn sdCone(p: v3, q: v2) -> f32 {
    let w = v2(length(p.xz), p.y);
    let a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    let b = w - q * v2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    let k = sign(q.y);
    let d = min(dot(a, a), dot(b, b));
    let s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}
fn sdEll(p: v3, r: v3) -> f32 { let k0 = length(p / r); let k1 = length(p / (r * r)); return k0 * (k0 - 1.0) / max(k1, 1e-5); }
fn sdCap(p: v3, a: v3, b: v3, r: f32) -> f32 { let pa = p - a; let ba = b - a; return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)) - r; }
// x = distance, y = material, z = aux
fn mn(a: v3, b: v3) -> v3 { return select(a, b, b.x < a.x); }

// materials
const M_GROUND = 1.0; const M_STONE = 2.0; const M_ROOF = 3.0; const M_BUSH = 4.0; const M_BOMB = 5.0; const M_SPARK = 6.0;
const M_CRAB = 7.0; const M_CRABD = 8.0; const M_BUG = 9.0; const M_BUGD = 10.0; const M_LEAF = 11.0; const M_WOOD = 12.0;
const M_ORB = 14.0; const M_EYE = 15.0; const M_GATE = 16.0;

// game read once per workgroup into shared memory (atomic loads are slow)
// WE: 0-7 bombs (x, y, etime, range), 8-15 bugs (vx, vy, dir, alive), 16-23 bug cells, 24 crab (vx, vy, facing, state),
// 25 (crab cell x, y, blast time, blast x), 26 (blast y)
var<workgroup> WT: array<u32, 143>;
var<workgroup> WF: array<f32, 143>;
var<workgroup> WE: array<v4, 27>;
fn wload(li: u32) {
    for (var i = li; i < 143u; i += 256u) { WT[i] = u32(G(TB + i32(i))); WF[i] = G(FB + i32(i)); }
    if (li < 8u) { let b = BB + i32(li) * 4; WE[li] = v4(G(b), G(b + 1), G(b + 2), G(b + 3)); }
    else if (li < 16u) { let b = EB + i32(li - 8u) * ES; WE[li] = v4(G(b + 3), G(b + 4), G(b + 5), G(b)); WE[li + 8u] = v4(G(b + 1), G(b + 2), 0.0, 0.0); }
    else if (li == 24u) { WE[24] = v4(G(3), G(4), G(11), G(0)); WE[25] = v4(G(1), G(2), G(30), G(12)); WE[26] = v4(G(13), 0.0, 0.0, 0.0); }
}
fn gtr(x: i32, y: i32) -> u32 { if (x < 0 || y < 0 || x >= GW || y >= GH) { return 1u; } return WT[y * GW + x]; }
fn gfr(x: i32, y: i32) -> f32 { if (x < 0 || y < 0 || x >= GW || y >= GH) { return 0.0; } return WF[y * GW + x]; }

struct Th { sun: v3, sc: v3, zen: v3, hor: v3, fog: v3, amb: f32, night: f32 };
var<private> TH: Th;
var<private> NOW: f32;
var<private> LV: i32;
fn theme(l: i32) -> Th {
    if (l == 1) { return Th(normalize(v3(0.75, 0.32, -0.45)), v3(1.0, 0.55, 0.3) * 2.8, v3(0.2, 0.22, 0.48), v3(1.0, 0.6, 0.42), v3(0.78, 0.5, 0.42), 0.75, 0.0); }
    if (l == 2) { return Th(normalize(v3(-0.45, 0.62, -0.55)), v3(0.45, 0.58, 1.0) * 0.55, v3(0.01, 0.015, 0.05), v3(0.035, 0.05, 0.12), v3(0.025, 0.04, 0.09), 0.55, 1.0); }
    return Th(normalize(v3(-0.55, 0.62, -0.4)), v3(1.0, 0.9, 0.76) * 3.0, v3(0.3, 0.5, 0.85), v3(0.75, 0.83, 0.92), v3(0.68, 0.77, 0.86), 1.0, 0.0);
}

fn tower(q: v3, r: f32, h: f32, rr: f32, rh: f32) -> v3 {
    let body = sdCyl(q - v3(0.0, h * 0.5, 0.0), r, h * 0.5) - 0.012;
    let plinth = sdCyl(q - v3(0.0, 0.07, 0.0), r + 0.035, 0.07) - 0.012;
    let band = sdCyl(q - v3(0.0, h - 0.03, 0.0), r + 0.03, 0.035) - 0.01;
    var res = v3(min(min(body, plinth), band), M_STONE, 0.0);
    return mn(res, v3(sdCone(q - v3(0.0, h + rh, 0.0), v2(rr, -rh)) - 0.008, M_ROOF, 0.0));
}

// castle, one tile at a time
fn tile_sdf(p: v3, c: i2) -> v3 {
    let t = gtr(c.x, c.y);
    let q = v3(p.x - f32(c.x) - 0.5, p.y, p.z - f32(c.y) - 0.5);
    if (t == 1u) {
        let edge = c.x == 0 || c.y == 0 || c.x == GW - 1 || c.y == GH - 1;
        if (!edge) { return tower(q, 0.3, 0.95, 0.42, 0.6); }
        if ((c.x == 0 || c.x == GW - 1) && (c.y == 0 || c.y == GH - 1)) { return tower(q, 0.45, select(1.45, 0.75, c.y == GH - 1), 0.5, 0.75); }
        // walls: tall back, low front parapet
        let h = select(select(0.75, 1.05, c.y == 0), 0.32, c.y == GH - 1);
        var d = sdRBox(q - v3(0.0, h * 0.5, 0.0), v3(0.5, h * 0.5, 0.5), 0.035);
        if ((c.x + c.y) % 2 == 0) { d = min(d, sdRBox(q - v3(0.0, h + 0.09, 0.0), v3(0.27, 0.09, 0.27), 0.03)); }
        var r = v3(d, M_STONE, 0.0);
        // closed gate: a wooden door in the back wall
        if (c.y == 0 && c.x == GW / 2) { r = mn(r, v3(sdRBox(q - v3(0.0, 0.32, 0.47), v3(0.26, 0.32, 0.05), 0.03), M_WOOD, 1.0)); }
        return r;
    }
    if (t == 3u) {
        // open gate: an arch with a glowing portal
        let h = 1.05;
        let wall = sdRBox(q - v3(0.0, h * 0.5, 0.0), v3(0.5, h * 0.5, 0.5), 0.035);
        let hole = min(sdRBox(q - v3(0.0, 0.3, 0.0), v3(0.27, 0.3, 0.6), 0.0), length(q.xy - v2(0.0, 0.6)) - 0.27);
        let r = v3(max(wall, -hole), M_STONE, 0.0);
        return mn(r, v3(sdRBox(q - v3(0.0, 0.45, 0.1), v3(0.27, 0.45, 0.02), 0.0), M_GATE, 0.0));
    }
    if (t == 2u) {
        let cq = q - v3(0.0, 0.34, 0.0);
        let bnd = length(cq) - 0.5;
        if (bnd > 0.12) { return v3(bnd, M_BUSH, 0.0); }
        let o = h22(v2(c) * 1.31);
        var d = length(cq) - 0.33;
        d = smin(d, length(cq - v3(0.14 * (o.x - 0.5) * 2.0, 0.13, 0.1)) - 0.22, 0.12);
        d = smin(d, length(cq - v3(-0.13, 0.08, -0.12 * (o.y - 0.5) * 2.0)) - 0.23, 0.12);
        d += 0.045 * gyr(cq * 10.0 + o.x * 30.0) + 0.02 * gyr(cq * 21.0);
        return v3(d * 0.7, M_BUSH, o.x);
    }
    if (t == 4u || t == 5u) {
        let y = 0.36 + 0.05 * sin(NOW * 3.0 + f32(c.x));
        return v3(length(q - v3(0.0, y, 0.0)) - 0.13, M_ORB, f32(t));
    }
    return v3(1e5, 0.0, 0.0);
}

// forest, one tree per tile
fn tree_sdf(p: v3, c: i2) -> v3 {
    if (c.x >= -2 && c.x <= GW + 1 && c.y >= -2 && c.y <= GH + 1) { return v3(1e5, 0.0, 0.0); }
    let ce = v2(c) + 0.5;
    let h = h22(ce * 1.7);
    if (h.x < 0.1) { return v3(1e5, 0.0, 0.0); }
    let q = v3(p.x - ce.x, p.y, p.z - ce.y);
    let R = 0.34 + 0.1 * h.y;
    let H = 0.75 + 0.7 * h21(ce * 3.1);
    let trunk = sdCyl(q - v3(0.0, H * 0.5, 0.0), 0.055, H * 0.5);
    let cc = q - v3(0.0, H + R * 0.35, 0.0);
    let bnd = length(cc - v3(0.0, R * 0.4, 0.0)) - R * 1.5;
    var r = v3(trunk, M_WOOD, 0.0);
    if (bnd > 0.1) { return mn(r, v3(bnd, M_LEAF, h.x)); }
    var d = smin(length(cc) - R, length(cc - v3(0.05, R * 0.85, -0.03)) - R * 0.72, 0.15);
    d += 0.05 * gyr(cc * 9.0 + h.x * 20.0) + 0.022 * gyr(cc * 19.0 + 3.0);
    return mn(r, v3(d * 0.7, M_LEAF, h.x));
}

fn smap(p: v3) -> v3 {
    let c = i2(floor(p.xz));
    if (c.x >= 0 && c.x < GW && c.y >= 0 && c.y < GH) { return tile_sdf(p, c); }
    return tree_sdf(p, c);
}

// facing from a move direction: forward vector in xz
fn fwd(d: u32) -> v2 {
    if (d == 1u) { return v2(0.0, -1.0); }
    if (d == 3u) { return v2(-1.0, 0.0); }
    if (d == 4u) { return v2(1.0, 0.0); }
    return v2(0.0, 1.0);
}
fn local(p: v3, c: v3, f: v2) -> v3 { let d = p - c; return v3(d.x * f.y - d.z * f.x, d.y, d.x * f.x + d.z * f.y); }

// Ferris, in its own frame: +z forward
fn crab(q: v3, walk_: f32) -> v3 {
    var d = sdEll(q - v3(0.0, 0.27, 0.0), v3(0.34, 0.16, 0.27));
    // spiky rim
    let an = atan2(q.x, -q.z);
    d -= 0.03 * max(0.0, sin(an * 7.0)) * smoothstep(0.1, 0.3, q.y) * step(q.z, 0.05);
    var r = v3(d, M_CRAB, 0.0);
    for (var i = 0; i < 2; i++) {
        let s = f32(i) * 2.0 - 1.0;
        let arm = sdCap(q, v3(s * 0.26, 0.24, 0.12), v3(s * 0.38, 0.3, 0.3), 0.045);
        var claw = sdEll(q - v3(s * 0.38, 0.32, 0.38), v3(0.1, 0.075, 0.12));
        claw = max(claw, -(length(q - v3(s * 0.38, 0.32, 0.52)) - 0.07));
        r = mn(r, v3(smin(arm, claw, 0.04), M_CRAB, 0.0));
        for (var k = 0; k < 3; k++) {
            let z = -0.12 + f32(k) * 0.11;
            let sw = sin(walk_ * 14.0 + f32(k) * 2.1 + s) * 0.05;
            r = mn(r, v3(sdCap(q, v3(s * 0.24, 0.2, z), v3(s * 0.44, 0.02, z + sw), 0.028), M_CRABD, 0.0));
        }
        r = mn(r, v3(sdCap(q, v3(s * 0.09, 0.36, 0.1), v3(s * 0.11, 0.5, 0.13), 0.022), M_CRAB, 0.0));
        r = mn(r, v3(length(q - v3(s * 0.11, 0.53, 0.14)) - 0.06, M_EYE, 0.0));
    }
    return r;
}

fn beetle(q: v3, walk_: f32, e: f32) -> v3 {
    let body = sdEll(q - v3(0.0, 0.25, -0.03), v3(0.24, 0.17, 0.31));
    var r = v3(body, M_BUG, e);
    r = mn(r, v3(length(q - v3(0.0, 0.21, 0.27)) - 0.12, M_BUGD, e));
    for (var i = 0; i < 2; i++) {
        let s = f32(i) * 2.0 - 1.0;
        r = mn(r, v3(sdCap(q, v3(s * 0.05, 0.28, 0.33), v3(s * 0.14, 0.44, 0.5), 0.013), M_BUGD, e));
        r = mn(r, v3(sdCap(q, v3(s * 0.05, 0.16, 0.37), v3(s * 0.1, 0.12, 0.45), 0.02), M_BUGD, e));
        for (var k = 0; k < 3; k++) {
            let z = -0.15 + f32(k) * 0.15;
            let sw = sin(walk_ * 16.0 + f32(k) * 2.4 + s * 1.5) * 0.06;
            r = mn(r, v3(sdCap(q, v3(s * 0.2, 0.18, z), v3(s * 0.38, 0.015, z + sw), 0.022), M_BUGD, e));
        }
    }
    return r;
}

fn bomb(q: v3, life: f32) -> v3 {
    let s = 1.0 + 0.06 * sin(NOW * mix(8.0, 26.0, 1.0 - life));
    var r = v3(length(q - v3(0.0, 0.27, 0.0)) - 0.25 * s, M_BOMB, 0.0);
    r = mn(r, v3(sdCyl(q - v3(0.0, 0.53, 0.0), 0.07, 0.04) - 0.01, M_BOMB, 0.0));
    r = mn(r, v3(sdCap(q, v3(0.0, 0.57, 0.0), v3(0.07, 0.68, 0.04), 0.016), M_WOOD, 0.0));
    return mn(r, v3(length(q - v3(0.07, 0.7, 0.04)) - 0.035 * (1.0 + 0.4 * sin(NOW * 40.0)), M_SPARK, 0.0));
}

// moving things (bounding-sphere culled)
fn emap(p: v3, best: f32) -> v3 {
    var r = v3(1e5, 0.0, 0.0);
    let s = u32(WE[24].w);
    for (var i = 0; i < NB; i++) {
        let bb = WE[i];
        if (bb.z <= 0.001) { continue; }
        let c = v3(bb.x + 0.5, 0.0, bb.y + 0.5);
        let bd = length(p - c - v3(0.0, 0.35, 0.0)) - 0.5;
        if (bd > min(best, r.x)) { r = mn(r, v3(bd, 0.0, 0.0)); continue; }
        r = mn(r, bomb(p - c, clamp((bb.z - NOW) / FUSE, 0.0, 1.0)));
    }
    for (var e = 0; e < NE; e++) {
        let bg = WE[8 + e];
        if (bg.w < 0.5) { continue; }
        let c = v3(bg.x + 0.5, 0.0, bg.y + 0.5);
        let bd = length(p - c - v3(0.0, 0.25, 0.0)) - 0.62;
        if (bd > min(best, r.x)) { r = mn(r, v3(bd, 0.0, 0.0)); continue; }
        let mv = length(WE[16 + e].xy - bg.xy);
        let q = local(p, c + v3(0.0, 0.015 * sin(NOW * 9.0 + f32(e)), 0.0), fwd(u32(bg.z)));
        r = mn(r, beetle(q, NOW + f32(e) * 0.37 + mv * 2.0, f32(e)));
    }
    if (s != 0u) {
        let cb = WE[24];
        let c = v3(cb.x + 0.5, 0.0, cb.y + 0.5);
        let bd = length(p - c - v3(0.0, 0.3, 0.0)) - 0.66;
        if (bd > min(best, r.x)) { r = mn(r, v3(bd, 0.0, 0.0)); }
        else {
            let mv = length(WE[25].xy - cb.xy);
            let sq = select(1.0, 0.35, s == 3u);
            var q = local(p, c, fwd(u32(cb.z)));
            q.y /= sq;
            var cr = crab(q, NOW * (0.2 + mv * 4.0));
            cr.x *= sq;
            r = mn(r, cr);
        }
    }
    return r;
}

fn map(p: v3) -> v3 { let a = smap(p); return mn(a, emap(p, a.x)); }

// distance to the next tile border
fn cell_exit(p: v3, rd: v3) -> f32 {
    let fx = select(floor(p.x) - p.x, floor(p.x) + 1.0 - p.x, rd.x > 0.0) / select(rd.x, 1e-6, abs(rd.x) < 1e-6);
    let fz = select(floor(p.z) - p.z, floor(p.z) + 1.0 - p.z, rd.z > 0.0) / select(rd.z, 1e-6, abs(rd.z) < 1e-6);
    return max(min(abs(fx), abs(fz)), 0.0) + 0.002;
}

// nothing stands taller (tree crowns top out near 2.36)
const TOP: f32 = 2.45;

// fire column on a burning tile
fn fire(p: v3) -> f32 {
    let c = i2(floor(p.xz));
    if (c.x < 0 || c.y < 0 || c.x >= GW || c.y >= GH || p.y > 1.4) { return 0.0; }
    let fe = gfr(c.x, c.y);
    if (fe <= NOW) { return 0.0; }
    let life = clamp((fe - NOW) / FDUR, 0.0, 1.0);
    let q = v3(p.x - f32(c.x) - 0.5, p.y, p.z - f32(c.y) - 0.5);
    let n = gyr(v3(q.x * 6.0, q.y * 5.0 - NOW * 9.0, q.z * 6.0) + f32(c.x * 7 + c.y)) * 0.5 + 0.5;
    let rad = 0.42 * (1.0 - q.y / 1.4) * (0.6 + 0.6 * life);
    let core = 1.0 - length(q.xz) / max(rad, 0.01);
    return clamp(core * (0.6 + 0.7 * n), 0.0, 1.0) * smoothstep(1.4, 0.2, q.y / max(life, 0.3)) * life;
}

struct Ray { t: f32, m: f32, a: f32, fire: v3, tr: f32 };
fn march(ro: v3, rd: v3, tmax: f32, cone: f32) -> Ray {
    var t = max((TOP - ro.y) / rd.y, 0.0);
    var r = Ray(tmax, M_GROUND, 0.0, v3(0.0), 1.0);
    // over-relaxed marching (Keinert et al. 2014)
    var pt = t; var pr = 0.0; var tent = false;
    for (var i = 0; i < 160; i++) {
        if (t >= tmax) { break; }
        let p = ro + rd * t;
        let h = map(p);
        let rs = h.x * 0.9;
        if (tent && pr + max(rs, 0.0) < t - pt) { t = pt + pr; tent = false; continue; }
        let ce = cell_exit(p, rd);
        var stp = min(rs, ce);
        tent = false;
        if (rs * 1.5 < ce) { stp = rs * 1.5; tent = true; }
        pt = t; pr = rs;
        // flame volume
        let fd = fire(p);
        if (fd > 0.0 || (p.y < 1.4 && gfr(i32(floor(p.x)), i32(floor(p.z))) > NOW)) {
            stp = min(stp, 0.05);
            let fc = mix(v3(3.0, 0.35, 0.05), v3(5.0, 3.6, 1.4), fd * fd);
            r.fire += fc * fd * r.tr * 0.05 * 9.0;
            r.tr *= exp(-fd * 0.05 * 6.0);
        }
        if (h.x < max(0.0008, t * cone)) { r.t = t; r.m = h.y; r.a = h.z; return r; }
        if (stp == 0.05) { tent = false; }
        t += max(stp, 0.002);
    }
    return r;
}

fn normal(p: v3, e: f32) -> v3 {
    let k = v2(1.0, -1.0);
    return normalize(k.xyy * map(p + k.xyy * e).x + k.yyx * map(p + k.yyx * e).x + k.yxy * map(p + k.yxy * e).x + k.xxx * map(p + k.xxx * e).x);
}

// soft sun shadow, tile-border stepping
fn shadow(ro: v3, rd: v3) -> f32 {
    var t = 0.02; var res = 1.0;
    for (var i = 0; i < 48; i++) {
        let p = ro + rd * t;
        if (p.y > TOP) { break; }
        let h = map(p).x;
        res = min(res, 10.0 * h / t);
        if (res < 0.01) { return 0.0; }
        t += clamp(min(h, cell_exit(p, rd)), 0.01, 0.6);
    }
    return clamp(res, 0.0, 1.0);
}

fn occl(p: v3, n: v3) -> f32 {
    var o = 0.0; var w = 1.0;
    for (var i = 0; i < 4; i++) {
        let h = 0.04 + 0.12 * f32(i * i);
        o += w * max(0.0, h - map(p + n * h).x) / h;
        w *= 0.6;
    }
    return clamp(1.0 - o * 0.55, 0.15, 1.0);
}

// cobbles (voronoi): border distance + id
fn cobble(p: v2) -> v2 {
    let i = floor(p); var d1 = 9.0; var d2 = 9.0; var id = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = i + v2(f32(x), f32(y));
            let o = c + 0.15 + 0.7 * h22(c);
            let d = length(p - o);
            if (d < d1) { d2 = d1; d1 = d; id = h21(c); } else if (d < d2) { d2 = d; }
        }
    }
    return v2(d2 - d1, id);
}

struct Mat { alb: v3, rough: f32, em: v3, wrap: f32, bump: v3 };

fn material(p: v3, n: v3, m: f32, aux: f32) -> Mat {
    var r = Mat(v3(0.5), 0.8, v3(0.0), 0.0, v3(0.0));
    let night = TH.night;
    if (m == M_GROUND) {
        let inside = p.x > 0.0 && p.x < f32(GW) && p.z > 0.0 && p.z < f32(GH);
        if (inside) {
            let cb = cobble(p.xz * 3.2);
            let stone = mix(v3(0.5, 0.47, 0.42), v3(0.62, 0.58, 0.5), cb.y) * (0.85 + 0.3 * vn(p.xz * 11.0));
            let moss = v3(0.25, 0.33, 0.14);
            let gap = smoothstep(0.02, 0.12, cb.x);
            r.alb = mix(mix(moss * 0.7, v3(0.2, 0.18, 0.15), 0.5), mix(stone, moss, smoothstep(0.55, 0.8, fbm(p.xz * 1.3)) * 0.5), gap);
            r.rough = 0.85;
            let e = 0.01;
            let gx = smoothstep(0.02, 0.12, cobble((p.xz + v2(e, 0.0)) * 3.2).x) - gap;
            let gz = smoothstep(0.02, 0.12, cobble((p.xz + v2(0.0, e)) * 3.2).x) - gap;
            r.bump = v3(-gx, 0.0, -gz) * 0.6;
        } else {
            let g = fbm(p.xz * 0.9);
            r.alb = mix(v3(0.16, 0.3, 0.08), v3(0.3, 0.42, 0.12), g) * (0.85 + 0.25 * vn(p.xz * 14.0));
            // a dirt ring around the walls
            let dx = max(max(-p.x, p.x - f32(GW)), max(-p.z, p.z - f32(GH)));
            r.alb = mix(v3(0.36, 0.27, 0.17), r.alb, smoothstep(0.35, 1.1, dx + 0.3 * vn(p.xz * 3.0)));
            // flowers
            let fl = h22(floor(p.xz * 9.0));
            if (fl.x > 0.93 && dx > 1.0) { r.alb = select(v3(0.95, 0.92, 0.85), v3(0.85, 0.45, 0.85), fl.y > 0.5); }
            r.rough = 0.95; r.wrap = 0.3;
        }
        if (LV == 1) { r.alb = mix(r.alb, r.alb * v3(1.25, 0.95, 0.6), 0.5); }
    } else if (m == M_STONE) {
        // ashlar + moss
        let row = floor(p.y / 0.16);
        let u = (p.x + p.z) / 0.3 + row * 0.5;
        let bid = h21(v2(floor(u), row));
        let mortar = min(1.0 - smoothstep(0.84, 0.94, abs(fract(p.y / 0.16) - 0.5) * 2.0), smoothstep(0.0, 0.08, 1.0 - abs(fract(u) - 0.5) * 2.0));
        var s = mix(v3(0.58, 0.54, 0.47), v3(0.72, 0.67, 0.58), bid) * (0.9 + 0.2 * vn(p.xz * 9.0 + p.y * 7.0));
        s = mix(s, v3(0.27, 0.36, 0.15), smoothstep(0.25, 0.0, p.y) * 0.6 * smoothstep(0.3, 0.7, vn(p.xz * 4.0)));
        r.alb = mix(s * 0.55, s, mortar);
        r.rough = 0.9;
        // lit windows on the towers at night
        let ce = floor(p.xz) + 0.5;
        let w = abs(p.x - ce.x) < 0.07 && p.y > 0.45 && p.y < 0.68 && n.z > 0.6;
        if (w) { r.alb = v3(0.05); r.em = v3(1.6, 0.9, 0.35) * night * (0.8 + 0.2 * sin(NOW * 3.0 + ce.x * 5.0)); }
    } else if (m == M_ROOF) {
        // shingles: scalloped rows
        let ce = floor(p.xz) + 0.5;
        let an = atan2(p.z - ce.y, p.x - ce.x);
        let rw = p.y / 0.07;
        let sc = abs(fract(an * 4.0 + floor(rw) * 0.5) - 0.5);
        let sh = smoothstep(0.0, 0.25, fract(rw) - sc * 0.4);
        let tint = h21(ce);
        let base = select(mix(v3(0.55, 0.22, 0.12), v3(0.45, 0.28, 0.16), tint), mix(v3(0.22, 0.25, 0.32), v3(0.3, 0.28, 0.35), tint), tint > 0.6);
        r.alb = base * (0.6 + 0.4 * sh);
        r.rough = 0.7;
    } else if (m == M_BUSH || m == M_LEAF) {
        var g1 = v3(0.17, 0.34, 0.1); var g2 = v3(0.33, 0.48, 0.14);
        if (LV == 1) { g1 = mix(v3(0.5, 0.22, 0.05), v3(0.3, 0.32, 0.08), aux); g2 = mix(v3(0.85, 0.5, 0.12), v3(0.6, 0.45, 0.1), aux); }
        let k = 0.5 + 0.5 * gyr(p * 13.0);
        r.alb = mix(g1, g2, k * (0.6 + 0.4 * aux));
        if (m == M_LEAF) { r.alb *= 0.8 + 0.25 * aux; }
        // small blossoms on some bushes
        if (m == M_BUSH && aux > 0.6 && h21(floor(p.xz * 18.0) + floor(p.y * 18.0)) > 0.94) { r.alb = v3(0.95, 0.6, 0.75); }
        r.rough = 0.85; r.wrap = 0.5;
    } else if (m == M_BOMB) { r.alb = v3(0.04, 0.045, 0.06); r.rough = 0.25; }
    else if (m == M_SPARK) { r.alb = v3(1.0); r.em = v3(6.0, 3.2, 1.0) * (0.7 + 0.3 * sin(NOW * 50.0)); }
    else if (m == M_CRAB) { r.alb = v3(0.95, 0.3, 0.05); r.rough = 0.35; r.wrap = 0.25; }
    else if (m == M_CRABD) { r.alb = v3(0.65, 0.16, 0.03); r.rough = 0.5; }
    else if (m == M_BUG || m == M_BUGD) {
        var pal = array<v3, 4>(v3(0.3, 0.55, 0.12), v3(0.48, 0.18, 0.62), v3(0.1, 0.45, 0.6), v3(0.65, 0.42, 0.08));
        let tint = pal[u32(aux) % 4u];
        r.alb = select(tint, v3(0.03, 0.03, 0.035), m == M_BUGD);
        r.rough = select(0.22, 0.4, m == M_BUGD);
    } else if (m == M_WOOD) { r.alb = v3(0.3, 0.19, 0.1) * (0.8 + 0.3 * vn(v2(p.y * 30.0, p.x + p.z))); r.rough = 0.8; }
    else if (m == M_ORB) {
        let c = select(v3(0.3, 1.2, 3.0), v3(4.0, 1.2, 0.3), aux == 4.0);
        r.alb = c * 0.2; r.em = c * (0.8 + 0.3 * sin(NOW * 6.0)); r.rough = 0.1;
    } else if (m == M_EYE) { r.alb = v3(0.95); r.rough = 0.1; }
    else if (m == M_GATE) {
        let sw = 0.5 + 0.5 * sin(atan2(p.y - 0.45, p.x - floor(p.x) - 0.5) * 3.0 + NOW * 2.0 + length(v2(p.x - floor(p.x) - 0.5, p.y - 0.45)) * 18.0);
        r.alb = v3(0.0); r.em = mix(v3(0.3, 0.7, 2.0), v3(1.4, 2.0, 3.0), sw);
    }
    return r;
}

fn sky(d: v3) -> v3 { return mix(TH.hor, TH.zen, smoothstep(-0.05, 0.6, d.y)); }

fn shade(p: v3, n0: v3, rd: v3, m: f32, aux: f32, foot: f32) -> v3 {
    var mt = material(p, n0, m, aux);
    let n = normalize(n0 + mt.bump);
    let v = -rd;
    let ao = occl(p, n0);
    let l = TH.sun;
    let nl = dot(n, l);
    // facing away from the sun: no light to shadow
    var sh = 0.0;
    if (nl > -mt.wrap) { sh = shadow(p + n0 * 0.015, l); }
    let dif = clamp((nl + mt.wrap) / (1.0 + mt.wrap), 0.0, 1.0);
    // specular never sharper than the pixel can hold
    let rough = sqrt(mt.rough * mt.rough + min(foot * foot * 40.0, 0.3));
    let h = normalize(l + v);
    let a2 = rough * rough * rough * rough;
    let nh = max(dot(n, h), 0.0);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let fre = 0.04 + 0.96 * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    let spec = a2 / (3.14159 * dd * dd) * fre * max(nl, 0.0) * 0.25;
    var col = mt.alb * TH.sc * dif * sh + TH.sc * spec * sh * (1.0 - rough * 0.5);
    // sky, ground bounce and a cool fill opposite the sun
    col += mt.alb * sky(n) * TH.amb * ao * (0.55 + 0.45 * n.y);
    col += mt.alb * v3(0.25, 0.3, 0.12) * 0.25 * ao * clamp(-n.y * 0.5 + 0.5, 0.0, 1.0) * (1.0 - TH.night * 0.8);
    col += mt.alb * TH.zen * 0.4 * max(dot(n, normalize(v3(-l.x, 0.3, -l.z))), 0.0) * ao;
    // blast light, and the burning fuses
    let ba = NOW - WE[25].z;
    if (WE[25].z > 0.0 && ba < 0.8) {
        let bp = v3(WE[25].w + 0.5, 0.45, WE[26].x + 0.5) - p;
        let bl = length(bp);
        col += mt.alb * v3(1.0, 0.55, 0.2) * 14.0 * exp(-ba * 5.0) * max(dot(n, bp / bl) * 0.7 + 0.3, 0.0) / (1.0 + bl * bl * 1.5) * ao;
    }
    for (var i = 0; i < NB; i++) {
        if (WE[i].z <= 0.001) { continue; }
        let bp = v3(WE[i].x + 0.57, 0.7, WE[i].y + 0.54) - p;
        let bl = length(bp);
        col += mt.alb * v3(1.0, 0.6, 0.25) * 0.5 * max(dot(n, bp / bl), 0.0) / (0.05 + bl * bl) * ao * (0.8 + 0.2 * sin(NOW * 50.0));
    }
    // the crab's pupils look ahead
    if (m == M_EYE) {
        let f = fwd(u32(WE[24].z));
        if (dot(n, normalize(v3(f.x, 0.25, f.y))) > 0.75) { col = v3(0.01); }
        col += v3(1.0) * pow(max(dot(reflect(rd, n), l), 0.0), 60.0) * 2.0;
    }
    // wet lacquer on the shells
    if (m == M_BUG || m == M_CRAB || m == M_BOMB) {
        let rf = reflect(rd, n);
        col += sky(rf) * (0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0)) * ao * 0.8;
    }
    return col + mt.em;
}

fn cam(px: v2, D: v2, ro: ptr<function, v3>, rd: ptr<function, v3>) {
    let asp = D.x / D.y; let tf = 0.27;
    var tg = v3(6.5, 0.2, 6.0);
    let ba = NOW - WE[25].z;
    if (WE[25].z > 0.0 && ba > 0.0 && ba < 0.35) { tg += v3(sin(NOW * 90.0), 0.0, cos(NOW * 97.0)) * (1.0 - ba / 0.35) * 0.06; }
    let dir = normalize(v3(0.0, 1.2, 1.0));
    let dist = max(8.8 / (tf * asp), 6.6 / tf);
    let f = -dir; let rt = normalize(cross(f, v3(0.0, 1.0, 0.0))); let up = cross(rt, f);
    let uv = (px - 0.5 * D) / D.y * 2.0;
    *ro = tg + dir * dist;
    *rd = normalize(f + (uv.x * rt - uv.y * up) * tf);
}

@compute @workgroup_size(16, 16, 1)
fn scene(@builtin(global_invocation_id) gid: u3, @builtin(local_invocation_index) li: u32) {
    wload(li);
    workgroupBarrier();
    let D = v2(textureDimensions(out));
    if (any(v2(gid.xy) >= D)) { return; }
    NOW = u_t.time; LV = lv(); TH = theme(LV);
    var ro: v3; var rd: v3;
    cam(v2(gid.xy) + 0.5, D, &ro, &rd);
    let cone = 0.54 / D.y;
    let tg = -ro.y / rd.y;
    let r = march(ro, rd, tg, cone);
    let p = ro + rd * r.t;
    let foot = r.t * cone;
    var n = v3(0.0, 1.0, 0.0);
    if (r.m != M_GROUND) { n = normal(p, max(0.0015, foot * 0.5)); }
    var col = shade(p, n, rd, r.m, r.a, foot);
    col = col * r.tr + r.fire;

    // fireflies at dusk and night
    let nff = select(0, select(10, 28, TH.night > 0.5), LV >= 1);
    for (var i = 0; i < nff; i++) {
        let h = h22(v2(f32(i), 7.3));
        let fp = v3(-1.5 + h.x * 16.0 + sin(NOW * 0.3 + f32(i)) * 0.6, 0.4 + 0.9 * h21(v2(f32(i), 2.1)) + 0.2 * sin(NOW * 0.7 + f32(i) * 2.0), -1.5 + h.y * 14.0 + cos(NOW * 0.25 + f32(i) * 1.3) * 0.6);
        let tt = dot(fp - ro, rd);
        if (tt < 0.0 || tt > r.t) { continue; }
        let dd = length(ro + rd * tt - fp);
        let bl = pow(max(0.0, sin(NOW * (1.3 + h.x) + f32(i) * 4.0)), 3.0);
        col += v3(1.0, 0.85, 0.35) * bl * (exp(-dd * dd / (foot * foot * 9.0 + 1e-5)) * 2.5 + exp(-dd * 22.0) * 0.25);
    }

    // haze, thicker near the ground and far away
    let fogd = 1.0 - exp(-max(r.t - 20.0, 0.0) * 0.05 - max(1.0 - p.y, 0.0) * 0.0);
    col = mix(col, TH.fog, clamp(fogd, 0.0, 1.0));
    textureStore(out, vec2<i32>(gid.xy), v4(max(col, v3(0.0)), r.t));
}

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
fn pxuv(id: u3) -> v2 { return (v2(id.xy) + 0.5) / v2(textureDimensions(out)); }
@compute @workgroup_size(16, 16, 1)
fn bd1(@builtin(global_invocation_id) id: u3) {
    if (any(id.xy >= textureDimensions(out))) { return; }
    let c = dsmp(t0, s0, pxuv(id)); let m = max(c.x, max(c.y, c.z));
    textureStore(out, id.xy, v4(c * max(m - 1.0, 0.0) / max(m, 1e-4), 1.0));
}
@compute @workgroup_size(16, 16, 1)
fn bd2(@builtin(global_invocation_id) id: u3) { if (any(id.xy >= textureDimensions(out))) { return; } textureStore(out, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd3(@builtin(global_invocation_id) id: u3) { if (any(id.xy >= textureDimensions(out))) { return; } textureStore(out, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd4(@builtin(global_invocation_id) id: u3) { if (any(id.xy >= textureDimensions(out))) { return; } textureStore(out, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu3(@builtin(global_invocation_id) id: u3) { if (any(id.xy >= textureDimensions(out))) { return; } textureStore(out, id.xy, v4(usmp(pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu2(@builtin(global_invocation_id) id: u3) { if (any(id.xy >= textureDimensions(out))) { return; } textureStore(out, id.xy, v4(usmp(pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu1(@builtin(global_invocation_id) id: u3) { if (any(id.xy >= textureDimensions(out))) { return; } textureStore(out, id.xy, v4(usmp(pxuv(id)), 1.0)); }
// tilt-shift source: plain blur pyramid of the scene
@compute @workgroup_size(16, 16, 1)
fn db1(@builtin(global_invocation_id) id: u3) { if (any(id.xy >= textureDimensions(out))) { return; } textureStore(out, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn db2(@builtin(global_invocation_id) id: u3) { if (any(id.xy >= textureDimensions(out))) { return; } textureStore(out, id.xy, v4(dsmp(t0, s0, pxuv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn du1(@builtin(global_invocation_id) id: u3) {
    if (any(id.xy >= textureDimensions(out))) { return; }
    let uv = pxuv(id); let lp = 1.0 / v2(textureDimensions(t1));
    var b = v3(0.0);
    for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) {
        b += textureSampleLevel(t1, s1, cuv(uv + v2(f32(i), f32(j)) * lp, t1), 0.0).xyz * f32((2 - abs(i)) * (2 - abs(j)));
    } }
    textureStore(out, id.xy, v4(mix(textureSampleLevel(t0, s0, cuv(uv, t0), 0.0).xyz, b / 16.0, 0.75), 1.0));
}

fn hud(pp: v2, ss: v2) -> v4 {
    let s = u32(G(0));
    // only the text bands
    if (s == 1u) { if (pp.y > 110.0) { return v4(0.0); } }
    else if (pp.y < ss.y * 0.36 - 8.0 || pp.y > ss.y * 0.36 + 190.0) { return v4(0.0); }
    var c = array<u32, 16>(32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);
    var a = 0.0; var tc = v3(1.0);
    if (s == 0u) {
        // BUGBERMAN
        c = array<u32, 16>(66u, 85u, 71u, 66u, 69u, 82u, 77u, 65u, 78u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);
        a = word(pp, v2(ss.x * 0.5 - 9.0 * adv(72.0) * 0.5, ss.y * 0.36), c, 9u, 72.0); tc = v3(1.0, 0.86, 0.45);
        // PRESS SPACE
        c = array<u32, 16>(80u, 82u, 69u, 83u, 83u, 32u, 83u, 80u, 65u, 67u, 69u, 32u, 32u, 32u, 32u, 32u);
        let b = word(pp, v2(ss.x * 0.5 - 11.0 * adv(30.0) * 0.5, ss.y * 0.36 + 100.0), c, 11u, 30.0);
        if (b > a) { a = b; tc = v3(1.0, 0.95, 0.85); }
        // ARROWS MOVE
        c = array<u32, 16>(65u, 82u, 82u, 79u, 87u, 83u, 32u, 77u, 79u, 86u, 69u, 32u, 32u, 32u, 32u, 32u);
        let k = word(pp, v2(ss.x * 0.5 - 11.0 * adv(22.0) * 0.5, ss.y * 0.36 + 150.0), c, 11u, 22.0);
        if (k > a) { a = k; tc = v3(0.8, 0.85, 0.9); }
    } else if (s == 1u) {
        // LEVEL n  BUGS
        c = array<u32, 16>(76u, 69u, 86u, 69u, 76u, 32u, 49u + u32(lv()), 32u, 32u, 66u, 85u, 71u, 83u, 32u, 32u, 32u);
        a = word(pp, v2(28.0, 22.0), c, 13u, 34.0);
        let b = num(pp, v2(28.0 + 14.0 * adv(34.0), 22.0), nliv(), 34.0);
        if (b > a) { a = b; tc = v3(1.0, 0.6, 0.45); }
        // FIRE n  BOMBS n
        c = array<u32, 16>(70u, 73u, 82u, 69u, 32u, 48u + u32(G(9)), 32u, 32u, 66u, 79u, 77u, 66u, 83u, 32u, 48u + u32(G(10)), 32u);
        let f = word(pp, v2(ss.x - 16.0 * adv(28.0) - 28.0, 26.0), c, 15u, 28.0);
        if (f > a) { a = f; tc = v3(1.0, 0.85, 0.6); }
        if (G(14) > 0.5) {
            // THE GATE IS OPEN
            c = array<u32, 16>(84u, 72u, 69u, 32u, 71u, 65u, 84u, 69u, 32u, 73u, 83u, 32u, 79u, 80u, 69u, 78u);
            let g = word(pp, v2(ss.x * 0.5 - 16.0 * adv(30.0) * 0.5, 70.0), c, 16u, 30.0) * (0.6 + 0.4 * sin(NOW * 4.0));
            if (g > a) { a = g; tc = v3(0.6, 0.85, 1.0); }
        }
    } else {
        var n = 9u; tc = v3(1.0, 0.3, 0.25);
        // GAME OVER
        c = array<u32, 16>(71u, 65u, 77u, 69u, 32u, 79u, 86u, 69u, 82u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);
        // LEVEL CLEAR
        if (s == 2u) { c = array<u32, 16>(76u, 69u, 86u, 69u, 76u, 32u, 67u, 76u, 69u, 65u, 82u, 32u, 32u, 32u, 32u, 32u); n = 11u; tc = v3(0.6, 1.0, 0.65); }
        // YOU WIN!
        if (s == 4u) { c = array<u32, 16>(89u, 79u, 85u, 32u, 87u, 73u, 78u, 33u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u); n = 8u; tc = v3(1.0, 0.85, 0.4); }
        a = word(pp, v2(ss.x * 0.5 - f32(n) * adv(68.0) * 0.5, ss.y * 0.38), c, n, 68.0);
        c = array<u32, 16>(80u, 82u, 69u, 83u, 83u, 32u, 83u, 80u, 65u, 67u, 69u, 32u, 32u, 32u, 32u, 32u);
        let b = word(pp, v2(ss.x * 0.5 - 11.0 * adv(28.0) * 0.5, ss.y * 0.38 + 90.0), c, 11u, 28.0);
        if (b > a) { a = b; tc = v3(0.95); }
    }
    return v4(tc, a);
}

fn aces(x: v3) -> v3 { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), v3(0.0), v3(1.0)); }

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) gid: u3, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: u3, @builtin(num_workgroups) nw: u3) {
    // audio: one thread per sample
    let ai = (wid.y * nw.x + wid.x) * 256u + li;
    if (ai < gm.sn) { let v = snd(gm.so + ai) * gm.vol; au[ai * 2u] = v.x; au[ai * 2u + 1u] = v.y; }

    let ss = v2(textureDimensions(out));
    let pp = v2(gid.xy);
    if (any(pp >= ss)) { return; }
    NOW = u_t.time;
    let uv = (pp + 0.5) / ss;
    let s = u32(G(0));

    // tilt-shift
    let sc = textureLoad(t0, vec2<i32>(gid.xy), 0);
    let bl = textureSampleLevel(t2, s2, cuv(uv, t2), 0.0).xyz;
    let coc = smoothstep(0.18, 0.5, abs(uv.y - 0.56)) * gm.tsh;
    var col = mix(sc.xyz, bl, clamp(coc, 0.0, 1.0));
    col += textureSampleLevel(t1, s1, cuv(uv, t1), 0.0).xyz * gm.blm * 0.35;
    if (G(30) > 0.0 && NOW - G(30) > 0.0 && NOW - G(30) < 0.12) { col += v3(1.0, 0.75, 0.4) * (1.0 - (NOW - G(30)) / 0.12) * 0.3; }

    // grade per level, then filmic curve
    let l = lv();
    var ex = select(select(1.1, 1.0, l == 1), 1.0, l == 0);
    col = aces(col * ex);
    let lu = lum(col);
    if (l == 1) { col = mix(col, col * v3(1.08, 0.97, 0.88), 0.6); }
    if (l == 2) { col = mix(col * v3(0.85, 0.95, 1.15), col * v3(1.15, 1.0, 0.8), smoothstep(0.25, 0.7, lu)); }
    col = mix(v3(lu), col, 1.1);
    col *= 0.72 + 0.28 * pow(16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.18);
    if (s == 0u || s >= 2u) { col *= select(0.55, 0.7, s == 2u || s == 4u); }
    if (s == 3u) { col = mix(col, col * v3(1.0, 0.4, 0.35), 0.4); }

    // hud: soft shadow under the letters
    let h = hud(pp, ss);
    let hs = hud(pp - v2(3.0, 3.0), ss).w;
    col = mix(col, v3(0.0), hs * 0.6);
    col = mix(col, h.rgb, h.w);

    col += (h21(pp + fract(NOW)) - 0.5) * 0.015;
    // the display expects linear light
    textureStore(out, vec2<i32>(gid.xy), v4(s2l(clamp(col, v3(0.0), v3(1.0))), 1.0));
}
