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
fn atime() -> f32 { return f32(gm.so) / gm.sr; }

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
// open for enemies: empty, no bomb, no live flame (so freed enemies don't walk into a lingering blast)
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

// half chase the player (greedy on the longer axis), half wander; turn at junctions, no backtrack
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

    // enemies spread away from the player corner; clear their cell + crate neighbors for room
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
fn nz(t: f32) -> f32 { return fract(sin(t * 101.17) * 43758.5453) * 2.0 - 1.0; }
fn nf(m: f32) -> f32 { return 440.0 * pow(2.0, (m - 69.0) / 12.0); }
fn sq(ph: f32) -> f32 { return select(-1.0, 1.0, fract(ph) < 0.5); }
fn mv(t: f32, fr: f32) -> f32 { return sq(t * fr) * (1.0 - exp(-t * 500.0)) * exp(-t * 7.0); }

fn snd(ta: f32) -> f32 {
    var s = 0.0;

    // BOMBERMAN BGM 1 (Atsushi Chikuma, 1987) - bass groove, 4 bars / 64 sixteenths @ Q=129
    if (u32(g[0]) == 1u && g[26] > 0.0) {
        let mt = ta - g[26];
        if (mt >= 0.0) {
            let s16 = 60.0 / 129.0 / 4.0;
            let pos = (mt / s16) % 64.0;
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
            var acc = 0.0; var cur = 0.0; var nt = 0.0;
            for (var i = 0u; i < 45u; i++) {
                if (pos < acc + dur[i]) { cur = note[i]; nt = (pos - acc) * s16; break; }
                acc += dur[i];
            }
            if (cur > 0.5) { s += mv(nt, nf(cur)) * 0.10; }
        }
    }

    // sfx: place / boom / die / win / step / kill
    let dp = ta - g[20]; if (g[20] > 0.0 && dp >= 0.0 && dp < 0.12) { s += sin(dp * tau * 880.0) * exp(-dp * 45.0) * 0.2; }
    let db = ta - g[21]; if (g[21] > 0.0 && db >= 0.0 && db < 0.5) { let e = exp(-db * 7.0); s += (nz(ta) * 0.6 + sin(db * tau * (90.0 - db * 120.0)) * 0.8) * e * 0.5; }
    let dd = ta - g[22];
    if (g[22] > 0.0 && dd >= 0.0 && dd < 1.2) { var gn = array<f32, 3>(392.0, 311.13, 233.08); for (var j = 0u; j < 3u; j++) { let n = dd - f32(j) * 0.16; if (n >= 0.0) { s += sin(n * tau * gn[j]) * exp(-n * 5.0) * 0.22; } } }
    let dw = ta - g[23];
    if (g[23] > 0.0 && dw >= 0.0 && dw < 1.3) { var wn = array<f32, 4>(523.25, 659.25, 783.99, 1046.5); for (var j = 0u; j < 4u; j++) { let n = dw - f32(j) * 0.12; if (n >= 0.0) { s += sin(n * tau * wn[j]) * exp(-n * 5.0) * 0.2; } } }
    let dk = ta - g[24]; if (g[24] > 0.0 && dk >= 0.0 && dk < 0.05) { s += sin(dk * tau * 300.0) * exp(-dk * 60.0) * 0.08; }
    let dx = ta - g[25]; if (g[25] > 0.0 && dx >= 0.0 && dx < 0.25) { s += sin(dx * tau * (440.0 - dx * 500.0)) * exp(-dx * 16.0) * 0.22; }
    return clamp(s, -1.0, 1.0);
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
fn main(@builtin(global_invocation_id) gid: u3) {
    let ss = v2(textureDimensions(out));
    let pp = v2(gid.xy);
    if (any(pp >= ss)) { return; }

    // thread 0: logic + this frame's pcm
    if (all(gid.xy == vec2(0u))) {
        init(); upd();
        for (var i = 0u; i < gm.sn; i++) {
            let v = snd(f32(gm.so + i) / gm.sr) * gm.vol;
            au[i * 2u] = v; au[i * 2u + 1u] = v;
        }
    }

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
