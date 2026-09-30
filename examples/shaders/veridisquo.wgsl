// Veridis Quo - Daft Punk tribute, coded entirely in WGSL
// Enes Altun, 2025-2026; MIT License
// My attempt at recreating that Discovery-era sound with math :-)
// Soft saw lead, Moog-ish bass pluck, warm saw pads, strummed guitar, drums,
// ping-pong tape echo, a small room, sidechain compression — the whole thing.
// On screen: an LED pyramid stage and a dot-matrix banner, lit by the same notes you hear.
// Originally prototyped on Shadertoy, ported to cuneus PcmStreamManager.
// Modal drums, plucked strings and additive filters follow the techniques in Tonny Espeset's
// Shadertoy music (https://www.shadertoy.com/user/Espeset).
// Still a WIP — the mix could always be better, but that's music for you.

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

struct SongParams {
    volume: f32,
    tempo_multiplier: f32,
    sample_offset: u32,
    samples_to_generate: u32,
    sample_rate: f32,
    mix_drums: f32,
    mix_bass: f32,
    mix_lead: f32,
    mix_guitar: f32,
    mix_pads: f32,
    mix_echo: f32,
    mix_space: f32,
    // tone: types are 0/1/2 switches
    lead_type: f32,
    lead_tone: f32,
    lead_detune: f32,
    lead_glide: f32,
    bass_type: f32,
    bass_tone: f32,
    bass_drive: f32,
    pad_type: f32,
    pad_tone: f32,
    pad_width: f32,
    guitar_tone: f32,
    guitar_mute: f32,
    kick_tune: f32,
    kick_decay: f32,
    hat_decay: f32,
    drum_pattern: f32,
    echo_time: f32,
    echo_feedback: f32,
    swing: f32,
    // the sample playing now, so the visuals follow the sound
    play_sample: u32,
    // the sample where the song (re)started
    song_origin: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(1) @binding(1) var<uniform> u_song: SongParams;

struct FontUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    _padding: vec2<f32>,
};
@group(2) @binding(0) var<uniform> u_font: FontUniforms;
@group(2) @binding(1) var t_font_atlas: texture_2d<f32>;
@group(2) @binding(2) var<storage, read_write> audio_buffer: array<f32>;

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

// Song timing: 8 measures of 16 sixteenths
const BPM: f32 = 107.0;
const SIX: f32 = 15.0 / BPM;
const LOOP: f32 = 128.0 * 15.0 / BPM;

fn measure_duration() -> f32 {
    return (60.0 / BPM) * 4.0;
}

fn mtof(m: f32) -> f32 { return 440.0 * exp2((m - 69.0) / 12.0); }
fn wrap_t(t: f32) -> f32 { return t - floor(t / LOOP) * LOOP; }
// age of a note that started on `step`, across the loop seam
fn age_at(t: f32, step: i32) -> f32 { return wrap_t(t - f32(step) * SIX); }
// a rate with a whole number of cycles per loop, so modulation never jumps at the seam
fn loop_hz(hz: f32) -> f32 { return round(hz * LOOP) / LOOP; }
fn release(age: f32, gate: f32, r: f32) -> f32 { return select(1.0, exp(-(age - gate) / r), age > gate); }
fn lp12(f: f32, fc: f32, res: f32) -> f32 {
    let w = f / fc;
    let q = 0.5 + res * 2.0;
    let d = 1.0 - w * w;
    return inverseSqrt(d * d + w * w / (q * q));
}

// noise indexed by the global sample, so it is the same whatever the frame split
fn hash_u(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u; v *= 0x7feb352du;
    v ^= v >> 15u; v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}
fn wn(n: u32, seed: u32) -> f32 { return f32(hash_u(n * 0x9e3779b9u ^ seed) >> 8u) / 8388608.0 - 1.0; }
fn vn(n: u32, m: u32, seed: u32) -> f32 {
    let i = n / m;
    var f = f32(n % m) / f32(m);
    f = f * f * (3.0 - 2.0 * f);
    return mix(wn(i, seed), wn(i + 1u, seed), f);
}
fn hash_i(i: i32, seed: u32) -> f32 { return wn(u32(i), seed) * 0.5 + 0.5; }

// melody: start step, midi note, length in sixteenths
const MEL_S = array<i32, 32>(0, 1, 2, 3, 6, 7, 8, 9, 32, 33, 34, 35, 38, 39, 40, 41, 64, 65, 66, 67, 70, 71, 72, 73, 96, 97, 98, 99, 100, 101, 102, 103);
const MEL_N = array<f32, 32>(77., 76., 77., 74., 77., 76., 77., 71., 76., 74., 76., 72., 76., 74., 76., 69., 77., 76., 77., 74., 77., 76., 77., 71., 76., 74., 76., 72., 76., 74., 76., 69.);
const MEL_L = array<i32, 32>(1, 1, 1, 3, 1, 1, 1, 23, 1, 1, 1, 3, 1, 1, 1, 23, 1, 1, 1, 3, 1, 1, 1, 23, 1, 1, 1, 1, 1, 1, 1, 25);

fn mel_cur(step: i32) -> i32 {
    var c = 0;
    for (var i = 0; i < 32; i++) {
        if (MEL_S[i] <= step) { c = i; }
    }
    return c;
}

// driving bass (the bounce): midi, accent
fn bass_note(step: i32) -> vec2<f32> {
    let s = step % 16;
    let lm = (step / 16) % 4;
    let hi = s == 0 || s == 4 || s == 7 || s == 10 || s == 13;
    var h = 62.0;
    var l = 50.0;
    if (lm == 1) { h = 67.0; l = 55.0; }
    else if (lm == 2) {
        if (s < 4) { h = 67.0; l = 57.0; } else { h = 69.0; l = 57.0; }
    } else if (lm == 3) {
        if (s < 4) { h = 69.0; l = 53.0; }
        else if (s < 7) { h = 65.0; l = 53.0; }
        else if (s < 10) { h = 65.0; l = 52.0; }
        else if (s < 13) { h = 64.0; l = 52.0; }
        else { h = 64.0; l = 50.0; }
    }
    return vec2<f32>(select(l, h, hi), select(0.0, 1.0, hi));
}

// pad chords: root midi, minor, start step, length
fn chord_seg(step: i32) -> vec4<f32> {
    let ms = f32((step / 16) * 16);
    let lm = (step / 16) % 4;
    if (lm == 0) { return vec4<f32>(62.0, 1.0, ms, 16.0); }
    if (lm == 1) { return vec4<f32>(67.0, 0.0, ms, 16.0); }
    if (lm == 2) { return vec4<f32>(69.0, 1.0, ms, 16.0); }
    if (step % 16 < 8) { return vec4<f32>(65.0, 0.0, ms, 8.0); }
    return vec4<f32>(64.0, 1.0, ms + 8.0, 8.0);
}

// instruments

// 1. Lead: soft saw (detuned saws through a dark 12 dB low-pass over a sine body), drawbar organ
// or FM keys. `bright` darkens the echoes and the room
fn leadOrgan(freq: f32, age: f32, bright: f32) -> vec2<f32> {
    let hm = array<f32, 7>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0);
    let ha = array<f32, 7>(1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1);
    let rot = TAU * fract(5.6 * age);
    let dop = 0.00006 * sin(rot);
    let lp = 5000.0 * bright * u_song.lead_tone;
    var y = vec2<f32>(0.0);
    for (var h = 0; h < 7; h++) {
        let hz = freq * hm[h];
        var amp = ha[h] / (1.0 + (hz / lp) * (hz / lp));
        if (h == 2) { amp += 0.45 * exp(-age / 0.22); }
        y += vec2<f32>(sin(TAU * fract(hz * (age + dop))), sin(TAU * fract(hz * (age - dop)))) * amp;
    }
    return y * vec2<f32>(1.0 + 0.12 * sin(rot), 1.0 - 0.12 * sin(rot)) / 3.5;
}

fn leadKeys(freq: f32, age: f32, bright: f32) -> vec2<f32> {
    let w = TAU * fract(freq * age);
    let m = 1.3 * u_song.lead_tone * bright * exp(-age / 0.5) * sin(w) + 0.5 * exp(-age / 0.03) * sin(TAU * fract(freq * 14.0 * age));
    let trem = 0.2 * sin(TAU * 4.5 * age);
    return vec2<f32>(1.0 + trem, 1.0 - trem) * sin(w + m) * exp(-age * 1.2) * 0.9;
}

fn leadSynth(freq: f32, prev: f32, age: f32, gate: f32, bright: f32) -> vec2<f32> {
    let env = smoothstep(0.0, 0.015, age) * exp(-age * 0.25) * release(age, gate, 0.12);
    if (env < 0.0001) { return vec2<f32>(0.0); }
    let ty = u32(u_song.lead_type + 0.5);
    if (ty == 1u) { return leadOrgan(freq, age, bright) * env; }
    if (ty == 2u) { return leadKeys(freq, age, bright) * env; }
    let fc = freq * (1.5 + 2.0 * u_song.lead_tone * exp(-age / 0.15)) * bright;
    let top = min(fc * 3.0, 9000.0);
    let g = u_song.lead_glide;
    var y = vec2<f32>(0.0);
    for (var v = 0; v < 5; v++) {
        let r = exp2((f32(v) - 2.0) * 0.5 * u_song.lead_detune / 1200.0);
        let f = freq * r;
        var ph = f * age + f32(v) * 0.137;
        if (g > 0.001) { ph += (prev - freq) * r * g * (1.0 - exp(-age / g)); }
        let th = TAU * fract(ph);
        let c2 = 2.0 * cos(th);
        var s0 = 0.0;
        var s1 = sin(th);
        var sw = 0.0;
        let n = min(20, i32(top / f));
        for (var h = 1; h <= n; h++) {
            sw += s1 / f32(h) * lp12(f * f32(h), fc, 0.0);
            let s2 = c2 * s1 - s0;
            s0 = s1;
            s1 = s2;
        }
        let pan = f32(v) / 4.0;
        y += sw * vec2<f32>(1.0 - pan * 0.4, 0.6 + pan * 0.4);
    }
    let body = sin(TAU * fract(freq * age));
    return (y * 0.22 + vec2<f32>(body * 0.55)) * env;
}

// 2. Bass: Moog pluck (sine body plus a detuned saw through a closing 12 dB low-pass),
// round sub, or FM with a falling index
fn bassPluck(freq: f32, age: f32, gate: f32, hi: bool, bright: f32) -> f32 {
    let env = smoothstep(0.0, 0.003, age) * exp(-age * select(10.0, 5.0, hi)) * release(age, gate, 0.012);
    if (env < 0.0001) { return 0.0; }
    let ty = u32(u_song.bass_type + 0.5);
    let tone = u_song.bass_tone;
    let w = TAU * fract(freq * age);
    let sine = sin(w);
    let sub = sin(TAU * fract(freq * 0.5 * age));
    var x = 0.0;
    if (ty == 1u) {
        x = sine * 0.8 + sub * 0.6 + 0.15 * tone * sin(TAU * fract(freq * 2.0 * age));
    } else if (ty == 2u) {
        x = sin(w + 2.2 * tone * bright * exp(-age / 0.08) * sine) * 0.9 + sub * 0.4;
    } else {
        let fc = freq * (1.2 + select(7.0, 10.0, hi) * tone * exp(-age / 0.06)) * bright;
        var saw = 0.0;
        for (var o = 0; o < 2; o++) {
            let f = freq * select(1.0, 1.006, o == 1);
            let th = TAU * fract(f * age + f32(o) * 0.31);
            let c2 = 2.0 * cos(th);
            var s0 = 0.0;
            var s1 = sin(th);
            for (var h = 1; h <= 20; h++) {
                let hz = f * f32(h);
                if (hz > 12000.0) { break; }
                saw += s1 / f32(h) * lp12(hz, fc, 0.35);
                let s2 = c2 * s1 - s0;
                s0 = s1;
                s1 = s2;
            }
        }
        x = sine * 0.7 + sub * 0.4 + saw * 0.3;
    }
    return tanh(x * u_song.bass_drive) * env;
}

// 3. Lead + bass voices at a time: the current note and the tail of the previous one
fn leadAt(t: f32, bright: f32) -> vec2<f32> {
    let tt = wrap_t(t);
    let cur = mel_cur(i32(tt / SIX));
    var y = vec2<f32>(0.0);
    for (var k = 0; k < 2; k++) {
        let i = (cur - k + 32) % 32;
        let age = age_at(tt, MEL_S[i]);
        let gate = f32(MEL_L[i]) * SIX;
        if (age < gate + 0.6) { y += leadSynth(mtof(MEL_N[i]), mtof(MEL_N[(i + 31) % 32]), age, gate, bright); }
    }
    return y;
}

fn bassAt(t: f32, bright: f32) -> f32 {
    let tt = wrap_t(t);
    let st = i32(tt / SIX);
    var y = 0.0;
    for (var k = 0; k < 2; k++) {
        let s = (st - k + 128) % 128;
        let bn = bass_note(s);
        y += bassPluck(mtof(bn.x), age_at(tt, s), SIX * 0.9, bn.y > 0.5, bright);
    }
    return y;
}

// 4. Triad helper (midi)
fn getTriad(root: f32, minor: bool) -> vec3<f32> {
    return vec3<f32>(root, root + select(4.0, 3.0, minor), root + 7.0);
}

// 5. Strummed guitar: a string plucked near the bridge, highs choked first, pick noise
fn guitarString(f: f32, age: f32, gate: f32) -> f32 {
    if (age <= 0.0 || age > gate + 0.2) { return 0.0; }
    let th = TAU * fract(f * age);
    let c2 = 2.0 * cos(th);
    var s0 = 0.0;
    var s1 = sin(th);
    // pluck position comb: sin(pi*h*0.16)
    let q2 = 2.0 * cos(PI * 0.16);
    var q0 = 0.0;
    var q1 = sin(PI * 0.16);
    var v = 0.0;
    let n = min(28, i32(9000.0 / f));
    for (var h = 1; h <= n; h++) {
        let hf = f32(h);
        let pres = 1.0 + 0.8 * exp(-pow(log2(hf * f / 2600.0), 2.0) * 2.0);
        v += q1 * s1 / hf * exp(-age * (1.8 + 0.35 / max(u_song.guitar_tone, 0.2) * hf * hf)) * pres;
        let s2 = c2 * s1 - s0;
        s0 = s1;
        s1 = s2;
        let q = q2 * q1 - q0;
        q0 = q1;
        q1 = q;
    }
    let pick = wn(u32(age * 44100.0), u32(f * 7.0)) * 0.15 * exp(-age / 0.0015);
    return (v + pick) * smoothstep(0.0, 0.0006, age) * release(age, gate, 0.025);
}

// offbeat up-strums and muted ghost down-strums, per measure
const STRUM_S = array<i32, 6>(2, 3, 6, 10, 11, 14);
const STRUM_V = array<f32, 6>(1.0, 0.35, 0.9, 1.0, 0.35, 0.85);

fn guitarAt(t: f32) -> f32 {
    let tt = wrap_t(t);
    let st = i32(tt / SIX);
    var idx = -1;
    for (var i = 0; i < 6; i++) {
        if (STRUM_S[i] <= st % 16) { idx = i; }
    }
    let e = (st / 16) * 6 + idx;
    var y = 0.0;
    for (var k = 0; k < 2; k++) {
        let ek = (e - k + 48) % 48;
        let step = (ek / 6) * 16 + STRUM_S[ek % 6];
        let ghost = STRUM_V[ek % 6] < 0.5;
        let vel = STRUM_V[ek % 6] * (0.85 + 0.3 * hash_i(ek, 11u));
        let age = age_at(tt, step);
        let gate = select(SIX * 1.6, SIX * 0.45, ghost) * u_song.guitar_mute;
        let cs = chord_seg(step);
        let tri = getTriad(cs.x, cs.y > 0.5);
        let notes = array<f32, 5>(cs.x - 12.0, cs.x - 5.0, tri.x, tri.y, tri.z);
        for (var j = 0; j < 5; j++) {
            if (ghost && j < 2) { continue; }
            let order = select(f32(j), f32(4 - j), !ghost);
            let a = age - order * 0.008;
            y += guitarString(mtof(notes[j]), a, gate) * vel;
        }
    }
    return y;
}

// 6. Pads: warm saws (three detuned per chord note, filter swelling open and breathing),
// drawbar organ, or strings (slower bow and a vibrato that fades in)
fn padOrgan(cs: vec4<f32>, age: f32) -> vec2<f32> {
    let tri = getTriad(cs.x, cs.y > 0.5);
    let roots = array<f32, 4>(mtof(cs.x - 12.0), mtof(tri.x), mtof(tri.y), mtof(tri.z));
    let hm = array<f32, 7>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0);
    let ha = array<f32, 7>(1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1);
    let det = exp2(u_song.pad_width * 0.3 / 1200.0);
    let lp = 2500.0 * u_song.pad_tone;
    var y = vec2<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        var v = vec2<f32>(0.0);
        for (var h = 0; h < 7; h++) {
            let f = roots[i] * hm[h];
            let amp = ha[h] / (f32(i) + 1.0) / (1.0 + (f / lp) * (f / lp));
            v += vec2<f32>(sin(TAU * fract(f * det * age)), sin(TAU * fract(f / det * age))) * amp;
        }
        let am = 0.12 * sin(TAU * 0.8 * age + f32(i) * 1.7);
        y += v * vec2<f32>(1.0 + am, 1.0 - am);
    }
    return y / 5.0;
}

fn padSynth(cs: vec4<f32>, age: f32, gate: f32, lfo: f32) -> vec2<f32> {
    let ty = u32(u_song.pad_type + 0.5);
    let strings = ty == 2u;
    let env = (1.0 - exp(-age / select(0.15, 0.45, strings))) * release(age, gate, 0.5);
    if (ty == 1u) { return padOrgan(cs, age) * env; }
    let tri = getTriad(cs.x, cs.y > 0.5);
    let notes = array<f32, 4>(cs.x - 12.0, tri.x, tri.y, tri.z);
    let fc = select(1100.0, 1700.0, strings) * u_song.pad_tone * (1.0 + 0.35 * lfo) * (0.5 + 0.5 * (1.0 - exp(-age / 0.6)));
    let vib = select(0.0, 0.003 * smoothstep(0.2, 0.8, age) * (1.0 - cos(TAU * 5.2 * age)) / (TAU * 5.2), strings);
    var y = vec2<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        for (var v = 0; v < 3; v++) {
            let f = mtof(notes[i]) * exp2(f32(v - 1) * u_song.pad_width / 1200.0);
            let th = TAU * fract(f * age + f * vib * f32(v + 1) / 2.0 + f32(i * 3 + v) * 0.23);
            let c2 = 2.0 * cos(th);
            var s0 = 0.0;
            var s1 = sin(th);
            var sw = 0.0;
            let n = min(24, i32(fc * 2.5 / f));
            for (var h = 1; h <= n; h++) {
                sw += s1 / f32(h) * lp12(f * f32(h), fc, 0.15);
                let s2 = c2 * s1 - s0;
                s0 = s1;
                s1 = s2;
            }
            let pan = f32(v) * 0.5;
            y += sw * vec2<f32>(1.0 - pan * 0.5, 0.5 + pan * 0.5);
        }
    }
    return y * env / 6.0;
}

fn padAt(t: f32, lfo: f32) -> vec2<f32> {
    let tt = wrap_t(t);
    let cs = chord_seg(i32(tt / SIX));
    let ps = chord_seg((i32(cs.z) - 1 + 128) % 128);
    return padSynth(cs, age_at(tt, i32(cs.z)), cs.w * SIX, lfo) + padSynth(ps, age_at(tt, i32(ps.z)), ps.w * SIX, lfo);
}

// 7. Kick drum: integrated pitch sweep, a knock overtone, soft drive
fn kickDrum(a: f32) -> f32 {
    if (a < 0.0 || a > u_song.kick_decay * 5.0) { return 0.0; }
    let kt = u_song.kick_tune;
    let ph = (48.0 * a + (165.0 - 48.0) * 0.032 * (1.0 - exp(-a / 0.032))) * kt;
    let body = sin(TAU * fract(ph)) * exp(-a / u_song.kick_decay);
    let knock = sin(TAU * fract(ph * 1.72)) * exp(-a / 0.035) * 0.25;
    return tanh((body + knock) * 1.8) * smoothstep(0.0, 0.0008, a) / 0.95;
}

// 8. Clap: three quick band-noise bursts, a room tail and a little snare tone
fn clap(a: f32, n: u32) -> f32 {
    if (a < 0.0 || a > 0.8) { return 0.0; }
    let bn = vn(n, 2u, 7u) - vn(n, 9u, 7u);
    var e = 0.0;
    for (var i = 0; i < 3; i++) {
        let x = a - f32(i) * 0.0105;
        if (x >= 0.0) { e += exp(-x / 0.0035) * select(0.75, 1.0, i == 2); }
    }
    e += 0.6 * exp(-max(a - 0.021, 0.0) / 0.13) * step(0.021, a);
    let tone = sin(TAU * fract(185.0 * a + 60.0 * 0.02 * (1.0 - exp(-a / 0.02)))) * exp(-a / 0.05) * 0.35;
    return bn * e * 0.8 + tone;
}

// 9. Hi-hat: high-passed noise plus six detuned square "cymbal" tones
fn hatMetal(a: f32) -> f32 {
    let fs = array<f32, 6>(205.3, 304.4, 369.6, 522.7, 540.0, 800.0);
    var m = 0.0;
    for (var i = 0; i < 6; i++) { m += select(-1.0, 1.0, fract(fs[i] * 2.0 * a) < 0.5); }
    return m;
}

fn hat(a: f32, n: u32, decay: f32) -> f32 {
    if (a < 0.0 || a > decay * 8.0) { return 0.0; }
    let hp = wn(n, 3u) - 0.6 * wn(n - 1u, 3u) - 0.4 * wn(n - 2u, 3u);
    let metal = (hatMetal(a) - hatMetal(a - 1.0 / 44100.0)) / 6.0;
    return (hp * 0.6 + metal * 0.5) * exp(-a / decay) * smoothstep(0.0, 0.0004, a);
}

fn hatHit(step: i32) -> vec2<f32> {
    let s = step % 16;
    if (s == 14 && (step / 16) % 2 == 1) { return vec2<f32>(0.75, 0.22); }
    if (s % 4 == 2) { return vec2<f32>(0.8, 0.04); }
    if (s % 2 == 0) { return vec2<f32>(0.45, 0.03); }
    if (hash_i(step, 5u) < 0.45) { return vec2<f32>(0.18, 0.025); }
    return vec2<f32>(0.0);
}

// song time for a global sample: kept inside one loop so every phase stays precise
fn song_time(n: u32) -> f32 {
    let tm = max(u_song.tempo_multiplier, 0.1);
    let sr = u_song.sample_rate;
    let loop_n = u32(LOOP * sr / tm);
    let k = select(0u, n - u_song.song_origin, n >= u_song.song_origin);
    return f32(k % loop_n) / sr * tm;
}

fn mainSound(n: u32) -> vec2<f32> {
    let t = song_time(n);
    let lfo = sin(TAU * fract(t * loop_hz(0.1)));
    let wow = 1.0 + 0.003 * sin(TAU * fract(t * loop_hz(0.45)));
    let ec = u_song.mix_echo;
    let ed = u_song.echo_time * wow;
    let fb = u_song.echo_feedback;

    // lead + bass with a ping-pong tape echo, each repeat darker
    let lead = leadAt(t, 1.0) * 0.40
        + leadAt(t - ed, 0.5) * vec2<f32>(0.15, 0.05) * ec
        + leadAt(t - 2.0 * ed, 0.32) * vec2<f32>(0.03, 0.08) * ec * fb
        + leadAt(t - 3.0 * ed, 0.22) * vec2<f32>(0.04, 0.012) * ec * fb * fb;
    let bass = vec2<f32>(bassAt(t, 1.0) * 0.35)
        + bassAt(t - ed, 0.6) * vec2<f32>(0.10, 0.03) * ec
        + bassAt(t - 2.0 * ed, 0.4) * vec2<f32>(0.015, 0.05) * ec * fb;

    let pads = padAt(t, lfo) * 0.14;
    let guitar = guitarAt(t) * vec2<f32>(0.06, 0.08) + guitarAt(t - 3.0 * SIX * wow) * vec2<f32>(0.03, 0.012) * ec;

    // small room: early reflections of the lead and guitar, darker and alternating sides
    let er = array<f32, 6>(0.029, 0.047, 0.071, 0.098, 0.131, 0.173);
    var room = vec2<f32>(0.0);
    for (var i = 0; i < 6; i++) {
        let gn = 0.5 * pow(0.72, f32(i));
        let side = select(vec2<f32>(1.0, 0.45), vec2<f32>(0.45, 1.0), i % 2 == 1);
        room += (leadAt(t - er[i], 0.25) * 0.4 * u_song.mix_lead + vec2<f32>(guitarAt(t - er[i]) * 0.07 * u_song.mix_guitar)) * side * gn;
    }
    room *= u_song.mix_space;

    // drums: a new hit chokes the one still ringing
    let tt = wrap_t(t);
    let st = i32(tt / SIX);
    let b = st / 4;
    let ka = age_at(tt, b * 4);
    let kick = kickDrum(ka) + kickDrum(age_at(tt, ((b + 31) % 32) * 4)) * exp(-ka / 0.01)
        + (wn(n, 9u) - wn(n - 1u, 9u)) * 0.2 * exp(-ka / 0.0012);
    var cs = (st / 8) * 8 + 4;
    if (st % 8 < 4) { cs = (cs - 8 + 128) % 128; }
    let clp = clap(age_at(tt, cs), n) * (0.9 + 0.2 * hash_i(cs, 13u));
    // swing pushes the off 16ths late
    let sw = u_song.swing * 0.5 * SIX;
    let h0 = hatHit(st);
    let ps = (st + 127) % 128;
    let h1 = hatHit(ps);
    let ha = age_at(tt, st) - select(0.0, sw, st % 2 == 1);
    var hats = hat(ha, n, h0.y * u_song.hat_decay) * h0.x;
    hats += hat(age_at(tt, ps) - select(0.0, sw, ps % 2 == 1), n, h1.y * u_song.hat_decay) * h1.x * select(1.0, exp(-max(ha, 0.0) / 0.008), h0.x > 0.0 && ha > 0.0);
    // pattern: 0 full kit, 1 no clap, 2 kick only
    let pat = u32(u_song.drum_pattern + 0.5);
    let drums = vec2<f32>(kick * 0.6 + clp * 0.26 * select(1.0, 0.0, pat >= 1u)) + hats * vec2<f32>(0.09, 0.12) * select(1.0, 0.0, pat == 2u);

    // sidechain duck
    let md = measure_duration();
    let m = i32(tt / md);
    let pm = fract(tt / md);
    var sc_depth: f32 = 0.35;
    if (m == 3 || m == 7) {
        sc_depth = mix(0.35, 0.05, pm);
    }
    let sidechain = 1.0 - (exp(-ka * 10.0) * sc_depth);

    let music = (lead * u_song.mix_lead + bass * u_song.mix_bass + pads * u_song.mix_pads + guitar * u_song.mix_guitar + room) * sidechain
        + drums * u_song.mix_drums;

    // soft clip
    return tanh(music) * 0.5;
}


// visuals: an LED pyramid stage, a dot-matrix banner and a crowd, lit by the same song data as the audio

fn sd_box(q: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(q) - b;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}
fn sd_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}
fn h2(a: i32, b: i32) -> f32 { return f32(hash_u(u32(a) * 73856093u ^ u32(b) * 19349663u) >> 8u) / 16777216.0; }

// Dm red, Em violet, F amber, G cyan, Am magenta
fn chord_col(root: f32) -> vec3<f32> {
    if (root < 63.0) { return vec3<f32>(1.0, 0.12, 0.1); }
    if (root < 64.5) { return vec3<f32>(0.55, 0.3, 1.0); }
    if (root < 66.0) { return vec3<f32>(1.0, 0.55, 0.1); }
    if (root < 68.0) { return vec3<f32>(0.1, 0.8, 1.0); }
    return vec3<f32>(1.0, 0.2, 0.7);
}

struct Vis {
    col: vec3<f32>,
    kp: f32,
    mel_row: f32,
    mel_t: f32,
    mel_env: f32,
    prev_row: f32,
    prev_env: f32,
    bas_env: f32,
    bas_hi: f32,
    bas_side: f32,
    hat_env: f32,
    clap_t: f32,
    strum_t: f32,
    strum_v: f32,
    st: i32,
    time: f32,
};

fn mel_row(m: f32) -> f32 { return 5.0 + round((m - 69.0) * 7.0 / 8.0); }

// every light reads the song at T, the time of the sample being heard
fn vis_state(T: f32, time: f32) -> Vis {
    var v: Vis;
    let st = i32(T / SIX);
    v.st = st;
    v.time = time;
    let pat = u32(u_song.drum_pattern + 0.5);
    let dm = min(u_song.mix_drums, 1.0);
    v.kp = exp(-age_at(T, (st / 4) * 4) * 9.0) * dm;

    let mi = mel_cur(st);
    let ma = age_at(T, MEL_S[mi]);
    v.mel_row = mel_row(MEL_N[mi]);
    v.mel_t = ma;
    v.mel_env = smoothstep(0.0, 0.02, ma) * (0.35 + 0.65 * exp(-ma * 3.0)) * min(u_song.mix_lead, 1.0);
    let pi = (mi + 31) % 32;
    let pa = age_at(T, MEL_S[pi]) - f32(MEL_L[pi]) * SIX;
    v.prev_row = mel_row(MEL_N[pi]);
    v.prev_env = 0.6 * exp(-max(pa, 0.0) / 0.3) * min(u_song.mix_lead, 1.0);

    let bn = bass_note(st);
    v.bas_hi = bn.y;
    v.bas_env = exp(-(T - f32(st) * SIX) * select(10.0, 5.0, bn.y > 0.5)) * min(u_song.mix_bass, 1.0);
    v.bas_side = select(-1.0, 1.0, st % 2 == 0);

    let hh = hatHit(st);
    v.hat_env = hh.x * exp(-age_at(T, st) / max(hh.y * u_song.hat_decay, 0.02)) * dm * select(1.0, 0.0, pat == 2u);
    var cs = (st / 8) * 8 + 4;
    if (st % 8 < 4) { cs = (cs - 8 + 128) % 128; }
    v.clap_t = select(age_at(T, cs), 99.0, pat >= 1u || dm <= 0.0);

    var idx = -1;
    for (var i = 0; i < 6; i++) {
        if (STRUM_S[i] <= st % 16) { idx = i; }
    }
    let e = ((st / 16) * 6 + idx + 48) % 48;
    v.strum_t = age_at(T, (e / 6) * 16 + STRUM_S[e % 6]);
    v.strum_v = STRUM_V[e % 6] * min(u_song.mix_guitar, 1.0);

    let ch = chord_seg(st);
    let pc = chord_seg((i32(ch.z) - 1 + 128) % 128);
    v.col = mix(chord_col(pc.x), chord_col(ch.x), smoothstep(0.0, 0.35, age_at(T, i32(ch.z))));
    return v;
}

const PA: f32 = 0.40;
const PB: f32 = -0.42;
const PW: f32 = 0.95;
const ROWS: f32 = 14.0;

fn pyramid(p: vec2<f32>, v: Vis) -> vec3<f32> {
    let h = PA - PB;
    var col = vec3<f32>(0.0);

    // neon outline, flaring on the kick
    let apex = vec2<f32>(0.0, PA);
    let de = min(min(sd_seg(p, apex, vec2<f32>(-PW, PB)), sd_seg(p, apex, vec2<f32>(PW, PB))), sd_seg(p, vec2<f32>(-PW, PB), vec2<f32>(PW, PB)));
    col += v.col * (exp(-de * 160.0) * (0.6 + 1.8 * v.kp) + exp(-de * 22.0) * 0.07 * (1.0 + 2.0 * v.kp));

    let hw = PW * (PA - p.y) / h;
    if (p.y < PB || p.y > PA || abs(p.x) > hw) { return col; }

    // DJ booth opening
    if (abs(p.x) < 0.17 && p.y < PB + 0.15) {
        let desk = exp(-abs(p.y - (PB + 0.05)) * 300.0) * step(abs(p.x), 0.15);
        var b = v.col * desk * (0.6 + v.kp);
        for (var i = 0; i < 2; i++) {
            let hc = vec2<f32>(select(-0.06, 0.06, i == 1), PB + 0.085);
            let dh = length(p - hc) - 0.022;
            b = mix(b, vec3<f32>(0.0), smoothstep(0.002, -0.002, dh));
            b += vec3<f32>(1.0, 0.9, 0.8) * exp(-abs(sd_box(p - hc - vec2<f32>(0.0, 0.004), vec2<f32>(0.016, 0.001))) * 900.0) * (0.3 + v.mel_env);
        }
        return col + b;
    }

    // LED panels
    let rh = h / ROWS;
    let r = floor((p.y - PB) / rh);
    let cy = PB + (r + 0.5) * rh;
    let cw = rh * 1.3;
    let c = floor(p.x / cw + 0.5);
    let cx = c * cw;
    let hwr = PW * (PA - (PB + (r + 1.0) * rh)) / h;
    if (abs(cx) + cw * 0.45 > hwr + cw * 0.25) { return col; }
    let d = sd_box(p - vec2<f32>(cx, cy), vec2<f32>(cw * 0.36, rh * 0.28)) - rh * 0.06;
    let rn = r / (ROWS - 1.0);
    let xn = cx / max(hwr, 0.02);

    var I = 0.05 + 0.25 * v.kp;
    var tint = v.col;
    // melody: a bar at the note's height, growing out from the centre
    if (r == v.mel_row && abs(xn) < min(1.0, v.mel_t * 5.0 + 0.1)) {
        I += 1.4 * v.mel_env;
        tint = mix(v.col, vec3<f32>(1.0, 0.95, 0.9), 0.55);
    }
    if (r == v.prev_row) { I += 0.5 * v.prev_env; }
    // bass: bottom rows, accents full width, the rest alternating sides
    if (r < 3.0 && (v.bas_hi > 0.5 || xn * v.bas_side > 0.0)) {
        I += 0.9 * v.bas_env * (1.0 - r / 3.0);
        tint = mix(tint, vec3<f32>(1.0, 0.45, 0.1), 0.5);
    }
    // clap: a band rising up the face
    if (abs(r - v.clap_t * 45.0) < 0.6) { I += exp(-v.clap_t * 3.0); }
    // guitar strum: a chevron travelling up
    let ph = rn + abs(xn) * 0.6 - v.strum_t * 4.0;
    if (ph > -0.1 && ph < 0.0) { I += 0.8 * v.strum_v * exp(-v.strum_t * 4.0); }
    // hats: sparkles
    if (h2(i32(c) * 31 + i32(r), v.st) < 0.1) {
        I += 1.2 * v.hat_env;
        tint = mix(tint, vec3<f32>(1.0), 0.6);
    }

    let led = smoothstep(0.002, -0.001, d) + exp(-max(d, 0.0) * 90.0) * 0.35;
    return col + tint * I * led + vec3<f32>(0.015) * smoothstep(0.002, -0.001, d);
}

fn beams(p: vec2<f32>, v: Vis) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    let bc = mix(v.col, vec3<f32>(1.0), 0.4);
    for (var i = 0; i < 4; i++) {
        let sd = select(-1.0, 1.0, i % 2 == 1);
        let o = vec2<f32>(sd * (0.5 + 0.55 * f32(i / 2)), 1.1);
        let ang = -PI * 0.5 - sd * (0.3 + 0.22 * sin(v.time * 0.35 + f32(i) * 1.9));
        let dir = vec2<f32>(cos(ang), sin(ang));
        let rel = p - o;
        let along = dot(rel, dir);
        if (along > 0.0) {
            let perp = abs(rel.x * dir.y - rel.y * dir.x);
            let w = along * 0.08 + 0.008;
            col += bc * exp(-(perp / w) * (perp / w)) * exp(-along * 0.6) * (0.1 + 0.25 * v.kp);
        }
    }
    return col;
}

// crowd silhouettes: heads, shoulders and a few raised arms bobbing on the kick
fn crowd(p: vec2<f32>, kp: f32) -> f32 {
    let cw = 0.075;
    let i = i32(floor(p.x / cw));
    var sd = p.y + 0.86;
    for (var k = -1; k <= 1; k++) {
        let j = i + k;
        let hh = h2(j, 7);
        let base = -0.8 + 0.05 * hh + 0.015 * kp * (0.5 + hh);
        let cx = (f32(j) + 0.5) * cw + (hh - 0.5) * 0.02;
        sd = min(sd, length(p - vec2<f32>(cx, base + 0.06)) - (0.026 + 0.008 * hh));
        let q = (p - vec2<f32>(cx, base - 0.03)) / vec2<f32>(0.045, 0.06);
        sd = min(sd, (length(q) - 1.0) * 0.045);
        sd = min(sd, p.y - (base - 0.05));
        if (h2(j, 11) > 0.72) {
            let s = select(-1.0, 1.0, h2(j, 13) > 0.5);
            let sh = vec2<f32>(cx + s * 0.03, base + 0.01);
            sd = min(sd, sd_seg(p, sh, sh + vec2<f32>(s * 0.035, 0.14 + 0.03 * kp)) - 0.011);
        }
    }
    return sd;
}

// scrolling dot-matrix "cuneus" from the font atlas
const MSG = array<u32, 10>(99u, 117u, 110u, 101u, 117u, 115u, 32u, 32u, 32u, 32u);

fn glyph(k: i32, lx: f32, ly: f32, sz: f32) -> f32 {
    if (lx < 0.0 || lx >= sz || ly < 0.0 || ly >= sz) { return 0.0; }
    let code = MSG[((k % 10) + 10) % 10];
    let puv = vec2<f32>(lx, ly) / sz * 0.9 + vec2<f32>(0.05);
    let uv = (vec2<f32>(f32(code % 16u), f32(code / 16u)) + puv) / 16.0;
    return textureLoad(t_font_atlas, vec2<i32>(uv * u_font.atlas_size), 0).r;
}

fn banner(p: vec2<f32>, v: Vis) -> vec3<f32> {
    let top = 0.93;
    let bot = 0.66;
    var col = v.col * 0.35 * exp(-abs(sd_box(p - vec2<f32>(0.0, (top + bot) * 0.5), vec2<f32>(1.6, (top - bot) * 0.5 + 0.02))) * 250.0);
    let pitch = (top - bot) / 13.0;
    let gy = floor((top - p.y) / pitch);
    if (gy < 0.0 || gy > 12.0) { return col; }
    let gx = floor(p.x / pitch);
    let cc = vec2<f32>((gx + 0.5) * pitch, top - (gy + 0.5) * pitch);
    let sz = top - bot;
    let adv = sz * 0.5;
    let tx = cc.x + v.time * 0.25;
    let ty = top - cc.y;
    let k = i32(floor(tx / adv));
    let cov = max(glyph(k, tx - f32(k) * adv, ty, sz), glyph(k - 1, tx - f32(k - 1) * adv, ty, sz));
    let on = smoothstep(0.25, 0.6, cov);
    let dd = length(p - cc) / pitch;
    let dot_ = smoothstep(0.38, 0.28, dd) + exp(-dd * 5.0) * 0.25 * on;
    return col + mix(vec3<f32>(0.05, 0.01, 0.01), vec3<f32>(1.0, 0.16, 0.06) * (1.6 + 0.6 * v.kp), on) * dot_;
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) g: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // audio: one thread per sample, packed into whole workgroups
    let ai = (wid.y * nwg.x + wid.x) * 256u + li;
    if (ai < u_song.samples_to_generate) {
        let stereo = mainSound(u_song.sample_offset + ai);
        audio_buffer[ai * 2u] = stereo.x * u_song.volume;
        audio_buffer[ai * 2u + 1u] = stereo.y * u_song.volume;
    }

    let d = textureDimensions(output);
    if (g.x >= d.x || g.y >= d.y) { return; }
    let res = vec2<f32>(d);
    let p = vec2<f32>(f32(g.x) - res.x * 0.5, res.y * 0.5 - f32(g.y)) / res.y * 2.0;

    // song time of the sample being heard, the same clock the audio runs on
    let T = song_time(u_song.play_sample);
    let v = vis_state(T, u_time.time);

    // haze and beams
    var col = vec3<f32>(0.012, 0.008, 0.02) + v.col * 0.035 * (1.0 - abs(p.y)) * (1.0 + v.kp);
    col += beams(p, v);
    col += pyramid(p, v);
    // wet floor reflection
    if (p.y < PB) {
        let rp = vec2<f32>(p.x + 0.004 * sin(p.y * 120.0 + v.time * 2.0), 2.0 * PB - p.y);
        col += pyramid(rp, v) * 0.22 * exp(-(PB - p.y) * 5.0);
    }
    col += banner(p, v);
    // crowd in front, rim-lit by the stage
    let cd = crowd(p, v.kp);
    col = col * smoothstep(-0.002, 0.003, cd) + v.col * exp(-max(cd, 0.0) * 180.0) * step(0.0, cd) * 0.25 * (1.0 + v.kp) * step(p.y, -0.5);

    // finish: tone map, scanlines, vignette, grain, sidechain breathing
    col = 1.0 - exp(-col * 1.3);
    col *= 0.93 + 0.07 * cos(f32(g.y) * PI);
    let vp = f32(g.x) / res.x - 0.5;
    col *= 1.0 - 0.35 * (vp * vp * 2.0 + (p.y * 0.5) * (p.y * 0.5));
    col += (h2(i32(g.x), i32(g.y) + i32(u_time.frame) * 7919) - 0.5) * 0.02;
    col *= 0.9 + 0.1 * (1.0 - v.kp);

    textureStore(output, g.xy, vec4<f32>(max(col, vec3<f32>(0.0)), 1.0));
}
