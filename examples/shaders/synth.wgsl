// Cuneus GPU Synth — a polyphonic keyboard instrument written entirely in WGSL.
// Enes Altun, 2025-2026; MIT License
// Play the top letter row like a piano: Q W E R T Y U I O white, 2 3 5 6 7 9 0 black (C4 .. D#5).
// Everything is computed per-sample at 44.1kHz
// on the GPU: PolyBLEP oscillators, a plucked string, FM e-piano and strings, ADSR, a real
// state-variable filter, soft-clip drive, a modulated stereo chorus, a feedback delay line,
// a freeverb style reverb and a drum machine.
//
// The trick that makes the *real* DSP possible in cuneus: effects need state (past samples), so a
// persistent storage buffer (`dsp`) holds the delay/reverb lines and the filter state. The
// sample loop runs sequentially on one thread, carrying recursive state across samples and
// frames. Circular buffers are indexed by the monotonic global sample counter (no write-pos
// bookkeeping); only the recursive filter integrators are saved/restored each frame.

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: SynthParams;
@group(2) @binding(0) var<storage, read_write> audio_buffer: array<f32>;
@group(3) @binding(0) var<storage, read_write> dsp: array<f32>;

struct SynthParams {
    tempo: f32,
    waveform_type: u32,
    octave: f32,
    volume: f32,
    beat_enabled: u32,
    reverb_mix: f32,
    delay_time: f32,
    delay_feedback: f32,
    filter_cutoff: f32,
    filter_resonance: f32,
    distortion_amount: f32,
    chorus_rate: f32,
    chorus_depth: f32,
    attack_time: f32,
    decay_time: f32,
    sustain_level: f32,
    release_time: f32,
    sample_offset: u32,
    samples_to_generate: u32,
    sample_rate: u32,
    drum_level: f32,
    swing: f32,
    // the sample playing now, so the visuals follow the sound
    play_sample: u32,
    _pad1: f32,
    // note-on / note-off sample per key, 0 = none
    key_on: array<vec4<u32>, 4>,
    key_off: array<vec4<u32>, 4>,
};

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

// DSP state buffer layout (all in `dsp`, units = f32 samples)
const H_IC1: u32 = 0u;
const H_IC2: u32 = 1u;
// scope trigger found last frame
const H_TRIG: u32 = 2u;
const HDR: u32 = 4u;
// Circular delay/reverb lines
const DELAY_LEN: u32 = 44100u;
const CHORUS_LEN: u32 = 4096u;
// freeverb comb + allpass 
const C0: u32 = 1116u; const C1: u32 = 1188u; const C2: u32 = 1277u; const C3: u32 = 1356u;
const A0: u32 = 225u;  const A1: u32 = 556u;
const O_DELAY: u32 = HDR;
const O_CHORUS: u32 = O_DELAY + DELAY_LEN;
const O_C0: u32 = O_CHORUS + CHORUS_LEN;
const O_C1: u32 = O_C0 + C0;
const O_C2: u32 = O_C1 + C1;
const O_C3: u32 = O_C2 + C2;
const O_A0: u32 = O_C3 + C3;
const O_A1: u32 = O_A0 + A0;
// scope ring of the final mono mix, for the triggered oscilloscope
const SCOPE_LEN: u32 = 8192u;
const SCOPE_W: u32 = 1024u;
const O_SCOPE: u32 = O_A1 + A1;

// 16 semitones from C of the chosen octave
fn get_note_frequency(idx: u32, octave: f32) -> f32 {
    return 261.63 * exp2(octave - 4.0 + f32(idx) / 12.0);
}

fn get_key(i: u32) -> vec2<u32> {
    return vec2<u32>(params.key_on[i / 4u][i % 4u], params.key_off[i / 4u][i % 4u]);
}

// level `a` seconds after the press while the key is held
fn adsr_held(a: f32) -> f32 {
    // min 5ms to avoid clicks
    let A = max(params.attack_time, 0.005);
    let D = max(params.decay_time, 0.001);
    let S = params.sustain_level;
    if (a < A) { return smoothstep(0.0, A, a); }
    if (a < A + D) { return 1.0 - (1.0 - S) * (a - A) / D; }
    return S;
}

// note times are whole samples, so an envelope never drifts however long the app runs
fn adsr_envelope(n: u32, on: u32, off: u32, sr: f32) -> f32 {
    if (on == 0u || n < on) { return 0.0; }
    if (off < on || n < off) { return adsr_held(f32(n - on) / sr); }
    let R = max(params.release_time, 0.02);
    let level = adsr_held(f32(off - on) / sr) * exp(-f32(n - off) / sr * 5.0 / R);
    if (level < 0.001) { return 0.0; }
    return level;
}

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

fn poly_blep(t: f32, dt: f32) -> f32 {
    if (t < dt) { let x = t / dt; return x + x - x * x - 1.0; }
    if (t > 1.0 - dt) { let x = (t - 1.0) / dt; return x * x + x + x + 1.0; }
    return 0.0;
}

// plucked string: pluck-position comb, highs die first, pick noise
fn pluck(a: f32, f: f32) -> f32 {
    let th = TAU * fract(f * a);
    let c2 = 2.0 * cos(th);
    var s0 = 0.0;
    var s1 = sin(th);
    let q2 = 2.0 * cos(PI * 0.14);
    var q0 = 0.0;
    var q1 = sin(PI * 0.14);
    var v = 0.0;
    let n = min(30, i32(10000.0 / f));
    for (var h = 1; h <= n; h++) {
        let hf = f32(h);
        v += q1 * s1 / hf * exp(-a * (0.9 + 0.12 * hf * hf));
        let s2 = c2 * s1 - s0;
        s0 = s1;
        s1 = s2;
        let q = q2 * q1 - q0;
        q0 = q1;
        q1 = q;
    }
    let pick = wn(u32(a * 44100.0), u32(f)) * 0.2 * exp(-a / 0.0015);
    return (v + pick) * smoothstep(0.0, 0.0006, a) * 1.1;
}

// One oscillator sample; `a` is the note's own age
fn osc(a: f32, freq: f32, sr: f32, wtype: u32, n: u32, voice: u32) -> f32 {
    let t = a;
    let dt = freq / sr;
    let ph = fract(t * freq);
    switch wtype {
        case 0u: { return sin(ph * TAU); }                                   
        case 1u: { return (2.0 * ph - 1.0) - poly_blep(ph, dt); }          
        case 2u: {                                                           
            var s = select(-1.0, 1.0, ph < 0.5);
            s += poly_blep(ph, dt);
            s -= poly_blep(fract(ph + 0.5), dt);
            return s;
        }
        case 3u: {                                                           
            return select(4.0 * ph - 1.0, 3.0 - 4.0 * ph, ph > 0.5);
        }
        case 4u: {                                                          
            let duty = 0.25;
            var s = select(-1.0, 1.0, ph < duty);
            s += poly_blep(ph, dt);
            s -= poly_blep(fract(ph + (1.0 - duty)), dt);
            return s * 0.9;
        }
        case 5u: {                                                           
            var s = 0.0;
            let det = array<f32, 7>(-0.011, -0.007, -0.003, 0.0, 0.004, 0.008, 0.012);
            for (var j = 0u; j < 7u; j++) {
                let f = freq * (1.0 + det[j]);
                let p = fract(t * f + f32(j) * 0.13);
                s += (2.0 * p - 1.0) - poly_blep(p, f / sr);
            }
            return s / 7.0 * 1.3;
        }
        case 6u: {                                                           
            let modu = sin(TAU * t * freq * 2.0);
            return sin(TAU * t * freq + 2.5 * modu);
        }
        case 7u: {                                                           
            var s = sin(ph * TAU);
            s += 0.5 * sin(ph * TAU * 2.0);
            s += 0.33 * sin(ph * TAU * 3.0);
            s += 0.25 * sin(ph * TAU * 4.0);
            s += 0.2 * sin(ph * TAU * 6.0);
            return s / 2.28;
        }
        case 8u: { return wn(n, voice); }
        case 9u: { return pluck(a, freq); }
        case 10u: {
            // FM electric piano: a 1:1 body and a short 14:1 tine
            let w = TAU * fract(freq * a);
            let m1 = 1.6 * exp(-a / 0.6) * sin(w);
            let m2 = 0.9 * exp(-a / 0.03) * sin(TAU * fract(freq * 14.0 * a));
            return (sin(w + m1 + m2) + 0.15 * sin(TAU * fract(freq * 0.5 * a))) * 0.8;
        }
        case 11u: {
            // string ensemble: five detuned saws, vibrato fades in
            let vib = 0.004 * smoothstep(0.15, 0.6, a);
            var s = 0.0;
            for (var j = 0u; j < 5u; j++) {
                let f = freq * (1.0 + (f32(j) - 2.0) * 0.0035);
                let p = fract(f * a + f * vib * (1.0 - cos(TAU * 5.3 * a)) / (TAU * 5.3) + f32(j) * 0.21);
                s += (2.0 * p - 1.0) - poly_blep(p, f / sr);
            }
            return s / 5.0 * 1.2;
        }
        default: { return sin(ph * TAU); }
    }
}

fn distort(s: f32, amount: f32) -> f32 {
    if (amount < 0.01) { return s; }
    let drive = 1.0 + amount * 8.0;
    return mix(s, tanh(s * drive), amount);
}

fn read_frac(off: u32, len: u32, gs: u32, dly: f32) -> f32 {
    let di = u32(dly);
    let fr = dly - f32(di);
    let r0 = off + ((gs + len - di) % len);
    let r1 = off + ((gs + len - di - 1u) % len);
    return mix(dsp[r0], dsp[r1], fr);
}

fn delay_proc(input: f32, gs: u32, dtime: f32, fb: f32, sr: f32) -> f32 {
    let ds = clamp(u32(dtime * sr), 1u, DELAY_LEN - 1u);
    let w = O_DELAY + (gs % DELAY_LEN);
    let r = O_DELAY + ((gs + DELAY_LEN - ds) % DELAY_LEN);
    let delayed = dsp[r];
    dsp[w] = input + delayed * fb;
    return input + delayed * 0.5;
}

fn comb(off: u32, len: u32, gs: u32, input: f32, fb: f32) -> f32 {
    let w = off + (gs % len);
    let d = dsp[w];
    dsp[w] = input + d * fb;
    return d;
}

fn allpass(off: u32, len: u32, gs: u32, input: f32) -> f32 {
    let w = off + (gs % len);
    let bufout = dsp[w];
    dsp[w] = input + bufout * 0.5;
    return -input + bufout;
}

fn reverb_proc(input: f32, gs: u32, wet: f32) -> f32 {
    if (wet < 0.01) { return input; }
    let fb = 0.87;
    var c = comb(O_C0, C0, gs, input, fb);
    c += comb(O_C1, C1, gs, input, fb);
    c += comb(O_C2, C2, gs, input, fb);
    c += comb(O_C3, C3, gs, input, fb);
    var rv = c * 0.25;
    rv = allpass(O_A0, A0, gs, rv);
    rv = allpass(O_A1, A1, gs, rv);
    return mix(input, rv, wet);
}

fn chorus_proc(input: f32, gs: u32, t: f32, rate: f32, depth: f32, sr: f32) -> vec2<f32> {
    dsp[O_CHORUS + (gs % CHORUS_LEN)] = input;
    if (depth < 0.01) { return vec2<f32>(input, input); }
    let base = 0.015 * sr; 
    let dep = depth * 0.008 * sr;
    let lfoL = sin(t * rate * TAU);
    let lfoR = sin(t * rate * TAU + 1.6);
    let wetL = read_frac(O_CHORUS, CHORUS_LEN, gs, base + lfoL * dep);
    let wetR = read_frac(O_CHORUS, CHORUS_LEN, gs, base + lfoR * dep);
    return vec2<f32>(mix(input, wetL, 0.5), mix(input, wetR, 0.5));
}

// drums: integrated pitch sweeps, band-limited noise from the sample index
fn kickDrum(a: f32) -> f32 {
    if (a > 1.2) { return 0.0; }
    let ph = 48.0 * a + (165.0 - 48.0) * 0.032 * (1.0 - exp(-a / 0.032));
    let body = sin(TAU * fract(ph)) * exp(-a / 0.3);
    let knock = sin(TAU * fract(ph * 1.72)) * exp(-a / 0.035) * 0.25;
    return tanh((body + knock) * 1.8) * smoothstep(0.0, 0.0008, a);
}

fn snare(a: f32, n: u32) -> f32 {
    if (a > 0.8) { return 0.0; }
    let ph = 180.0 * a + 70.0 * 0.015 * (1.0 - exp(-a / 0.015));
    let tone = sin(TAU * fract(ph)) * exp(-a / 0.05) + 0.5 * sin(TAU * fract(ph * 1.83)) * exp(-a / 0.03);
    let wires = (vn(n, 2u, 7u) - vn(n, 8u, 7u)) * (exp(-a / 0.16) + 0.8 * exp(-a / 0.012));
    return (tone * 0.5 + wires * 0.9) * smoothstep(0.0, 0.0005, a);
}

fn hatMetal(a: f32) -> f32 {
    let fs = array<f32, 6>(205.3, 304.4, 369.6, 522.7, 540.0, 800.0);
    var m = 0.0;
    for (var i = 0; i < 6; i++) { m += select(-1.0, 1.0, fract(fs[i] * 2.0 * a) < 0.5); }
    return m;
}

fn hat(a: f32, n: u32, decay: f32) -> f32 {
    if (a > decay * 8.0) { return 0.0; }
    let hp = wn(n, 3u) - 0.6 * wn(n - 1u, 3u) - 0.4 * wn(n - 2u, 3u);
    let metal = (hatMetal(a) - hatMetal(a - 1.0 / 44100.0)) / 6.0;
    return (hp * 0.6 + metal * 0.5) * exp(-a / decay) * smoothstep(0.0, 0.0004, a);
}

// one bar of 16ths; `swing` delays the off 16ths, a new kick or hat chokes the ringing one
fn drum_machine(n: u32, sr: f32) -> vec2<f32> {
    let sn = max(u32(sr * 15.0 / params.tempo), 1u);
    let st = n / sn;
    var out = vec2<f32>(0.0);
    var kick_new = -1.0;
    var hat_new = -1.0;
    for (var k = 0u; k < 6u; k++) {
        if (k > st) { break; }
        let s = st - k;
        let p = s % 16u;
        var a = f32(n - s * sn) / sr;
        if (p % 2u == 1u) { a -= params.swing * 0.5 * f32(sn) / sr; }
        if (a < 0.0) { continue; }
        let odd_bar = (s / 16u) % 2u == 1u;
        let r = wn(s, 21u) * 0.5 + 0.5;
        let vel = 0.9 + 0.2 * r;

        var kv = 0.0;
        if (p == 0u || p == 8u) { kv = 1.0; }
        else if (p == 10u && odd_bar) { kv = 0.6; }
        if (kv > 0.0) {
            var choke = 1.0;
            if (kick_new >= 0.0) { choke = exp(-kick_new / 0.01); }
            else {
                kick_new = a;
                out += vec2<f32>((wn(n, 9u) - wn(n - 1u, 9u)) * 0.12 * exp(-a / 0.0012));
            }
            out += vec2<f32>(kickDrum(a) * kv * choke * 0.55);
        }

        var sv = 0.0;
        if (p == 4u || p == 12u) { sv = 1.0; }
        else if ((p == 15u && r < 0.4) || (p == 7u && r < 0.25)) { sv = 0.22; }
        if (sv > 0.0) { out += vec2<f32>(snare(a, n) * sv * vel * 0.3); }

        let open = p == 14u;
        var hv = 0.2 + 0.1 * r;
        if (p % 4u == 2u) { hv = 0.7; } else if (p % 2u == 0u) { hv = 0.45; }
        var choke = 1.0;
        if (hat_new >= 0.0) { choke = exp(-hat_new / 0.008); } else { hat_new = a; }
        out += hat(a, n, select(0.035, 0.25, open)) * hv * vel * choke * vec2<f32>(0.08, 0.11);
    }
    return out;
}

// visuals: every panel reads the song at `play_sample`, the sample being heard

fn hue(h: f32) -> vec3<f32> { return 0.5 + 0.5 * cos(TAU * (h + vec3<f32>(0.0, 0.33, 0.67))); }

// panel coordinates: x right, y up, both 0..1
fn local(p: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> vec2<f32> {
    let q = (p - lo) / (hi - lo);
    return vec2<f32>(q.x, 1.0 - q.y);
}
fn inside(q: vec2<f32>) -> bool { return all(q >= vec2<f32>(0.0)) && all(q <= vec2<f32>(1.0)); }

fn frame(q: vec2<f32>, size: vec2<f32>, accent: vec3<f32>) -> vec3<f32> {
    let e = min(min(q.x, 1.0 - q.x) * size.x, min(q.y, 1.0 - q.y) * size.y);
    return vec3<f32>(0.012, 0.014, 0.03) + accent * 0.35 * exp(-max(e, 0.0) * 0.7);
}

// a glowing line `d` pixels away
fn glow_line(d: f32, c: vec3<f32>) -> vec3<f32> { return c * (smoothstep(1.6, 0.4, d) + 0.25 * exp(-d * 0.15)); }

// distance in pixels from `q` to the curve y = y0 with slope dy (both in panel units per pixel column)
fn curve_dist(q: vec2<f32>, y0: f32, dy: f32, size: vec2<f32>) -> f32 {
    return abs(q.y - y0) * size.y / sqrt(1.0 + dy * dy * size.y * size.y);
}

fn scope_at(i: u32) -> f32 { return dsp[O_SCOPE + i % SCOPE_LEN]; }

// triggered oscilloscope of the final mix
fn scope_view(q: vec2<f32>, size: vec2<f32>, accent: vec3<f32>) -> vec3<f32> {
    let trig = u32(dsp[H_TRIG]);
    let fx = q.x * f32(SCOPE_W - 2u);
    let i0 = u32(fx);
    let a = scope_at(trig + i0);
    let b = scope_at(trig + i0 + 1u);
    let gain = 0.42;
    let y0 = 0.5 + mix(a, b, fract(fx)) * gain;
    let dy = (b - a) * gain * f32(SCOPE_W) / size.x;
    var col = vec3<f32>(0.05, 0.06, 0.1) * (smoothstep(1.0, 0.0, abs(q.y - 0.5) * size.y) + smoothstep(1.0, 0.0, abs(fract(q.x * 8.0 + 0.5) - 0.5) * size.x / 8.0) * 0.5);
    col += glow_line(curve_dist(q, y0, dy, size), accent * 1.3);
    return col;
}

fn adsr_shape(t: f32, hold: f32) -> f32 {
    let A = max(params.attack_time, 0.005);
    let D = max(params.decay_time, 0.001);
    if (t < A + D + hold) { return adsr_held(t); }
    return params.sustain_level * exp(-(t - A - D - hold) * 5.0 / max(params.release_time, 0.02));
}

// envelope curve with a dot riding it for every sounding key
fn env_view(q: vec2<f32>, size: vec2<f32>, accent: vec3<f32>, sr: f32) -> vec3<f32> {
    let A = max(params.attack_time, 0.005);
    let D = max(params.decay_time, 0.001);
    let R = max(params.release_time, 0.02);
    let hold = 0.25 * (A + D + R) + 0.1;
    let total = A + D + hold + R;
    let t = q.x * total;
    let lv = adsr_shape(t, hold);
    let y0 = 0.08 + lv * 0.84;
    let dy = (adsr_shape(t + total / size.x, hold) - lv) * 0.84;
    var col = accent * 0.06 * step(q.y, y0);
    for (var m = 0; m < 3; m++) {
        let mt = select(select(A + D + hold, A + D, m == 1), A, m == 0);
        col += vec3<f32>(0.05, 0.06, 0.1) * smoothstep(1.0, 0.0, abs(q.x - mt / total) * size.x);
    }
    col += glow_line(curve_dist(q, y0, dy, size), accent);

    let ps = params.play_sample;
    for (var v = 0u; v < 16u; v++) {
        let k = get_key(v);
        let env = adsr_envelope(ps, k.x, k.y, sr);
        if (env <= 0.0) { continue; }
        var tt = min(f32(ps - k.x) / sr, A + D + hold * 0.5);
        if (k.y >= k.x && ps >= k.y) { tt = min(A + D + hold + f32(ps - k.y) / sr, total); }
        let d = length((q - vec2<f32>(tt / total, 0.08 + env * 0.84)) * size);
        col += hue(f32(v) / 16.0) * (smoothstep(6.0, 3.5, d) * 1.5 + exp(-d * 0.2) * 0.3);
    }
    return col;
}

// low-pass response in dB, the same curve the audio filter uses
fn filter_db(f: f32) -> f32 {
    let fc = 20.0 * pow(1000.0, params.filter_cutoff);
    let w = f / fc;
    let kf = 2.0 - 1.9 * clamp(params.filter_resonance, 0.0, 0.98);
    let d = 1.0 - w * w;
    return -3.0103 * log2(d * d + kf * kf * w * w);
}

fn partial(h: f32, wave: u32) -> f32 {
    let odd = fract(h * 0.5) > 0.25;
    if (wave == 0u) { return select(0.0, 1.0, h < 1.5); }
    if (wave == 2u) { return select(0.0, 1.0 / h, odd); }
    if (wave == 3u) { return select(0.0, 1.0 / (h * h), odd); }
    if (wave == 8u) { return 0.0; }
    return 1.0 / h;
}

// filter curve on a log axis, 20 Hz .. 20 kHz, with the harmonics of every sounding key under it
fn filter_view(q: vec2<f32>, size: vec2<f32>, accent: vec3<f32>, sr: f32) -> vec3<f32> {
    let f = 20.0 * pow(1000.0, q.x);
    let db = filter_db(f);
    let y0 = clamp((db + 36.0) / 54.0, 0.0, 1.0);
    let dy = (clamp((filter_db(20.0 * pow(1000.0, q.x + 1.0 / size.x)) + 36.0) / 54.0, 0.0, 1.0) - y0);
    var col = accent * 0.06 * step(q.y, y0);
    col += vec3<f32>(0.05, 0.06, 0.1) * smoothstep(1.0, 0.0, abs(q.y - 36.0 / 54.0) * size.y);
    col += accent * 0.3 * smoothstep(1.0, 0.0, abs(q.x - params.filter_cutoff) * size.x);

    let ps = params.play_sample;
    for (var v = 0u; v < 16u; v++) {
        let k = get_key(v);
        let env = adsr_envelope(ps, k.x, k.y, sr);
        if (env <= 0.0) { continue; }
        let f0 = get_note_frequency(v, params.octave);
        let h = max(round(f / f0), 1.0);
        let xh = log2(f0 * h / 20.0) / log2(1000.0);
        if (h <= 32.0 && abs(q.x - xh) * size.x < 1.2) {
            let amp = partial(h, params.waveform_type) * pow(10.0, filter_db(f0 * h) / 20.0) * env;
            if (q.y < clamp(amp, 0.0, 1.0) * 0.8) { col += hue(f32(v) / 16.0); }
        }
    }
    col += glow_line(curve_dist(q, y0, dy, size), accent);
    return col;
}

// 16-step drum row: kick, snare and hat lanes, the current step lit
fn drum_view(q: vec2<f32>, size: vec2<f32>, accent: vec3<f32>, sr: f32) -> vec3<f32> {
    let sn = max(u32(sr * 15.0 / params.tempo), 1u);
    let ps = params.play_sample;
    let st = ps / sn;
    let cur = st % 16u;
    let age = f32(ps - st * sn) / sr;
    // the extra kick on step 11 plays every other bar
    let odd_bar = (st / 16u) % 2u == 1u;
    let on = select(0.35, 1.0, params.beat_enabled > 0u);
    let c = min(u32(q.x * 16.0), 15u);
    let fx = fract(q.x * 16.0);
    let cell = size.x / 16.0;
    var col = vec3<f32>(0.05, 0.06, 0.1) * smoothstep(1.0, 0.0, min(fx, 1.0 - fx) * cell) * select(0.4, 1.0, c % 4u == 0u);
    // the playhead only runs while the drums play
    let playing = params.beat_enabled > 0u && c == cur;
    if (playing) { col += accent * 0.22; }
    let flash = select(0.0, 1.4 * exp(-age * 8.0), playing);
    let kick = c == 0u || c == 8u || (c == 10u && odd_bar);
    let snare = c == 4u || c == 12u;
    let lanes = array<f32, 3>(0.22, 0.5, 0.78);
    let hits = array<bool, 3>(kick, snare, true);
    let tints = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.35, 0.2), vec3<f32>(1.0, 0.85, 0.3), vec3<f32>(0.3, 0.9, 1.0));
    for (var l = 0; l < 3; l++) {
        if (!hits[l]) { continue; }
        var r = size.y * 0.1;
        if (l == 2 && c == 14u) { r *= 1.6; }
        let d = length((vec2<f32>(fx, q.y) - vec2<f32>(0.5, lanes[l])) * vec2<f32>(cell, size.y));
        col += tints[l] * smoothstep(r, r * 0.6, d) * (0.35 + flash) * on;
    }
    return col;
}

// piano strip: 10 white keys C..E (the last one is out of range) and the black keys between them
fn keys_view(q: vec2<f32>, size: vec2<f32>, sr: f32) -> vec3<f32> {
    let whites = array<u32, 10>(0u, 2u, 4u, 5u, 7u, 9u, 11u, 12u, 14u, 16u);
    let blacks = array<i32, 11>(-1, 1, 3, -1, 6, 8, 10, -1, 13, 15, -1);
    let x = q.x * 10.0;
    let b = round(x);
    var key = whites[min(u32(x), 9u)];
    var black = false;
    if (b >= 1.0 && b <= 9.0 && abs(x - b) < 0.3 && q.y > 0.38) {
        let s = blacks[i32(b)];
        if (s >= 0) {
            key = u32(s);
            black = true;
        }
    }
    var env = 0.0;
    if (key < 16u) {
        let k = get_key(key);
        env = adsr_envelope(params.play_sample, k.x, k.y, sr);
    }
    let kc = hue(f32(key) / 16.0);
    if (black) {
        let e = min((0.3 - abs(x - b)) * size.x / 10.0, (q.y - 0.38) * size.y);
        return mix(vec3<f32>(0.0), vec3<f32>(0.06) + kc * env * 1.4, smoothstep(0.5, 1.5, e));
    }
    let fx = fract(x);
    let e = min(fx, 1.0 - fx) * size.x / 10.0;
    var col = mix(vec3<f32>(0.02), vec3<f32>(0.7, 0.7, 0.76) * (0.8 + 0.2 * (1.0 - q.y)), smoothstep(0.5, 1.5, e));
    if (key >= 16u) { col *= 0.35; }
    return mix(col, kc * 1.2, env * 0.8) + kc * env * 0.3;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let dims = textureDimensions(output);
    if (g.x >= dims.x || g.y >= dims.y) { return; }

    if (g.x == 0u && g.y == 0u) {
        let sr = f32(params.sample_rate);
        let n = params.samples_to_generate;

        let fc = clamp(20.0 * pow(1000.0, params.filter_cutoff), 20.0, sr * 0.45);
        let gco = tan(PI * fc / sr);
        let kf = 2.0 - 1.9 * clamp(params.filter_resonance, 0.0, 0.98);
        let a1 = 1.0 / (1.0 + gco * (gco + kf));
        let a2 = gco * a1;
        let a3 = gco * a2;

        var ic1 = dsp[H_IC1];
        var ic2 = dsp[H_IC2];

        for (var i = 0u; i < n; i++) {
            let gs = params.sample_offset + i;
            let t = f32(gs) / sr;

            var mix_s: f32 = 0.0;
            var nact: f32 = 0.0;
            for (var v = 0u; v < 16u; v++) {
                let k = get_key(v);
                let env = adsr_envelope(gs, k.x, k.y, sr);
                if (env > 0.0005) {
                    let freq = get_note_frequency(v, params.octave);
                    mix_s += osc(f32(gs - k.x) / sr, freq, sr, params.waveform_type, gs, v) * env * 0.5;
                    nact += 1.0;
                }
            }
            if (nact > 1.0) { mix_s /= sqrt(nact); }

            let v3 = mix_s - ic2;
            let bp = a1 * ic1 + a2 * v3;
            let lp = ic2 + a2 * ic1 + a3 * v3;
            ic1 = 2.0 * bp - ic1;
            ic2 = 2.0 * lp - ic2;
            var s = lp;

            s = distort(s, params.distortion_amount);

            s = delay_proc(s, gs, params.delay_time, params.delay_feedback, sr);
            s = reverb_proc(s, gs, params.reverb_mix);

            var st = chorus_proc(s, gs, t, params.chorus_rate, params.chorus_depth, sr);
            // drums stay dry, after the effects
            if (params.beat_enabled > 0u) { st += drum_machine(gs, sr) * params.drum_level; }
            st *= params.volume;
            st = vec2<f32>(tanh(st.x), tanh(st.y));

            audio_buffer[i * 2u] = st.x;
            audio_buffer[i * 2u + 1u] = st.y;
            dsp[O_SCOPE + gs % SCOPE_LEN] = (st.x + st.y) * 0.5;
        }

        dsp[H_IC1] = ic1;
        dsp[H_IC2] = ic2;

        // a rising zero crossing just before the heard sample, so the scope stands still
        let ps = params.play_sample;
        if (ps > SCOPE_W + 1024u) {
            var trig = ps - SCOPE_W;
            for (var j = 0u; j < 700u; j++) {
                let k = ps - SCOPE_W - j;
                if (dsp[O_SCOPE + (k - 1u) % SCOPE_LEN] < 0.0 && dsp[O_SCOPE + k % SCOPE_LEN] >= 0.0) {
                    trig = k;
                    break;
                }
            }
            dsp[H_TRIG] = f32(trig % SCOPE_LEN);
        }
    }

    let R = vec2<f32>(dims);
    let p = (vec2<f32>(g.xy) + 0.5) / R;
    let sr = f32(params.sample_rate);
    // accent colour follows the wave
    let accent = hue(f32(params.waveform_type) / 12.0 + 0.55) * 1.1;
    var color = vec3<f32>(0.008, 0.01, 0.025) * (1.2 - p.y * 0.5);

    // scope, envelope, filter, drum steps, keyboard: (left, top, right, bottom)
    let panels = array<vec4<f32>, 5>(
        vec4<f32>(0.03, 0.04, 0.97, 0.38),
        vec4<f32>(0.03, 0.42, 0.485, 0.70),
        vec4<f32>(0.515, 0.42, 0.97, 0.70),
        vec4<f32>(0.03, 0.73, 0.97, 0.80),
        vec4<f32>(0.03, 0.83, 0.97, 0.97),
    );
    for (var i = 0; i < 5; i++) {
        let r = panels[i];
        let q = local(p, r.xy, r.zw);
        if (!inside(q)) { continue; }
        let size = (r.zw - r.xy) * R;
        color = frame(q, size, accent);
        switch i {
            case 0: { color += scope_view(q, size, accent); }
            case 1: { color += env_view(q, size, accent, sr); }
            case 2: { color += filter_view(q, size, accent, sr); }
            case 3: { color += drum_view(q, size, accent, sr); }
            default: { color = keys_view(q, size, sr); }
        }
    }

    color = 1.0 - exp(-color * 1.4);
    textureStore(output, g.xy, vec4<f32>(color, 1.0));
}
