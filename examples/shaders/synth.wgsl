// Cuneus GPU Synth — a polyphonic keyboard instrument written entirely in WGSL.
// Enes Altun, 2025-2026; MIT License
// Press keys 1-9 to play notes (C4 .. D5). Everything is computed per-sample at 44.1kHz
// on the GPU: PolyBLEP oscillators, ADSR, a real state-variable filter, soft-clip drive,
// a modulated stereo chorus, a feedback delay line, and a freeverb style reverb.
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
    key_states: array<vec4<f32>, 3>,
    key_decay: array<vec4<f32>, 3>,
};

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

// DSP state buffer layout (all in `dsp`, units = f32 samples)
const H_IC1: u32 = 0u;
const H_IC2: u32 = 1u;
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

fn get_note_frequency(idx: u32, octave: f32) -> f32 {
    let notes = array<f32, 9>(
        261.63, 293.66, 329.63, 349.23, 392.00,
        440.00, 493.88, 523.25, 587.33
    );
    return notes[idx] * pow(2.0, octave - 4.0);
}

// returns vec2(press_time, release_time) for voice i
fn get_key(i: u32) -> vec2<f32> {
    let vi = i / 4u;
    let ci = i % 4u;
    var press_t: f32 = 0.0;
    var release_t: f32 = 0.0;
    if (ci == 0u) { press_t = params.key_states[vi].x; release_t = params.key_decay[vi].x; }
    else if (ci == 1u) { press_t = params.key_states[vi].y; release_t = params.key_decay[vi].y; }
    else if (ci == 2u) { press_t = params.key_states[vi].z; release_t = params.key_decay[vi].z; }
    else { press_t = params.key_states[vi].w; release_t = params.key_decay[vi].w; }
    return vec2<f32>(press_t, release_t);
}

fn adsr_envelope(t: f32, press_time: f32, release_time: f32) -> f32 {
    if (press_time <= 0.0) { return 0.0; }
    let since_press = t - press_time;
    if (since_press < 0.0) { return 0.0; }

    let A = max(params.attack_time, 0.005); // min 5ms to avoid clicks
    let D = params.decay_time;
    let S = params.sustain_level;

    var level: f32;
    if (since_press < A) {
        level = smoothstep(0.0, A, since_press);
    } else if (since_press < A + D) {
        level = 1.0 - (1.0 - S) * (since_press - A) / D;
    } else {
        level = S;
    }

    if (release_time > 0.0) {
        let since_release = t - release_time;
        if (since_release < 0.0) { return level; }
        let R = max(params.release_time, 0.02);
        // figure out where the envelope was when the key was released
        let rsp = release_time - press_time;
        var release_level: f32;
        if (rsp < A) { release_level = rsp / A; }
        else if (rsp < A + D) { release_level = 1.0 - (1.0 - S) * (rsp - A) / D; }
        else { release_level = S; }
        level = release_level * exp(-since_release * 5.0 / R);
        if (level < 0.001) { return 0.0; }
    }

    return level;
}


fn poly_blep(t: f32, dt: f32) -> f32 {
    if (t < dt) { let x = t / dt; return x + x - x * x - 1.0; }
    if (t > 1.0 - dt) { let x = (t - 1.0) / dt; return x * x + x + x + 1.0; }
    return 0.0;
}

// One oscillator sample
fn osc(t: f32, freq: f32, sr: f32, wtype: u32) -> f32 {
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
        case 8u: {                                                           
            return 2.0 * fract(sin(t * freq * 12.9898 + 78.233) * 43758.5453) - 1.0;
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

fn kick(t: f32, tempo: f32) -> f32 {
    let beat_t = fract(t / (60.0 / tempo));
    if (beat_t < 0.1) {
        let env = exp(-beat_t * 30.0);
        let freq = mix(40.0, 120.0, exp(-beat_t * 40.0));
        return sin(TAU * freq * beat_t) * env * 0.3;
    }
    return 0.0;
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
            for (var v = 0u; v < 9u; v++) {
                let k = get_key(v);
                let env = adsr_envelope(t, k.x, k.y);
                if (env > 0.0005) {
                    let freq = get_note_frequency(v, params.octave);
                    mix_s += osc(t, freq, sr, params.waveform_type) * env * 0.5;
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
            if (params.beat_enabled > 0u) { s += kick(t, params.tempo); }

            s = delay_proc(s, gs, params.delay_time, params.delay_feedback, sr);
            s = reverb_proc(s, gs, params.reverb_mix);

            var st = chorus_proc(s, gs, t, params.chorus_rate, params.chorus_depth, sr);
            st *= params.volume;
            st = vec2<f32>(tanh(st.x), tanh(st.y));

            audio_buffer[i * 2u] = st.x;
            audio_buffer[i * 2u + 1u] = st.y;
        }

        dsp[H_IC1] = ic1;
        dsp[H_IC2] = ic2;
    }

    let uv = vec2<f32>(f32(g.x) / f32(dims.x), f32(g.y) / f32(dims.y));
    var color = vec3<f32>(0.02, 0.02, 0.1) * (1.0 - uv.y * 0.3);

    if (uv.y > 0.1 && uv.y < 0.34) {
        let n = max(params.samples_to_generate, 1u);
        let si = u32(uv.x * f32(n - 1u));
        let smp = audio_buffer[si * 2u];
        let cy = 0.22;
        let sy = cy - smp * 0.1;
        if (abs(uv.y - sy) < 0.004) { color = vec3<f32>(0.2, 0.9, 0.6); }
    }

    if (params.beat_enabled > 0u && uv.y > 0.98) {
        let beat_t = fract(u_time.time / (60.0 / params.tempo));
        let pulse = exp(-beat_t * 10.0) * 0.8;
        color = vec3<f32>(pulse, pulse * 0.5, pulse * 0.2);
    }

    let bar_top: f32 = 0.9;
    let bar_max_h: f32 = 0.5;
    let bar_w: f32 = 0.08;
    let bar_sp: f32 = 0.02;
    let total_w = 9.0 * bar_w + 8.0 * bar_sp;
    let start_x = (1.0 - total_w) * 0.5;

    for (var i = 0u; i < 9u; i++) {
        let bx = start_x + f32(i) * (bar_w + bar_sp);
        let k = get_key(i);
        let env = adsr_envelope(u_time.time, k.x, k.y);
        let is_held = k.x > 0.0 && k.y == 0.0;
        let intensity = max(0.1, env);
        let bar_h = bar_max_h * intensity;
        let bar_bot = bar_top - bar_h;

        if (uv.x >= bx && uv.x <= bx + bar_w && uv.y >= bar_bot && uv.y <= bar_top) {
            let hue = f32(i) / 8.0 * TAU;
            let rc = vec3<f32>(
                0.5 + 0.5 * sin(hue),
                0.5 + 0.5 * sin(hue + 2.094),
                0.5 + 0.5 * sin(hue + 4.188)
            );
            let grad = 1.0 - (bar_top - uv.y) / bar_h * 0.3;
            if (is_held) {
                let pulse = sin(u_time.time * 10.0) * 0.1 + 0.9;
                color = rc * intensity * grad * pulse;
            } else {
                color = rc * intensity * grad * 0.5;
            }
        }

        if (uv.x >= bx && uv.x <= bx + bar_w && uv.y >= 0.92 && uv.y <= 0.98) {
            color = vec3<f32>(0.8, 0.8, 0.9);
        }
    }

    if (uv.y < 0.05) {
        let h = f32(params.waveform_type) / 9.0 * TAU;
        color = vec3<f32>(0.5 + 0.5 * sin(h), 0.5 + 0.5 * sin(h + 2.094), 0.5 + 0.5 * sin(h + 4.188));
    }

    textureStore(output, g.xy, vec4<f32>(color, 1.0));
}
