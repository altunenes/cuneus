// Cuneus GPU Synth — a simple keyboard instrument written entirely in WGSL
// Enes Altun, 2025-2026; MIT License
// Press keys 1-9 to play notes (C4 through D5). Everything runs on the GPU:
// waveform generation, ADSR envelopes, lowpass filter, distortion, chorus,
// delay, reverb — all computed per-sample at 44.1kHz.
// It's not a Moog, but it's fun to play with :-)

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: SynthParams;
@group(2) @binding(0) var<storage, read_write> audio_buffer: array<f32>;

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

fn generate_waveform(phase: f32, waveform_type: u32) -> f32 {
    switch waveform_type {
        case 0u: { return sin(phase); }
        case 1u: { return 2.0 * fract(phase / TAU) - 1.0; }
        case 2u: { return select(-1.0, 1.0, sin(phase) > 0.0); }
        case 3u: {
            let t = fract(phase / TAU);
            return select(4.0 * t - 1.0, 3.0 - 4.0 * t, t > 0.5);
        }
        case 4u: {
            return 2.0 * fract(sin(phase * 12.9898) * 43758.5453) - 1.0;
        }
        default: { return sin(phase); }
    }
}

fn lowpass(s: f32, cutoff: f32, resonance: f32, t: f32) -> f32 {
    if (cutoff > 0.95) { return s; }
    let freq = cutoff * cutoff * 0.8;
    return s * (0.3 + freq * 0.7) + s * sin(t * 50.0) * resonance * 0.1;
}

fn distort(s: f32, amount: f32) -> f32 {
    if (amount < 0.01) { return s; }
    let drive = 1.0 + amount * 5.0;
    let driven = s * drive;
    return mix(s, driven / (1.0 + abs(driven)), amount);
}

fn chorus(s: f32, t: f32, rate: f32, depth: f32) -> f32 {
    if (depth < 0.01) { return s; }
    let lfo1 = sin(t * rate) * depth;
    let lfo2 = sin(t * rate * 1.3 + 1.57) * depth;
    return (s + s * (1.0 + lfo1 * 0.5) * 0.4 + s * (1.0 + lfo2 * 0.3) * 0.3) / 1.7;
}

fn reverb(s: f32, mix_amt: f32, t: f32) -> f32 {
    if (mix_amt < 0.01) { return s; }
    let r = s * 0.7
        + s * sin(t * 100.0) * 0.15 * mix_amt
        + s * sin(t * 150.0 + 0.08) * 0.1 * mix_amt
        + s * sin(t * 80.0 + 0.15) * 0.08 * mix_amt;
    return mix(s, r, mix_amt);
}

fn delay_fx(s: f32, t: f32, del_time: f32, feedback: f32) -> f32 {
    if (feedback < 0.01) { return s; }
    let dt = t - del_time;
    return s + s * sin(dt * 10.0) * feedback * 0.6 + s * sin(dt * 15.0) * feedback * 0.12;
}

fn kick(t: f32, tempo: f32) -> f32 {
    let beat_t = fract(t / (60.0 / tempo));
    if (beat_t < 0.1) {
        let env = exp(-beat_t * 30.0);
        let freq = mix(40.0, 120.0, exp(-beat_t * 40.0));
        return sin(TAU * freq * beat_t) * env * 0.25;
    }
    return 0.0;
}

fn synthSample(t: f32) -> vec2<f32> {
    var sum: f32 = 0.0;
    var num_active: f32 = 0.0;

    for (var i = 0u; i < 9u; i++) {
        let k = get_key(i);
        let press_time = k.x;
        let release_time = k.y;

        let env = adsr_envelope(t, press_time, release_time);
        if (env > 0.0005) {
            let freq = get_note_frequency(i, params.octave);
            let detune = (f32(i) - 4.0) * 0.002;
            let adj_freq = freq * (1.0 + detune);
            let phase_offset = f32(i) * 0.61803;
            let phase = (t * adj_freq + phase_offset) * TAU;

            var s = generate_waveform(phase, params.waveform_type);
            s = lowpass(s, params.filter_cutoff, params.filter_resonance, t);
            s = distort(s, params.distortion_amount);
            s = chorus(s, t + f32(i) * 0.1, params.chorus_rate, params.chorus_depth);

            sum += s * env * 0.6;
            num_active += 1.0;
        }
    }

    if (num_active > 1.0) { sum /= sqrt(num_active); }

    if (params.beat_enabled > 0u) {
        sum += kick(t, params.tempo);
    }

    sum = delay_fx(sum, t, params.delay_time, params.delay_feedback);
    sum = reverb(sum, params.reverb_mix, t);

    sum *= params.volume;
    let clipped = sum / (1.0 + abs(sum));

    let stereo_offset = sin(t * params.chorus_rate * 0.7) * params.chorus_depth * 0.1;
    return vec2<f32>(clipped - stereo_offset, clipped + stereo_offset);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let dims = textureDimensions(output);
    if (g.x >= dims.x || g.y >= dims.y) { return; }

    if (g.x == 0u && g.y == 0u) {
        let sr = f32(params.sample_rate);
        let n = params.samples_to_generate;
        for (var i = 0u; i < n; i++) {
            let global_sample = params.sample_offset + i;
            let t = f32(global_sample) / sr;
            let stereo = synthSample(t);
            audio_buffer[i * 2u] = stereo.x;
            audio_buffer[i * 2u + 1u] = stereo.y;
        }
    }

    let uv = vec2<f32>(f32(g.x) / f32(dims.x), f32(g.y) / f32(dims.y));

    var color = vec3<f32>(0.02, 0.02, 0.1) * (1.0 - uv.y * 0.3);

    if (params.beat_enabled > 0u && uv.y > 0.98) {
        let beat_t = fract(u_time.time / (60.0 / params.tempo));
        let pulse = exp(-beat_t * 10.0) * 0.8;
        color = vec3<f32>(pulse, pulse * 0.5, pulse * 0.2);
    }

    let bar_top: f32 = 0.9;
    let bar_max_h: f32 = 0.6;
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
        var wc = vec3<f32>(0.5);
        switch params.waveform_type {
            case 0u: { wc = vec3<f32>(0.3, 0.8, 0.3); }
            case 1u: { wc = vec3<f32>(0.8, 0.8, 0.3); }
            case 2u: { wc = vec3<f32>(0.8, 0.3, 0.3); }
            case 3u: { wc = vec3<f32>(0.3, 0.3, 0.8); }
            case 4u: { wc = vec3<f32>(0.8, 0.3, 0.8); }
            default: {}
        }
        color = wc;
    }

    textureStore(output, g.xy, vec4<f32>(color, 1.0));
}
