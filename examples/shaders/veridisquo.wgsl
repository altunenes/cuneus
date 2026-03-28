// Veridis Quo - Daft Punk tribute, coded entirely in WGSL
// Enes Altun, 2025-2026; MIT License
// My attempt at recreating that Discovery-era sound with math :-)
// Drawbar organ lead, Moog-ish bass pluck, chord pads, kick drums,
// phase-decoupled delay lines, sidechain compression — the whole thing.
// Originally prototyped on Shadertoy, ported to cuneus PcmStreamManager.
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
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
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

// Note frequencies
const F5: f32 = 698.46;
const E5: f32 = 659.25;
const D5: f32 = 587.33;
const C5: f32 = 523.25;
const B4: f32 = 493.88;
const A4: f32 = 440.0;
const G4: f32 = 392.00;
const F4: f32 = 349.23;
const E4: f32 = 329.63;
const D4: f32 = 293.66;
const C4: f32 = 261.63;
const A3: f32 = 220.0;
const G3: f32 = 196.0;
const F3: f32 = 174.615;
const E3: f32 = 164.815;
const D3: f32 = 146.83;
const C3: f32 = 130.815;
const A2: f32 = 110.0;
const B2: f32 = 123.47;

// Song timing
const BPM: f32 = 107.0;

fn measure_duration() -> f32 {
    return (60.0 / BPM) * 4.0;
}

// Song data structure passed around
struct SongData {
    mel: f32,
    mel_t: f32,
    drv_bas: f32,
    drv_bas_t: f32,
    bas_hi: bool,
    pad_root: f32,
    pad_minor: bool,
};

fn getSongData(time: f32) -> SongData {
    let md = measure_duration();
    let td = md * 8.0;
    let lt = time % td;
    let m = i32(lt / md);
    let pm = fract(lt / md);

    var mel = A4;
    var mel_t = pm;
    let pd: f32 = 0.25;
    let sh: f32 = 0.125;
    let p2s = pd + sh;
    let p2e = p2s + pd;

    // melody sequencer
    if (m == 0 || m == 4) {
        if (pm < pd) {
            let p = pm / pd;
            let idx = i32(floor(p * 4.0));
            mel_t = fract(p * 4.0) * (pd / 4.0);
            if (idx == 0) { mel = F5; } else if (idx == 1) { mel = E5; } else if (idx == 2) { mel = F5; } else { mel = D5; }
        } else if (pm < p2s) {
            mel_t = (pm - pd) + (pd / 4.0); mel = D5;
        } else if (pm < p2e) {
            let p = (pm - p2s) / pd;
            let idx = i32(floor(p * 4.0));
            mel_t = fract(p * 4.0) * (pd / 4.0);
            if (idx == 0) { mel = F5; } else if (idx == 1) { mel = E5; } else if (idx == 2) { mel = F5; } else { mel = B4; }
        } else {
            mel_t = (pm - p2e) + (pd / 4.0); mel = B4;
        }
    } else if (m == 1 || m == 5) {
        mel = B4; mel_t = pm + 0.4375;
    } else if (m == 2) {
        if (pm < pd) {
            let p = pm / pd;
            let idx = i32(floor(p * 4.0));
            mel_t = fract(p * 4.0) * (pd / 4.0);
            if (idx == 0) { mel = E5; } else if (idx == 1) { mel = D5; } else if (idx == 2) { mel = E5; } else { mel = C5; }
        } else if (pm < p2s) {
            mel_t = (pm - pd) + (pd / 4.0); mel = C5;
        } else if (pm < p2e) {
            let p = (pm - p2s) / pd;
            let idx = i32(floor(p * 4.0));
            mel_t = fract(p * 4.0) * (pd / 4.0);
            if (idx == 0) { mel = E5; } else if (idx == 1) { mel = D5; } else if (idx == 2) { mel = E5; } else { mel = A4; }
        } else {
            mel_t = (pm - p2e) + (pd / 4.0); mel = A4;
        }
    } else if (m == 3 || m == 7) {
        mel = A4;
        if (m == 7) { mel_t = pm + 0.5625; } else { mel_t = pm + 0.4375; }
    } else if (m == 6) {
        let rd: f32 = 0.5;
        if (pm < rd) {
            let p = pm / rd;
            let idx = i32(floor(p * 8.0));
            mel_t = fract(p * 8.0) * (rd / 8.0);
            if (idx == 0) { mel = E5; } else if (idx == 1) { mel = D5; } else if (idx == 2) { mel = E5; } else if (idx == 3) { mel = C5; }
            else if (idx == 4) { mel = E5; } else if (idx == 5) { mel = D5; } else if (idx == 6) { mel = E5; } else { mel = A4; }
        } else {
            mel_t = (pm - rd) + (rd / 8.0); mel = A4;
        }
    }

    // driving bass (the bounce)
    let step16 = i32(floor(pm * 16.0));
    let loop_m = m - (m / 4) * 4;
    let is_high = (step16 == 0 || step16 == 4 || step16 == 7 || step16 == 10 || step16 == 13);

    var bas_hi_f: f32 = D4;
    var bas_lo_f: f32 = D3;

    if (loop_m == 0) { bas_hi_f = D4; bas_lo_f = D3; }
    else if (loop_m == 1) { bas_hi_f = G4; bas_lo_f = G3; }
    else if (loop_m == 2) {
        if (step16 < 4) { bas_hi_f = G4; bas_lo_f = A3; }
        else { bas_hi_f = A4; bas_lo_f = A3; }
    } else if (loop_m == 3) {
        if (step16 < 4) { bas_hi_f = A4; bas_lo_f = F3; }
        else if (step16 < 7) { bas_hi_f = F4; bas_lo_f = F3; }
        else if (step16 < 10) { bas_hi_f = F4; bas_lo_f = E3; }
        else if (step16 < 13) { bas_hi_f = E4; bas_lo_f = E3; }
        else { bas_hi_f = E4; bas_lo_f = D3; }
    }

    var bas_drv: f32;
    if (is_high) { bas_drv = bas_hi_f; } else { bas_drv = bas_lo_f; }
    let bas_drv_t = fract(pm * 16.0) * (md / 16.0);

    // pad chords
    var pad_root = A4;
    var pad_minor = true;
    if (loop_m == 0) { pad_root = D4; pad_minor = true; }
    else if (loop_m == 1) { pad_root = G4; pad_minor = false; }
    else if (loop_m == 2) { pad_root = A4; pad_minor = true; }
    else if (loop_m == 3) {
        if (pm < 0.5) { pad_root = F4; pad_minor = false; }
        else { pad_root = E4; pad_minor = true; }
    }

    var data: SongData;
    data.mel = mel;
    data.mel_t = mel_t * md;
    data.drv_bas = bas_drv;
    data.drv_bas_t = bas_drv_t;
    data.bas_hi = is_high;
    data.pad_root = pad_root;
    data.pad_minor = pad_minor;
    return data;
}

// instruments

// 1. Drawbar organ lead
fn leadOrgan(freq: f32, phase_t: f32, env_t: f32) -> f32 {
    let hm = array<f32, 7>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0);
    let ha = array<f32, 7>(1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1);
    var y: f32 = 0.0;
    var tot: f32 = 0.0;
    for (var h = 0; h < 7; h++) {
        let amp = ha[h];
        y += sin(TAU * freq * hm[h] * phase_t) * amp;
        tot += amp;
    }
    let noise = fract(sin(dot(vec2<f32>(phase_t, freq), vec2<f32>(12.9898, 78.233))) * 43758.5453) * 2.0 - 1.0;
    y += noise * 0.05 * exp(-env_t * 50.0);
    return y / tot;
}

// 2. Soft Moog bass pluck
fn bassPluck(freq: f32, phase_t: f32, env_t: f32) -> f32 {
    let w1 = fract(phase_t * freq);
    let w2 = fract(phase_t * freq * 1.006);
    let saw = (w1 * 2.0 - 1.0) * 0.5 + (w2 * 2.0 - 1.0) * 0.5;
    let sine = sin(TAU * freq * phase_t);
    let bite = exp(-env_t * 18.0) * 0.4;
    return mix(sine, saw, bite);
}

// 3. Get lead + bass voices at a given delay time
fn getVoicesAtTime(delay_t: f32, phase_t: f32) -> vec2<f32> {
    let d = getSongData(delay_t);

    let lead_env = smoothstep(0.0, 0.03, d.mel_t) * exp(-d.mel_t * 0.25);
    let lead = leadOrgan(d.mel, phase_t, d.mel_t) * lead_env;

    var decayRate: f32 = 10.0;
    if (d.bas_hi) { decayRate = 5.0; }
    let bas_env = smoothstep(0.0, 0.02, d.drv_bas_t) * exp(-d.drv_bas_t * decayRate);
    let bass = bassPluck(d.drv_bas, phase_t, d.drv_bas_t) * bas_env;

    return vec2<f32>(lead, bass);
}

// 4. Triad helper
fn getTriad(root: f32, minor: bool) -> vec3<f32> {
    let n1 = root;
    var n2: f32;
    if (minor) { n2 = root * pow(2.0, 3.0 / 12.0); }
    else { n2 = root * pow(2.0, 4.0 / 12.0); }
    let n3 = root * pow(2.0, 7.0 / 12.0);
    return vec3<f32>(n1, n2, n3);
}

// 5. Guitar pad
fn guitarPad(root: f32, time: f32, minor: bool) -> f32 {
    let t = getTriad(root, minor);
    let padL = sin(TAU * t.x * time) + sin(TAU * t.x * 1.002 * time) * 0.3;
    let padC = sin(TAU * t.y * time) + sin(TAU * t.y * 1.001 * time) * 0.3;
    let padR = sin(TAU * t.z * time) + sin(TAU * t.z * 0.998 * time) * 0.3;
    let spread = sin(time * 0.2) * 0.5 + 0.5;
    return mix(padL, padC, spread) + mix(padC, padR, spread);
}

// 6. Drawbar organ synth (background chords)
fn organSynth(root: f32, time: f32, minor: bool) -> f32 {
    let tri = getTriad(root, minor);
    let roots = array<f32, 4>(root * 0.5, tri.x, tri.y, tri.z);
    let hm = array<f32, 7>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0);
    let ha = array<f32, 7>(1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1);
    var y: f32 = 0.0;
    var tot: f32 = 0.0;
    for (var i = 0; i < 4; i++) {
        for (var h = 0; h < 7; h++) {
            let freq = roots[i] * hm[h];
            let amp = ha[h] / (f32(i) + 1.0);
            y += sin(TAU * freq * time) * amp;
            tot += amp;
        }
    }
    return y / tot;
}

// 7. Kick drum
fn kickDrum(t: f32) -> f32 {
    let env = exp(-t * 12.0);
    let freq = mix(40.0, 150.0, exp(-t * 40.0));
    return sin(TAU * freq * t) * env;
}

fn mainSound(time: f32) -> vec2<f32> {
    // delay lines (phase-decoupled to avoid comb filtering)
    let v0 = getVoicesAtTime(time, time);
    let v1 = getVoicesAtTime(time - 0.15, time);
    let v2 = getVoicesAtTime(time - 0.30, time);
    let v3 = getVoicesAtTime(time - 0.45, time);

    let vR0 = getVoicesAtTime(time + 0.005, time + 0.005);
    let vR1 = getVoicesAtTime(time - 0.17, time + 0.005);
    let vR2 = getVoicesAtTime(time - 0.32, time + 0.005);
    let vR3 = getVoicesAtTime(time - 0.47, time + 0.005);

    let lead_L = v0.x * 0.40 + v1.x * 0.15 + v2.x * 0.08 + v3.x * 0.04;
    let lead_R = vR0.x * 0.40 + vR1.x * 0.15 + vR2.x * 0.08 + vR3.x * 0.04;

    let bas_L = v0.y * 0.35 + v1.y * 0.10 + v2.y * 0.05;
    let bas_R = vR0.y * 0.35 + vR1.y * 0.10 + vR2.y * 0.05;

    // background pads
    let data = getSongData(time);
    let wave_guitar = guitarPad(data.pad_root, time, data.pad_minor);
    let wave_organ = organSynth(data.pad_root, time, data.pad_minor);

    let md = measure_duration();
    let pm = fract((time % (md * 8.0)) / md);
    let m = i32((time % (md * 8.0)) / md);

    let kick_t = fract(pm * 4.0) * (md / 4.0);
    let wave_kick = kickDrum(kick_t);

    // sidechain duck
    var sc_depth: f32 = 0.35;
    if (m == 3 || m == 7) {
        sc_depth = mix(0.35, 0.05, pm);
    }
    let sidechain = 1.0 - (exp(-kick_t * 10.0) * sc_depth);

    let guitar_vol: f32 = 0.02;
    let organ_vol: f32 = 0.12;

    let L = (lead_L + bas_L + wave_guitar * guitar_vol + wave_organ * organ_vol) * sidechain + wave_kick * 0.6;
    let spread_pad = smoothstep(1.0, 0.0, pm) * 0.2;
    let R = (lead_R + bas_R + wave_guitar * guitar_vol * (1.0 + spread_pad) + wave_organ * organ_vol * (1.0 - spread_pad)) * sidechain + wave_kick * 0.6;

    // soft clip
    let Lc = (exp(2.0 * L) - 1.0) / (exp(2.0 * L) + 1.0);
    let Rc = (exp(2.0 * R) - 1.0) / (exp(2.0 * R) + 1.0);

    return vec2<f32>(Lc, Rc) * 0.5;
}


fn note_col(n: f32) -> vec3<f32> {
    let ni = u32(n);
    switch ni {
        case 0u: { return vec3<f32>(1.0, 0.2, 0.2); }
        case 1u: { return vec3<f32>(1.0, 0.5, 0.0); }
        case 2u: { return vec3<f32>(1.0, 0.9, 0.1); }
        case 3u: { return vec3<f32>(0.2, 1.0, 0.2); }
        case 4u: { return vec3<f32>(0.1, 0.6, 1.0); }
        case 5u: { return vec3<f32>(0.7, 0.2, 1.0); }
        default: { return vec3<f32>(0.5); }
    }
}

fn measure_data(m: u32) -> vec2<f32> {
    switch m {
        case 0u, 4u: { return vec2<f32>(F5, 5.0); }
        case 1u, 5u: { return vec2<f32>(B4, 1.0); }
        case 2u, 6u: { return vec2<f32>(E5, 4.0); }
        case 3u, 7u: { return vec2<f32>(A4, 0.0); }
        default: { return vec2<f32>(A4, 0.0); }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let d = textureDimensions(output);
    if (g.x >= d.x || g.y >= d.y) { return; }

    // audio: thread (0,0) fills the PCM buffer
    if (g.x == 0u && g.y == 0u) {
        let sr = u_song.sample_rate;
        let n = u_song.samples_to_generate;
        for (var i = 0u; i < n; i++) {
            let global_sample = u_song.sample_offset + i;
            let t = f32(global_sample) / sr;
            let stereo = mainSound(t * u_song.tempo_multiplier);
            let vol = u_song.volume;
            audio_buffer[i * 2u] = stereo.x * vol;
            audio_buffer[i * 2u + 1u] = stereo.y * vol;
        }
    }

    var uv = (vec2<f32>(g.xy) * 2.0 - vec2<f32>(d)) / f32(d.y);
    let len = length(uv);
    let ang = atan2(uv.y, -uv.x) + PI;

    let T = u_time.time * u_song.tempo_multiplier;
    let md = measure_duration();
    let td = md * 8.0;
    let data = getSongData(T);
    let cm = u32((T % td) / md);
    let pm = fract((T % td) / md);

    let mel_env = smoothstep(0.0, 0.03, data.mel_t) * exp(-data.mel_t * 0.25);
    let bas_env = exp(-data.drv_bas_t * 12.0);
    let kick_t = fract(pm * 4.0) * (md / 4.0);
    let kick_env = exp(-kick_t * 12.0);
    let sidechain = 1.0 - (exp(-kick_t * 10.0) * 0.3);

    var col = vec3<f32>(0.01, 0.015, 0.02);

    // pitch as a y position, with a glow that follows the melody
    let pitchNorm = clamp((data.mel - 400.0) / 300.0, 0.0, 1.0);
    let targetY = mix(-0.4, 0.5, pitchNorm);
    let mel_dist = abs(uv.y - targetY);
    let mel_color = mix(vec3<f32>(0.1, 0.5, 1.0), vec3<f32>(1.0, 0.3, 0.7), pitchNorm);
    col += mel_color * smoothstep(0.25, 0.0, mel_dist) * mel_env * 0.6;
    col += mel_color * smoothstep(0.005, 0.0, mel_dist) * mel_env;

    // bass: orange pulse at the bottom, brighter on accent hits
    let bas_y = -0.7;
    let bas_dist = abs(uv.y - bas_y);
    var bas_brightness = bas_env;
    if (data.bas_hi) { bas_brightness *= 1.5; }
    col += vec3<f32>(1.0, 0.4, 0.1) * smoothstep(0.15, 0.0, bas_dist) * bas_brightness * 0.5;
    col += vec3<f32>(1.0, 0.5, 0.15) * smoothstep(0.003, 0.0, bas_dist) * bas_brightness;

    // kick drum: flash from center
    col += vec3<f32>(0.2, 0.1, 0.05) * kick_env * smoothstep(0.6, 0.0, len) * 0.5;

    // pad chord: subtle background warmth that shifts with the harmony
    let pad_hue = data.pad_root / 500.0;
    let pad_col = vec3<f32>(0.5 + 0.5 * sin(pad_hue * TAU), 0.3, 0.5 + 0.5 * cos(pad_hue * TAU));
    col += pad_col * 0.015;

    // measure markers: 8 thin vertical lines so you can see the song structure
    let measure_x = fract(uv.x * 0.5 / (2.0 * f32(d.x) / f32(d.y)) * 8.0 + 0.5);
    // simpler: just use screen-space
    let sx = f32(g.x) / f32(d.x);
    for (var m = 1u; m < 8u; m++) {
        let mx = f32(m) / 8.0;
        if (abs(sx - mx) < 0.001) {
            col += vec3<f32>(0.04, 0.04, 0.07);
        }
    }

    // playhead: bright vertical line showing position within current measure
    let playhead_x = (f32(cm) + pm) / 8.0;
    if (abs(sx - playhead_x) < 0.002) {
        col += vec3<f32>(0.3, 0.35, 0.5) * 0.5;
    }

    // current measure gets a faint highlight
    let m_left = f32(cm) / 8.0;
    let m_right = f32(cm + 1u) / 8.0;
    if (sx > m_left && sx < m_right) {
        col += vec3<f32>(0.02, 0.025, 0.04);
    }

    col *= sidechain;
    col *= 1.1 - len * 0.35;

    textureStore(output, g.xy, vec4<f32>(col, 1.0));
}
