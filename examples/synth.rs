use cuneus::audio::PcmStreamManager;
use cuneus::compute::*;
use cuneus::prelude::*;
use log::error;
use std::ops::RangeInclusive;
use winit::keyboard::{KeyCode, PhysicalKey};

// enough audio for a frame even at low frame rates
const MAX_SAMPLES_PER_FRAME: u32 = 4096;
const SAMPLE_RATE: u32 = 44100;

cuneus::uniform_params! {
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
        play_sample: u32,
        // tablet playing
        tab_on: f32,
        key_on: [[u32; 4]; 4],
        key_off: [[u32; 4]; 4],
    }
}

// Tablet: songs written as rows of marks, like wedges pressed into clay (cuneus = wedge).
// One line per track: `name [instrument:] steps... [.effect(v)]*`. Steps share the bar evenly; `|` starts the
// next bar and a track loops its own bars. `.` rest, `-` hold, `x X o` hit / accent / ghost, `C4 Eb2 F#3` notes,
// `[C4 E4 G4]` a chord. `let name = steps` reuses a phrase, `#[tempo(120), swing(0.1)]` sets the groove.

// GPU layout (u32 words): header [switch sample, old slot, new slot], then two slots of SLOT words:
// slot [origin sample, tempo, -, tracks], per track 16 words at 16 + t * 16
// [kind, bars, events, event offset, lpf, drive, room, echo, gain, pan, -, name (4 words of 4 chars)],
// events from 16 + MAX_TRACKS * 16, 4 words each [start (bars), length (bars), note or 0, velocity]
const SLOT: usize = 4096;
const MAX_TRACKS: usize = 8;
const EV0: usize = 16 + MAX_TRACKS * 16;
const TABLET_WORDS: usize = 16 + 2 * SLOT;

// instruments 0..7, drums from 100
const INSTRUMENTS: [&str; 8] = ["saw", "square", "pluck", "epiano", "strings", "bass", "bell", "sine"];
const DRUMS: [&str; 5] = ["kick", "snare", "hat", "open", "clap"];
// default low-pass per instrument, Hz
const INST_LPF: [f32; 8] = [3200.0, 2200.0, 20000.0, 20000.0, 6000.0, 900.0, 20000.0, 20000.0];

struct Ev { start: f32, dur: f32, key: f32, vel: f32 }
struct Track { name: String, kind: u32, bars: u32, fx: [f32; 6], evs: Vec<Ev> }
struct Song { tempo: f32, tracks: Vec<Track> }

fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let (mut sq, mut par) = (0, 0);
    for c in s.chars() {
        match c {
            '[' => { sq += 1; cur.push(c); }
            ']' => { sq -= 1; cur.push(c); }
            '(' => { par += 1; cur.push(c); }
            ')' => { par -= 1; cur.push(c); }
            '|' if sq == 0 && par == 0 => {
                if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
                out.push("|".into());
            }
            c if c.is_whitespace() && sq == 0 && par == 0 => {
                if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

fn note(t: &str) -> Option<f32> {
    let mut ch = t.chars();
    let base = match ch.next()? { 'C' => 0, 'D' => 2, 'E' => 4, 'F' => 5, 'G' => 7, 'A' => 9, 'B' => 11, _ => return None };
    let rest: String = ch.collect();
    let (acc, oct) = if let Some(r) = rest.strip_prefix('#') { (1, r) } else if let Some(r) = rest.strip_prefix('b') { (-1, r) } else { (0, rest.as_str()) };
    let o: i32 = if oct.is_empty() { 4 } else { oct.parse().ok()? };
    Some(((o + 1) * 12 + base + acc) as f32)
}

// `name(value)` pairs, from `#[a(1), b(2)]` or `.a(1).b(2)`
fn calls(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while let Some(o) = s[i..].find('(') {
        let o = i + o;
        let c = s[o..].find(')').map_or(s.len(), |c| o + c);
        let mut k = o;
        while k > 0 && (b[k - 1].is_ascii_alphanumeric() || b[k - 1] == b'_') { k -= 1; }
        out.push((s[k..o].to_string(), s[o + 1..c].trim().to_string()));
        i = (c + 1).min(s.len());
    }
    out
}

fn parse_tablet(src: &str) -> Result<Song, String> {
    let mut song = Song { tempo: 120.0, tracks: Vec::new() };
    let mut swing = 0.0f32;
    let mut lets: std::collections::HashMap<String, Vec<String>> = Default::default();
    for (ln, raw) in src.lines().enumerate() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if line.is_empty() { continue; }
        let err = |m: String| format!("line {}: {m}", ln + 1);
        if let Some(a) = line.strip_prefix("#[") {
            for (k, v) in calls(a.trim_end_matches(']')) {
                let x: f32 = v.parse().map_err(|_| err(format!("'{v}' is not a number")))?;
                match k.as_str() {
                    "tempo" => song.tempo = x.clamp(30.0, 300.0),
                    "swing" => swing = x.clamp(0.0, 0.6),
                    _ => return Err(err(format!("unknown attribute '{k}' (tempo, swing)"))),
                }
            }
            continue;
        }
        if let Some(r) = line.strip_prefix("let ") {
            let (name, body) = r.split_once('=').ok_or_else(|| err("expected `let name = steps`".into()))?;
            lets.insert(name.trim().to_string(), tokenize(body));
            continue;
        }
        if song.tracks.len() >= MAX_TRACKS { return Err(err(format!("at most {MAX_TRACKS} tracks"))); }
        let toks = tokenize(line);
        let name = toks[0].trim_end_matches(':').to_string();
        let mut i = 1;
        let mut kind = DRUMS.iter().position(|d| *d == name).map(|d| 100 + d as u32);
        if toks.get(1).is_some_and(|t| t.ends_with(':')) {
            let inst = toks[1].trim_end_matches(':');
            let k = INSTRUMENTS.iter().position(|n| *n == inst)
                .ok_or_else(|| err(format!("unknown instrument '{inst}' ({})", INSTRUMENTS.join(" "))))?;
            kind = Some(k as u32);
            i = 2;
        }
        let kind = kind.ok_or_else(|| err(format!("'{name}' needs an instrument, e.g. `{name} saw: C4 ...`, or use a drum name ({})", DRUMS.join(" "))))?;
        let drum = kind >= 100;
        let mut fx = [if drum { 20000.0 } else { INST_LPF[kind as usize] }, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut octave = 0.0;
        // expand phrases, split off the effects
        let mut steps = Vec::new();
        for t in &toks[i..] {
            if t.len() > 1 && t.starts_with('.') && t.as_bytes()[1].is_ascii_alphabetic() {
                for (k, v) in calls(t) {
                    let x: f32 = v.parse().map_err(|_| err(format!("'{v}' is not a number")))?;
                    match k.as_str() {
                        "lpf" => fx[0] = x.clamp(40.0, 20000.0),
                        "drive" => fx[1] = x.clamp(0.0, 1.0),
                        "room" => fx[2] = x.clamp(0.0, 1.0),
                        "echo" => fx[3] = x.clamp(0.0, 1.0),
                        "gain" => fx[4] = x.clamp(0.0, 2.0),
                        "pan" => fx[5] = x.clamp(-1.0, 1.0),
                        "octave" => octave = x.round().clamp(-3.0, 3.0),
                        _ => return Err(err(format!("unknown effect '.{k}' (lpf drive room echo gain pan octave)"))),
                    }
                }
            } else if let Some(p) = lets.get(t) {
                steps.extend(p.iter().cloned());
            } else {
                steps.push(t.clone());
            }
        }
        let bars: Vec<&[String]> = steps.split(|t| t == "|").collect();
        let mut evs: Vec<Ev> = Vec::new();
        let mut last: Vec<usize> = Vec::new();
        for (b, bar) in bars.iter().enumerate() {
            if bar.is_empty() { return Err(err(format!("bar {} of '{name}' is empty", b + 1))); }
            let n = bar.len() as f32;
            for (s, t) in bar.iter().enumerate() {
                let len = 1.0 / n;
                let mut start = b as f32 + s as f32 * len;
                if (bar.len() == 8 || bar.len() == 16) && s % 2 == 1 { start += swing * 0.5 * len; }
                match t.as_str() {
                    "." => last.clear(),
                    "-" => { for &e in &last { evs[e].dur += len; } }
                    "x" | "X" | "o" => {
                        if !drum { return Err(err(format!("'{t}' is a drum hit; '{name}' plays notes"))); }
                        let vel = match t.as_str() { "X" => 1.0, "o" => 0.35, _ => 0.75 };
                        evs.push(Ev { start, dur: len, key: 0.0, vel });
                        last.clear();
                    }
                    _ => {
                        if drum { return Err(err(format!("'{t}': drum tracks use . x X o"))); }
                        let inner = t.strip_prefix('[').and_then(|r| r.strip_suffix(']'));
                        let notes: Vec<&str> = match inner { Some(r) => r.split_whitespace().collect(), None => vec![t.as_str()] };
                        last.clear();
                        for nt in notes {
                            let k = note(nt).ok_or_else(|| err(format!("'{nt}' is not a note (like C4, Eb2, F#3)")))?;
                            last.push(evs.len());
                            evs.push(Ev { start, dur: len, key: k + octave * 12.0, vel: 0.8 });
                        }
                    }
                }
            }
        }
        song.tracks.push(Track { name, kind, bars: bars.len() as u32, fx, evs });
    }
    let total: usize = song.tracks.iter().map(|t| t.evs.len()).sum();
    if EV0 + total * 4 > SLOT { return Err(format!("too many notes ({total}); the tablet holds {}", (SLOT - EV0) / 4)); }
    Ok(song)
}

fn song_words(song: &Song, origin: u32) -> Vec<u32> {
    let mut w = vec![0u32; SLOT];
    w[0] = origin;
    w[1] = song.tempo.to_bits();
    w[3] = song.tracks.len() as u32;
    let mut eo = EV0;
    for (t, tr) in song.tracks.iter().enumerate() {
        let h = 16 + t * 16;
        w[h] = tr.kind;
        w[h + 1] = tr.bars;
        w[h + 2] = tr.evs.len() as u32;
        w[h + 3] = eo as u32;
        for (k, v) in tr.fx.iter().enumerate() { w[h + 4 + k] = v.to_bits(); }
        let nb: Vec<u8> = tr.name.bytes().take(16).collect();
        for (k, c) in nb.iter().enumerate() { w[h + 11 + k / 4] |= (*c as u32) << ((k % 4) * 8); }
        for e in &tr.evs {
            w[eo] = e.start.to_bits(); w[eo + 1] = e.dur.to_bits(); w[eo + 2] = e.key.to_bits(); w[eo + 3] = e.vel.to_bits();
            eo += 4;
        }
    }
    w
}

const PRESETS: [(&str, &str); 3] = [
    ("House", "// one line per track: steps share the bar, | starts the next bar
#[tempo(122), swing(0.08)]

kick   X . . . x . . . X . . . x . . .
snare  . . . . X . . . . . . . X . . o
hat    . x . x . x . x . x . x . x . x
clap   . . . . . . . . . . . . x . . . | . . . . x . . . . . . . x . x .

let riff = C2 . C2 . Eb2 . C2 G1
bass   bass:   riff | riff | Ab1 . Ab1 . Bb1 . Bb1 C2   .drive(0.2)
keys   epiano: [C4 Eb4 G4] - - - . . [Bb3 D4 F4] - | [Ab3 C4 Eb4] - - - . . [G3 Bb3 D4] -   .room(0.35).echo(0.2)
lead   pluck:  . . G4 . Bb4 . C5 . | . Eb5 . D5 . C5 . .   .echo(0.4).pan(0.3)
"),
    ("Ambient", "#[tempo(76)]

pad    strings: [C3 G3 E4] - - - - - - - | [A2 E3 C4] - - - - - - - | [F2 C3 A3] - - - - - - - | [G2 D3 B3] - - - - - - -   .room(0.7)
bell   bell:    . . . G5 . . . . | . . E5 . . . D5 . | . . . C5 . . . . | . . B4 . . . . .   .echo(0.5).room(0.5).pan(-0.3)
low    sine:    C2 - - - - - - - | A1 - - - - - - - | F1 - - - - - - - | G1 - - - - - - -   .gain(0.8)
hat    . . o . . . x .
kick   X . . . . . . . . . . . . . . .
"),
    ("Disco", "#[tempo(112), swing(0.05)]

kick   X . . . X . . . X . . . X . . .
hat    . . X . . . X . . . X . . . X .
snare  . . . . x . . . . . . . x . . .
bass   bass:    A1 A2 A1 A2 A1 A2 A1 A2 | F1 F2 F1 F2 F1 F2 F1 F2 | G1 G2 G1 G2 G1 G2 G1 G2 | E1 E2 E1 E2 E1 E2 E1 E2   .lpf(900)
arp    saw:     A4 C5 E5 C5 A4 C5 E5 C5 | F4 A4 C5 A4 F4 A4 C5 A4 | G4 B4 D5 B4 G4 B4 D5 B4 | E4 G#4 B4 G#4 E4 G#4 B4 G#4   .lpf(2400).echo(0.3).gain(0.6)
pad    strings: [A3 C4 E4] - - - - - - - | [F3 A3 C4] - - - - - - - | [G3 B3 D4] - - - - - - - | [E3 G#3 B3] - - - - - - -   .room(0.5).gain(0.7)
"),
];

// piano layout on the top letter row by physical position: white keys Q W E R T Y U I O, black keys 2 3 5 6 7 9 0
const KEYS: [KeyCode; 16] = [
    KeyCode::KeyQ, KeyCode::Digit2, KeyCode::KeyW, KeyCode::Digit3, KeyCode::KeyE, KeyCode::KeyR,
    KeyCode::Digit5, KeyCode::KeyT, KeyCode::Digit6, KeyCode::KeyY, KeyCode::Digit7, KeyCode::KeyU,
    KeyCode::KeyI, KeyCode::Digit9, KeyCode::KeyO, KeyCode::Digit0,
];

const WAVES: [&str; 12] = ["Sin", "Saw", "Sqr", "Tri", "Pulse", "Super", "FM", "Organ", "Noise", "Guitar", "E-Piano", "Strings"];

// a patch sets every sound field, so the result never depends on the previous patch
struct Patch {
    name: &'static str,
    wave: u32,
    octave: f32,
    // attack, decay, sustain, release
    adsr: [f32; 4],
    // cutoff, resonance, drive
    filter: [f32; 3],
    // chorus rate, chorus depth, delay time, delay feedback, reverb
    space: [f32; 5],
}

const PATCHES: [Patch; 8] = [
    Patch { name: "Cathedral", wave: 7, octave: 4.0, adsr: [0.4, 0.6, 0.85, 1.8], filter: [0.5, 0.15, 0.0], space: [0.6, 0.2, 0.33, 0.0, 0.6] },
    Patch { name: "Harpsichord", wave: 4, octave: 5.0, adsr: [0.005, 1.5, 0.0, 0.3], filter: [0.8, 0.1, 0.0], space: [1.2, 0.1, 0.33, 0.0, 0.35] },
    Patch { name: "Choir", wave: 5, octave: 4.0, adsr: [0.5, 0.5, 0.8, 1.5], filter: [0.55, 0.1, 0.0], space: [0.8, 0.35, 0.33, 0.0, 0.55] },
    Patch { name: "Bells", wave: 6, octave: 5.0, adsr: [0.005, 0.8, 0.2, 1.0], filter: [0.85, 0.05, 0.0], space: [1.2, 0.2, 0.33, 0.0, 0.5] },
    Patch { name: "Guitar", wave: 9, octave: 3.0, adsr: [0.001, 1.0, 0.7, 0.4], filter: [0.9, 0.05, 0.0], space: [1.2, 0.15, 0.28, 0.2, 0.25] },
    Patch { name: "E-Piano", wave: 10, octave: 4.0, adsr: [0.002, 1.2, 0.35, 0.5], filter: [0.85, 0.05, 0.0], space: [0.9, 0.3, 0.33, 0.0, 0.3] },
    Patch { name: "Strings", wave: 11, octave: 4.0, adsr: [0.25, 0.5, 0.9, 1.2], filter: [0.62, 0.1, 0.0], space: [0.5, 0.3, 0.33, 0.0, 0.5] },
    Patch { name: "Lead", wave: 1, octave: 4.0, adsr: [0.01, 0.2, 0.5, 0.3], filter: [0.65, 0.3, 0.0], space: [1.2, 0.25, 0.33, 0.35, 0.2] },
];

impl Patch {
    fn apply(&self, p: &mut SynthParams) {
        p.waveform_type = self.wave;
        p.octave = self.octave;
        [p.attack_time, p.decay_time, p.sustain_level, p.release_time] = self.adsr;
        [p.filter_cutoff, p.filter_resonance, p.distortion_amount] = self.filter;
        [p.chorus_rate, p.chorus_depth, p.delay_time, p.delay_feedback, p.reverb_mix] = self.space;
    }
}

fn slider(ui: &mut egui::Ui, v: &mut f32, range: RangeInclusive<f32>, label: &str) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

fn seconds(ui: &mut egui::Ui, v: &mut f32, range: RangeInclusive<f32>, label: &str) -> bool {
    ui.add(egui::Slider::new(v, range).logarithmic(true).text(label).suffix(" s")).changed()
}

struct SynthManager {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SynthParams,
    pcm_stream: Option<PcmStreamManager>,
    keys_held: [bool; 16],
    last_samples_generated: u32,
    tablet_src: String,
    tablet_err: Option<String>,
    // slot playing, where its bar 0 began, its tempo
    tab_slot: usize,
    tab_origin: u32,
    tab_tempo: f32,
    tab_loaded: bool,
}

impl SynthManager {
    /// First sample not generated yet, so a note always starts with its full attack; 0 means none
    fn next_sample(&self) -> u32 {
        let written = self.pcm_stream.as_ref().map_or(0, |s| s.samples_written());
        ((written + self.last_samples_generated as u64) as u32).max(1)
    }

    // parse into the other slot, switched at the next bar
    fn apply_tablet(&mut self, core: &Core, now_: bool) {
        let song = match parse_tablet(&self.tablet_src) {
            Ok(s) => s,
            Err(e) => { self.tablet_err = Some(e); return; }
        };
        self.tablet_err = None;
        let now = self.next_sample() + 512;
        let (slot, at) = if now_ && self.tab_loaded && self.current_params.tab_on > 0.5 {
            (1 - self.tab_slot, now)
        } else if self.tab_loaded && self.current_params.tab_on > 0.5 {
            let bar = (240.0 / self.tab_tempo * SAMPLE_RATE as f32) as u64;
            let since = now.saturating_sub(self.tab_origin) as u64;
            (1 - self.tab_slot, self.tab_origin + (since.div_ceil(bar) * bar) as u32)
        } else {
            (self.tab_slot, now)
        };
        let buf = &self.compute_shader.storage_buffers[1];
        core.queue.write_buffer(buf, ((16 + slot * SLOT) * 4) as u64, cuneus::bytemuck::cast_slice(&song_words(&song, at)));
        let old = if self.tab_loaded { self.tab_slot } else { slot };
        core.queue.write_buffer(buf, 0, cuneus::bytemuck::cast_slice(&[at, old as u32, slot as u32]));
        self.tab_slot = slot;
        self.tab_origin = at;
        self.tab_tempo = song.tempo;
        self.tab_loaded = true;
        self.current_params.tab_on = 1.0;
    }
}

impl ShaderManager for SynthManager {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        let initial_params = SynthParams {
            tempo: 120.0,
            waveform_type: 1,
            octave: 4.0,
            volume: 0.6,
            beat_enabled: 0,
            reverb_mix: 0.22,
            delay_time: 0.33,
            delay_feedback: 0.25,
            filter_cutoff: 0.68,
            filter_resonance: 0.25,
            distortion_amount: 0.0,
            chorus_rate: 1.2,
            chorus_depth: 0.25,
            attack_time: 0.01,
            decay_time: 0.25,
            sustain_level: 0.6,
            release_time: 0.5,
            sample_offset: 0,
            samples_to_generate: MAX_SAMPLES_PER_FRAME,
            sample_rate: SAMPLE_RATE,
            drum_level: 1.0,
            swing: 0.0,
            play_sample: 0,
            tab_on: 0.0,
            key_on: [[0; 4]; 4],
            key_off: [[0; 4]; 4],
        };

        let audio_buffer_size = (MAX_SAMPLES_PER_FRAME * 2) as usize;
        // dsp: DSP state (~62,110 floats), from 65,536 the tablet bus, then its echo line (1 s)
        let dsp_buffer_size = ((65536 + 4 * MAX_SAMPLES_PER_FRAME as usize + 44100) * std::mem::size_of::<f32>()) as u64;
        let passes = vec![
            // one thread per sample for the tablet voices, then the sequential effects and the picture
            PassDescription::new("voices", &[]).with_workgroup_size([MAX_SAMPLES_PER_FRAME / 256, 1, 1]),
            PassDescription::new("main", &[]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("voices")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<SynthParams>()
            .with_fonts()
            .with_audio(audio_buffer_size)
            .with_storage_buffer(StorageBufferSpec::new("dsp", dsp_buffer_size))
            .with_storage_buffer(StorageBufferSpec::new("tablet", (TABLET_WORDS * 4) as u64))
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Synth")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/synth.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        let pcm_stream = match PcmStreamManager::new(Some(SAMPLE_RATE)) {
            Ok(mut stream) => {
                if let Err(e) = stream.start() {
                    error!("Failed to start PCM stream: {e}");
                    None
                } else {
                    Some(stream)
                }
            }
            Err(e) => {
                error!("Failed to create PCM stream: {e}");
                None
            }
        };

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            pcm_stream,
            keys_held: [false; 16],
            last_samples_generated: 0,
            tablet_src: PRESETS[0].1.to_string(),
            tablet_err: None,
            tab_slot: 0,
            tab_origin: 0,
            tab_tempo: 120.0,
            tab_loaded: false,
        }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader
            .set_time(current_time, 1.0 / 60.0, &core.queue);

        if let Some(ref mut stream) = self.pcm_stream {
            stream.set_master_volume(self.current_params.volume as f64);

            // Push previous frame's audio first
            let prev = self.last_samples_generated;
            if prev > 0 {
                if let Ok(audio_data) = pollster::block_on(
                    self.compute_shader
                        .read_audio_buffer(&core.device, &core.queue),
                ) {
                    let count = (prev * 2) as usize;
                    if audio_data.len() >= count {
                        let _ = stream.push_samples(&audio_data[..count]);
                    }
                }
            }

            // Calculate this frame's needs
            let (written, needed) = stream.next_block(MAX_SAMPLES_PER_FRAME);
            self.current_params.play_sample = stream.playback_sample() as u32;
            self.current_params.sample_offset = written as u32;
            self.current_params.samples_to_generate = needed;
            self.last_samples_generated = needed;
        }
        self.compute_shader
            .set_custom_params(self.current_params, &core.queue);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = false;
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let mut src = std::mem::take(&mut self.tablet_src);
        let tab_err = self.tablet_err.clone();
        let mut apply = false;
        let mut at_once = false;
        let mut stop = false;
        let playing = self.current_params.tab_on > 0.5;
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Tablet")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(460.0)
                    .default_pos([330.0, 10.0])
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            if ui.button(if playing { "Apply" } else { "Play" }).clicked() { apply = true; }
                            if playing && ui.button("Stop").clicked() { stop = true; }
                            ui.separator();
                            for (name, text) in &PRESETS {
                                if ui.button(*name).clicked() { src = text.to_string(); apply = true; at_once = true; }
                            }
                        });
                        // Ctrl/Cmd+Enter applies (consumed before the editor)
                        if ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Enter)) { apply = true; at_once = true; }
                        if ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter)) { apply = true; }
                        egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                            ui.add(egui::TextEdit::multiline(&mut src).code_editor().desired_rows(12).desired_width(f32::INFINITY));
                        });
                        if let Some(e) = &tab_err { ui.colored_label(egui::Color32::from_rgb(255, 120, 100), e); }
                        ui.small("Ctrl+Enter applies at the next bar, Shift+Ctrl+Enter at once.  . rest  - hold  x X o hit  C4 Eb2 F#3 notes  [C4 E4 G4] chord  | next bar");
                        ui.small(format!("instruments: {}   drums: {}   effects: .lpf .drive .room .echo .gain .pan .octave", INSTRUMENTS.join(" "), DRUMS.join(" ")));
                    });

                egui::Window::new("GPU Synth")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        ui.label("Play: Q W E R T Y U I O, black keys 2 3 5 6 7 9 0");
                        egui::CollapsingHeader::new("Sound").default_open(true).show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                for patch in &PATCHES {
                                    if ui.button(patch.name).clicked() {
                                        patch.apply(&mut params);
                                        changed = true;
                                    }
                                }
                            });
                            ui.separator();
                            ui.horizontal_wrapped(|ui| {
                                for (i, name) in WAVES.iter().enumerate() {
                                    if ui.selectable_label(params.waveform_type == i as u32, *name).clicked() {
                                        params.waveform_type = i as u32;
                                        changed = true;
                                    }
                                }
                            });
                            changed |= ui.add(egui::Slider::new(&mut params.octave, 2.0..=7.0).step_by(1.0).text("Octave")).changed();
                            changed |= slider(ui, &mut params.volume, 0.0..=1.0, "Volume");
                        });
                        egui::CollapsingHeader::new("Envelope").default_open(true).show(ui, |ui| {
                            changed |= seconds(ui, &mut params.attack_time, 0.001..=0.5, "Attack");
                            changed |= seconds(ui, &mut params.decay_time, 0.01..=1.5, "Decay");
                            changed |= slider(ui, &mut params.sustain_level, 0.0..=1.0, "Sustain");
                            changed |= seconds(ui, &mut params.release_time, 0.01..=2.0, "Release");
                        });
                        egui::CollapsingHeader::new("Filter & Drive").show(ui, |ui| {
                            changed |= slider(ui, &mut params.filter_cutoff, 0.0..=1.0, "Cutoff");
                            changed |= slider(ui, &mut params.filter_resonance, 0.0..=0.9, "Resonance");
                            changed |= slider(ui, &mut params.distortion_amount, 0.0..=0.9, "Drive");
                        });
                        egui::CollapsingHeader::new("Space").show(ui, |ui| {
                            changed |= slider(ui, &mut params.chorus_rate, 0.1..=10.0, "Chorus rate");
                            changed |= slider(ui, &mut params.chorus_depth, 0.0..=0.5, "Chorus depth");
                            changed |= slider(ui, &mut params.delay_time, 0.01..=1.0, "Delay");
                            changed |= slider(ui, &mut params.delay_feedback, 0.0..=0.8, "Feedback");
                            changed |= slider(ui, &mut params.reverb_mix, 0.0..=0.8, "Reverb");
                        });
                        egui::CollapsingHeader::new("Drums").show(ui, |ui| {
                            let mut on = params.beat_enabled > 0;
                            if ui.checkbox(&mut on, "On").changed() {
                                params.beat_enabled = u32::from(on);
                                changed = true;
                            }
                            changed |= slider(ui, &mut params.drum_level, 0.0..=2.0, "Level");
                            changed |= slider(ui, &mut params.swing, 0.0..=0.6, "Swing");
                            changed |= slider(ui, &mut params.tempo, 60.0..=180.0, "Tempo");
                        });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.tablet_src = src;
        if apply { self.apply_tablet(core, at_once); }
        if stop { self.current_params.tab_on = 0.0; }
        if changed {
            // Preserve audio fields that are managed by update()
            params.sample_offset = self.current_params.sample_offset;
            params.samples_to_generate = self.current_params.samples_to_generate;
            params.sample_rate = self.current_params.sample_rate;
            params.play_sample = self.current_params.play_sample;
            params.key_on = self.current_params.key_on;
            params.key_off = self.current_params.key_off;
            params.tab_on = self.current_params.tab_on;
            self.current_params = params;
        }

        self.base.apply_control_request(controls_request);

        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(
            &mut frame.encoder,
            &frame.view,
            &self.compute_shader.get_output_texture().bind_group,
        );

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.forward_to_egui(core, event) {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let PhysicalKey::Code(code) = event.physical_key {
                if let Some(index) = KEYS.iter().position(|k| *k == code) {
                    let now = self.next_sample();
                    let (row, col) = (index / 4, index % 4);
                    let pressed = event.state == winit::event::ElementState::Pressed;
                    if pressed && !self.keys_held[index] {
                        self.keys_held[index] = true;
                        self.current_params.key_on[row][col] = now;
                        self.current_params.key_off[row][col] = 0;
                    } else if !pressed {
                        self.keys_held[index] = false;
                        self.current_params.key_off[row][col] = now;
                    }
                    self.compute_shader.set_custom_params(self.current_params, &core.queue);
                    return true;
                }
            }
            return self
                .base
                .key_handler
                .handle_keyboard_input(core.window(), event);
        }

        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    cuneus::gst::init()?;

    let (app, event_loop) = ShaderApp::new("Synth", 800, 600);
    app.run(event_loop, SynthManager::init)
}
