use anyhow::{anyhow, Result};
use gst::glib::ControlFlow;
use gst::prelude::*;
use gstreamer as gst;
use log::{debug, info, warn};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Envelope phase for ADSR
#[derive(Debug, Clone, Copy, PartialEq)]
enum EnvelopePhase {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// ADSR envelope configuration
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeConfig {
    pub attack_time: f32,
    pub decay_time: f32,
    pub sustain_level: f32,
    pub release_time: f32,
}

impl Default for EnvelopeConfig {
    fn default() -> Self {
        Self {
            attack_time: 0.01,
            decay_time: 0.1,
            sustain_level: 0.7,
            release_time: 0.3,
        }
    }
}

/// Envelope state for a single voice
#[derive(Debug, Clone)]
struct EnvelopeState {
    phase: EnvelopePhase,
    current_level: f32,
    phase_start_time: Instant,
    phase_start_level: f32,
    target_frequency: f64,
    config: EnvelopeConfig,
}

impl EnvelopeState {
    fn new() -> Self {
        Self {
            phase: EnvelopePhase::Idle,
            current_level: 0.0,
            phase_start_time: Instant::now(),
            phase_start_level: 0.0,
            target_frequency: 440.0,
            config: EnvelopeConfig::default(),
        }
    }

    fn trigger(&mut self, frequency: f64, config: EnvelopeConfig) {
        self.target_frequency = frequency;
        self.config = config;
        self.phase_start_level = self.current_level;
        self.phase_start_time = Instant::now();
        self.phase = EnvelopePhase::Attack;
    }

    fn release(&mut self) {
        if self.phase != EnvelopePhase::Idle && self.phase != EnvelopePhase::Release {
            self.phase_start_level = self.current_level;
            self.phase_start_time = Instant::now();
            self.phase = EnvelopePhase::Release;
        }
    }

    fn update(&mut self) -> f32 {
        let elapsed = self.phase_start_time.elapsed().as_secs_f32();

        match self.phase {
            EnvelopePhase::Idle => {
                self.current_level = 0.0;
            }
            EnvelopePhase::Attack => {
                if self.config.attack_time > 0.0 {
                    let progress = (elapsed / self.config.attack_time).min(1.0);
                    // Smooth curve for attack
                    let curve = smooth_step(progress);
                    self.current_level =
                        self.phase_start_level + (1.0 - self.phase_start_level) * curve;

                    if progress >= 1.0 {
                        self.phase = EnvelopePhase::Decay;
                        self.phase_start_time = Instant::now();
                        self.phase_start_level = 1.0;
                    }
                } else {
                    self.current_level = 1.0;
                    self.phase = EnvelopePhase::Decay;
                    self.phase_start_time = Instant::now();
                    self.phase_start_level = 1.0;
                }
            }
            EnvelopePhase::Decay => {
                if self.config.decay_time > 0.0 {
                    let progress = (elapsed / self.config.decay_time).min(1.0);
                    let curve = smooth_step(progress);
                    self.current_level = 1.0 - (1.0 - self.config.sustain_level) * curve;

                    if progress >= 1.0 {
                        self.phase = EnvelopePhase::Sustain;
                        self.phase_start_time = Instant::now();
                    }
                } else {
                    self.current_level = self.config.sustain_level;
                    self.phase = EnvelopePhase::Sustain;
                    self.phase_start_time = Instant::now();
                }
            }
            EnvelopePhase::Sustain => {
                self.current_level = self.config.sustain_level;
            }
            EnvelopePhase::Release => {
                if self.config.release_time > 0.0 {
                    let progress = (elapsed / self.config.release_time).min(1.0);
                    // Use exponential decay for more natural release
                    let curve = 1.0 - smooth_step(progress);
                    self.current_level = self.phase_start_level * curve;

                    if progress >= 1.0 || self.current_level < 0.001 {
                        self.phase = EnvelopePhase::Idle;
                        self.current_level = 0.0;
                    }
                } else {
                    self.phase = EnvelopePhase::Idle;
                    self.current_level = 0.0;
                }
            }
        }

        self.current_level
    }

    fn is_active(&self) -> bool {
        self.phase != EnvelopePhase::Idle
    }

    fn is_releasing(&self) -> bool {
        self.phase == EnvelopePhase::Release
    }
}

/// Attempt smoothstep for smoother transitions
fn smooth_step(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Unison oscillator - one of multiple oscillators per voice
struct UnisonOscillator {
    audiotestsrc: gst::Element,
    volume: gst::Element,
    detune_cents: f64,
}

/// Individual audio voice for polyphonic synthesis with envelope
struct AudioVoice {
    oscillators: Vec<UnisonOscillator>,
    envelope: EnvelopeState,
    last_applied_volume: f64,
    base_frequency: f64,
    note: Option<MusicalNote>,
}

impl AudioVoice {
    /// Apply detuned frequency to all oscillators
    fn set_frequency(&mut self, freq: f64) {
        self.base_frequency = freq;
        for osc in &self.oscillators {
            // Convert cents to frequency multiplier: 2^(cents/1200)
            let detune_mult = 2.0_f64.powf(osc.detune_cents / 1200.0);
            let detuned_freq = freq * detune_mult;
            osc.audiotestsrc.set_property("freq", detuned_freq);
        }
    }

    fn apply_envelope(&mut self) {
        let level = self.envelope.update();
        let target_volume = (level as f64 * 0.35) / self.oscillators.len() as f64;

        // Smooth the volume changes to avoid clicks
        let smoothed = self.last_applied_volume + (target_volume - self.last_applied_volume) * 0.3;

        if (smoothed - self.last_applied_volume).abs() > 0.0001 {
            for osc in &self.oscillators {
                osc.volume.set_property("volume", smoothed);
                osc.audiotestsrc.set_property("volume", smoothed);
            }
            self.last_applied_volume = smoothed;
        }
    }

    /// Set waveform for all oscillators
    fn set_waveform(&self, wave: &str) {
        for osc in &self.oscillators {
            osc.audiotestsrc.set_property_from_str("wave", wave);
        }
    }
}

pub struct AudioSynthManager {
    pipeline: gst::Pipeline,
    _audiomixer: gst::Element,
    voices: Vec<AudioVoice>,
    current_waveform: AudioWaveform,
    master_volume: Arc<Mutex<f64>>,
    sample_rate: u32,
    last_update: Instant,
    active_notes: Arc<Mutex<Vec<MusicalNote>>>,
    default_envelope: EnvelopeConfig,
    update_interval: std::time::Duration,
    last_envelope_update: Instant,
}

impl AudioSynthManager {
    pub fn new(sample_rate: Option<u32>) -> Result<Self> {
        let sample_rate = sample_rate.unwrap_or(44100);
        let max_voices = 16;

        info!("Creating polyphonic audio synthesis manager with {max_voices} voices at {sample_rate} Hz");

        let pipeline = gst::Pipeline::new();

        let audiomixer = gst::ElementFactory::make("audiomixer")
            .name("audio_mixer")
            .build()
            .map_err(|_| anyhow!("Failed to create audiomixer element"))?;

        let final_convert = gst::ElementFactory::make("audioconvert")
            .name("final_convert")
            .build()
            .map_err(|_| anyhow!("Failed to create final audioconvert element"))?;

        let final_resample = gst::ElementFactory::make("audioresample")
            .name("final_resample")
            .build()
            .map_err(|_| anyhow!("Failed to create final audioresample element"))?;

        // Add a limiter to prevent clipping with multiple voices
        let master_volume = gst::ElementFactory::make("volume")
            .name("master_volume")
            .property("volume", 0.5f64)
            .build()
            .map_err(|_| anyhow!("Failed to create master volume element"))?;

        let audiosink = gst::ElementFactory::make("autoaudiosink")
            .name("audio_sink")
            .build()
            .map_err(|_| anyhow!("Failed to create autoaudiosink element"))?;

        pipeline
            .add_many([
                &audiomixer,
                &final_convert,
                &final_resample,
                &master_volume,
                &audiosink,
            ])
            .map_err(|_| anyhow!("Failed to add output elements to pipeline"))?;

        gst::Element::link_many([
            &audiomixer,
            &final_convert,
            &final_resample,
            &master_volume,
            &audiosink,
        ])
        .map_err(|_| anyhow!("Failed to link output elements"))?;

        let mut voices = Vec::new();
        for i in 0..max_voices {
            let voice = Self::create_voice(&pipeline, &audiomixer, i, sample_rate)?;
            voices.push(voice);
        }

        let bus = pipeline.bus().expect("Pipeline has no bus");
        let _ = bus.add_watch(move |_, message| {
            match message.view() {
                gst::MessageView::Error(err) => {
                    warn!(
                        "Audio pipeline error: {} ({})",
                        err.error(),
                        err.debug().unwrap_or_default()
                    );
                }
                gst::MessageView::Warning(warning) => {
                    debug!("Audio pipeline warning: {}", warning.error());
                }
                gst::MessageView::Eos(_) => {
                    info!("Audio pipeline reached end of stream");
                }
                _ => (),
            }
            ControlFlow::Continue
        });

        Ok(Self {
            pipeline,
            _audiomixer: audiomixer,
            voices,
            current_waveform: AudioWaveform::Sine,
            master_volume: Arc::new(Mutex::new(0.5)),
            sample_rate,
            last_update: Instant::now(),
            active_notes: Arc::new(Mutex::new(Vec::new())),
            default_envelope: EnvelopeConfig::default(),
            update_interval: std::time::Duration::from_millis(5),
            last_envelope_update: Instant::now(),
        })
    }

    fn create_voice(
        pipeline: &gst::Pipeline,
        mixer: &gst::Element,
        voice_id: usize,
        _sample_rate: u32,
    ) -> Result<AudioVoice> {
        // Unison detune amounts in cents (100 cents = 1 semitone)
        let detune_values = [-12.0, 0.0, 12.0];
        let mut oscillators = Vec::new();

        for (osc_idx, &detune_cents) in detune_values.iter().enumerate() {
            let audiotestsrc = gst::ElementFactory::make("audiotestsrc")
                .name(format!("voice_{voice_id}_osc_{osc_idx}"))
                .property("freq", 440.0f64)
                .property("volume", 0.0f64)
                .property("samplesperbuffer", 512i32)
                .property("is-live", true)
                .build()
                .map_err(|_| {
                    anyhow!(
                        "Failed to create audiotestsrc for voice {} osc {}",
                        voice_id,
                        osc_idx
                    )
                })?;

            audiotestsrc.set_property_from_str("wave", "sine");

            let audioconvert = gst::ElementFactory::make("audioconvert")
                .name(format!("voice_{voice_id}_osc_{osc_idx}_convert"))
                .build()
                .map_err(|_| {
                    anyhow!(
                        "Failed to create audioconvert for voice {} osc {}",
                        voice_id,
                        osc_idx
                    )
                })?;

            let audioresample = gst::ElementFactory::make("audioresample")
                .name(format!("voice_{voice_id}_osc_{osc_idx}_resample"))
                .build()
                .map_err(|_| {
                    anyhow!(
                        "Failed to create audioresample for voice {} osc {}",
                        voice_id,
                        osc_idx
                    )
                })?;

            let volume = gst::ElementFactory::make("volume")
                .name(format!("voice_{voice_id}_osc_{osc_idx}_volume"))
                .property("volume", 0.0f64)
                .build()
                .map_err(|_| {
                    anyhow!(
                        "Failed to create volume for voice {} osc {}",
                        voice_id,
                        osc_idx
                    )
                })?;

            pipeline
                .add_many([&audiotestsrc, &audioconvert, &audioresample, &volume])
                .map_err(|_| {
                    anyhow!("Failed to add voice {} osc {} elements", voice_id, osc_idx)
                })?;

            gst::Element::link_many([&audiotestsrc, &audioconvert, &audioresample, &volume])
                .map_err(|_| {
                    anyhow!("Failed to link voice {} osc {} elements", voice_id, osc_idx)
                })?;

            volume.link(mixer).map_err(|_| {
                anyhow!("Failed to link voice {} osc {} to mixer", voice_id, osc_idx)
            })?;

            oscillators.push(UnisonOscillator {
                audiotestsrc,
                volume,
                detune_cents,
            });
        }

        debug!(
            "Created voice {} with {} unison oscillators",
            voice_id,
            oscillators.len()
        );

        Ok(AudioVoice {
            oscillators,
            envelope: EnvelopeState::new(),
            last_applied_volume: 0.0,
            base_frequency: 440.0,
            note: None,
        })
    }

    pub fn set_envelope_config(&mut self, config: EnvelopeConfig) {
        self.default_envelope = config;
    }

    pub fn start(&mut self) -> Result<()> {
        info!("Starting polyphonic audio synthesis");
        match self.pipeline.set_state(gst::State::Playing) {
            Ok(_) => {
                std::thread::sleep(std::time::Duration::from_millis(50));
                Ok(())
            }
            Err(e) => Err(anyhow!("Failed to start audio synthesis: {:?}", e)),
        }
    }

    pub fn stop(&mut self) -> Result<()> {
        info!("Stopping audio synthesis");
        for voice in &mut self.voices {
            // Silence all oscillators in this voice
            for osc in &voice.oscillators {
                osc.volume.set_property("volume", 0.0f64);
                osc.audiotestsrc.set_property("volume", 0.0f64);
            }
            voice.last_applied_volume = 0.0;
            voice.envelope = EnvelopeState::new();
            voice.note = None;
        }
        self.active_notes.lock().unwrap().clear();

        match self.pipeline.set_state(gst::State::Null) {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow!("Failed to stop audio synthesis: {:?}", e)),
        }
    }

    pub fn set_master_volume(&mut self, vol: f64) -> Result<()> {
        let clamped_volume = vol.max(0.0).min(1.0);
        *self.master_volume.lock().unwrap() = clamped_volume;

        if let Some(master_vol) = self.pipeline.by_name("master_volume") {
            master_vol.set_property("volume", clamped_volume);
        }

        debug!("Set master volume to {clamped_volume:.3}");
        Ok(())
    }

    pub fn set_waveform(&mut self, wave_type: AudioWaveform) -> Result<()> {
        let wave_str = match wave_type {
            AudioWaveform::Sine => "sine",
            AudioWaveform::Square => "square",
            AudioWaveform::Saw => "saw",
            AudioWaveform::Triangle => "triangle",
        };

        self.current_waveform = wave_type;

        for voice in &self.voices {
            voice.set_waveform(wave_str);
        }

        debug!("Set waveform to {wave_type:?}");
        Ok(())
    }

    pub fn play_frequency(&mut self, frequency: f64, voice_id: usize) -> Result<()> {
        self.play_frequency_with_config(frequency, voice_id, self.default_envelope)
    }

    pub fn play_frequency_with_config(
        &mut self,
        frequency: f64,
        voice_id: usize,
        envelope_config: EnvelopeConfig,
    ) -> Result<()> {
        if voice_id >= self.voices.len() {
            return Err(anyhow!("Voice ID {} out of range", voice_id));
        }

        let voice = &mut self.voices[voice_id];

        // Set frequency on all oscillators (with detuning)
        voice.set_frequency(frequency);

        // Trigger envelope
        voice.envelope.trigger(frequency, envelope_config);
        voice.note = None;

        debug!("Triggered frequency {frequency:.2} Hz on voice {voice_id}");
        Ok(())
    }

    /// Release a specific voice (start release phase, don't stop immediately)
    pub fn release_voice(&mut self, voice_id: usize) -> Result<()> {
        if voice_id >= self.voices.len() {
            return Err(anyhow!("Voice ID {} out of range", voice_id));
        }

        let voice = &mut self.voices[voice_id];
        voice.envelope.release();
        debug!("Released voice {voice_id}");
        Ok(())
    }

    /// Stop a voice immediately (use sparingly, prefer release_voice)
    pub fn stop_voice(&mut self, voice_id: usize) -> Result<()> {
        if voice_id >= self.voices.len() {
            return Err(anyhow!("Voice ID {} out of range", voice_id));
        }

        let voice = &mut self.voices[voice_id];
        voice.envelope = EnvelopeState::new();
        // Silence all oscillators
        for osc in &voice.oscillators {
            osc.volume.set_property("volume", 0.0f64);
            osc.audiotestsrc.set_property("volume", 0.0f64);
        }
        voice.last_applied_volume = 0.0;
        voice.note = None;

        debug!("Stopped voice {voice_id}");
        Ok(())
    }

    /// Update frequency and amplitude for a voice with smooth transition
    pub fn update_voice_frequency(
        &mut self,
        voice_id: usize,
        frequency: f64,
        _amplitude: f64,
    ) -> Result<()> {
        if voice_id >= self.voices.len() {
            return Err(anyhow!("Voice ID {} out of range", voice_id));
        }

        let voice = &mut self.voices[voice_id];
        if voice.envelope.is_active() {
            voice.set_frequency(frequency);
            voice.envelope.target_frequency = frequency;
        }

        Ok(())
    }

    /// Find a free voice or steal the oldest releasing voice
    fn find_available_voice(&mut self) -> Option<usize> {
        // First, try to find an idle voice
        if let Some(idx) = self.voices.iter().position(|v| !v.envelope.is_active()) {
            return Some(idx);
        }

        // Then, try to find a releasing voice (steal it)
        if let Some(idx) = self.voices.iter().position(|v| v.envelope.is_releasing()) {
            return Some(idx);
        }

        // No available voices
        None
    }

    /// Play a note with proper voice allocation
    pub fn play_note(&mut self, note: MusicalNote) -> Result<()> {
        self.play_note_with_config(note, self.default_envelope)
    }

    /// Play a note with custom envelope
    pub fn play_note_with_config(
        &mut self,
        note: MusicalNote,
        envelope_config: EnvelopeConfig,
    ) -> Result<()> {
        // Check if note is already playing on a non-releasing voice
        if let Some(voice) = self
            .voices
            .iter_mut()
            .find(|v| v.note == Some(note) && v.envelope.is_active() && !v.envelope.is_releasing())
        {
            // Retrigger the existing voice
            voice.envelope.trigger(note.to_frequency(), envelope_config);
            return Ok(());
        }

        // Find an available voice
        if let Some(voice_idx) = self.find_available_voice() {
            let voice = &mut self.voices[voice_idx];
            let freq = note.to_frequency();

            voice.set_frequency(freq);
            voice.envelope.trigger(freq, envelope_config);
            voice.note = Some(note);

            let mut active_notes = self.active_notes.lock().unwrap();
            if !active_notes.contains(&note) {
                active_notes.push(note);
            }

            debug!("Playing note {note:?} ({freq:.2} Hz) on voice {voice_idx}");
        } else {
            warn!("No available voice for note {note:?}");
        }

        Ok(())
    }

    /// Release a specific note
    pub fn stop_note(&mut self, note: MusicalNote) -> Result<()> {
        for voice in &mut self.voices {
            if voice.note == Some(note) && !voice.envelope.is_releasing() {
                voice.envelope.release();
            }
        }

        // Note will be removed from active_notes when envelope finishes
        debug!("Released note {note:?}");
        Ok(())
    }

    /// Release all notes
    pub fn stop_all_notes(&mut self) -> Result<()> {
        for voice in &mut self.voices {
            if voice.envelope.is_active() {
                voice.envelope.release();
            }
        }
        debug!("Released all notes");
        Ok(())
    }

    /// Must be called regularly (every frame) to update envelopes
    pub fn update(&mut self) {
        let now = Instant::now();

        // Update envelopes at regular intervals
        if now.duration_since(self.last_envelope_update) >= self.update_interval {
            self.last_envelope_update = now;

            let mut finished_notes = Vec::new();

            for voice in &mut self.voices {
                voice.apply_envelope();

                // Track notes that finished their release phase
                if !voice.envelope.is_active() && voice.note.is_some() {
                    finished_notes.push(voice.note.take().unwrap());
                }
            }

            // Remove finished notes from active list
            if !finished_notes.is_empty() {
                let mut active_notes = self.active_notes.lock().unwrap();
                for note in finished_notes {
                    active_notes.retain(|&n| n != note);
                }
            }
        }

        self.last_update = now;
    }

    pub fn waveform(&self) -> AudioWaveform {
        self.current_waveform
    }

    pub fn master_volume(&self) -> f64 {
        *self.master_volume.lock().unwrap()
    }

    pub fn is_active(&self) -> bool {
        self.voices.iter().any(|v| v.envelope.is_active())
    }

    pub fn active_notes(&self) -> Vec<MusicalNote> {
        self.active_notes.lock().unwrap().clone()
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the current envelope level for a voice (useful for visualization)
    pub fn get_voice_level(&self, voice_id: usize) -> f32 {
        if voice_id < self.voices.len() {
            self.voices[voice_id].envelope.current_level
        } else {
            0.0
        }
    }

    /// Check if a specific voice is active
    pub fn is_voice_active(&self, voice_id: usize) -> bool {
        if voice_id < self.voices.len() {
            self.voices[voice_id].envelope.is_active()
        } else {
            false
        }
    }
}

impl Drop for AudioSynthManager {
    fn drop(&mut self) {
        info!("Shutting down audio synthesis pipeline");
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

/// Supported waveform types for audio synthesis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioWaveform {
    Sine,
    Square,
    Saw,
    Triangle,
}

/// Musical notes with their frequencies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MusicalNote {
    C4,
    CSharp4,
    D4,
    DSharp4,
    E4,
    F4,
    FSharp4,
    G4,
    GSharp4,
    A4,
    ASharp4,
    B4,
    C5,
}

impl MusicalNote {
    pub fn to_frequency(self) -> f64 {
        match self {
            MusicalNote::C4 => 261.63,
            MusicalNote::CSharp4 => 277.18,
            MusicalNote::D4 => 293.66,
            MusicalNote::DSharp4 => 311.13,
            MusicalNote::E4 => 329.63,
            MusicalNote::F4 => 349.23,
            MusicalNote::FSharp4 => 369.99,
            MusicalNote::G4 => 392.00,
            MusicalNote::GSharp4 => 415.30,
            MusicalNote::A4 => 440.00,
            MusicalNote::ASharp4 => 466.16,
            MusicalNote::B4 => 493.88,
            MusicalNote::C5 => 523.25,
        }
    }

    pub fn from_keyboard_number(num: u32) -> Option<Self> {
        match num {
            1 => Some(MusicalNote::C4),
            2 => Some(MusicalNote::D4),
            3 => Some(MusicalNote::E4),
            4 => Some(MusicalNote::F4),
            5 => Some(MusicalNote::G4),
            6 => Some(MusicalNote::A4),
            7 => Some(MusicalNote::B4),
            8 => Some(MusicalNote::C5),
            9 => Some(MusicalNote::CSharp4),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            MusicalNote::C4 => "C4",
            MusicalNote::CSharp4 => "C#4",
            MusicalNote::D4 => "D4",
            MusicalNote::DSharp4 => "D#4",
            MusicalNote::E4 => "E4",
            MusicalNote::F4 => "F4",
            MusicalNote::FSharp4 => "F#4",
            MusicalNote::G4 => "G4",
            MusicalNote::GSharp4 => "G#4",
            MusicalNote::A4 => "A4",
            MusicalNote::ASharp4 => "A#4",
            MusicalNote::B4 => "B4",
            MusicalNote::C5 => "C5",
        }
    }
}

/// Simple frequency-to-audio-data converter for visualization
pub struct AudioDataProvider {
    sample_count: usize,
}

impl AudioDataProvider {
    pub fn new() -> Self {
        Self { sample_count: 0 }
    }

    pub fn update(&mut self, _active_notes: &[MusicalNote], _master_volume: f64) {
        self.sample_count += 1;
    }

    pub fn generate_audio_data(
        &self,
        active_notes: &[MusicalNote],
        master_volume: f64,
    ) -> [[f32; 4]; 32] {
        let mut audio_data = [[0.0f32; 4]; 32];

        let note_mapping = [
            MusicalNote::C4,
            MusicalNote::D4,
            MusicalNote::E4,
            MusicalNote::F4,
            MusicalNote::G4,
            MusicalNote::A4,
            MusicalNote::B4,
            MusicalNote::C5,
            MusicalNote::CSharp4,
        ];

        if master_volume > 0.0 {
            for (note_index, &mapped_note) in note_mapping.iter().enumerate() {
                if active_notes.contains(&mapped_note) {
                    let positions_per_note = 128 / 9;
                    let start_pos = note_index * positions_per_note;
                    let amplitude = (master_volume * 0.9) as f32;

                    for i in 0..positions_per_note.min(10) {
                        let pos = start_pos + i;
                        if pos < 128 {
                            let array_index = pos / 4;
                            let component_index = pos % 4;

                            if array_index < 32 {
                                let distance_from_center =
                                    (i as f32 - positions_per_note as f32 / 2.0).abs();
                                let falloff = (1.0
                                    - distance_from_center / (positions_per_note as f32 / 2.0))
                                    .max(0.0);
                                let final_amplitude = amplitude * falloff;

                                audio_data[array_index][component_index] =
                                    (audio_data[array_index][component_index] + final_amplitude)
                                        .min(1.0);
                            }
                        }
                    }
                }
            }
        }

        audio_data
    }
}

impl Default for AudioDataProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AudioSynthUniform {
    pub note_frequencies: [[f32; 4]; 4],
    pub note_amplitudes: [[f32; 4]; 4],
    pub master_volume: f32,
    pub waveform_type: u32,
    pub active_note_count: u32,
    pub _padding: u32,
}

unsafe impl bytemuck::Pod for AudioSynthUniform {}
unsafe impl bytemuck::Zeroable for AudioSynthUniform {}

impl crate::UniformProvider for AudioSynthUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

impl Default for AudioSynthUniform {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioSynthUniform {
    pub fn new() -> Self {
        Self {
            note_frequencies: [[0.0; 4]; 4],
            note_amplitudes: [[0.0; 4]; 4],
            master_volume: 0.0,
            waveform_type: 0,
            active_note_count: 0,
            _padding: 0,
        }
    }

    pub fn update_from_synthesis(
        &mut self,
        active_notes: &[MusicalNote],
        master_volume: f64,
        waveform: AudioWaveform,
    ) {
        for vec4 in &mut self.note_frequencies {
            vec4.fill(0.0);
        }
        for vec4 in &mut self.note_amplitudes {
            vec4.fill(0.0);
        }

        self.master_volume = master_volume as f32;
        self.waveform_type = match waveform {
            AudioWaveform::Sine => 0,
            AudioWaveform::Square => 1,
            AudioWaveform::Saw => 2,
            AudioWaveform::Triangle => 3,
        };
        self.active_note_count = active_notes.len().min(16) as u32;

        for (i, note) in active_notes.iter().take(16).enumerate() {
            let vec4_index = i / 4;
            let component_index = i % 4;
            self.note_frequencies[vec4_index][component_index] = note.to_frequency() as f32;
            self.note_amplitudes[vec4_index][component_index] = master_volume as f32;
        }
    }
}

/// Voice state for the synthesis manager
#[derive(Clone)]
struct VoiceState {
    frequency: f32,
    amplitude: f32,
    active: bool,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self {
            frequency: 440.0,
            amplitude: 0.0,
            active: false,
        }
    }
}

// High-level synthesis manager that bridges GPU-computed parameters to GStreamer audio
pub struct SynthesisManager {
    audio_manager: Option<AudioSynthManager>,
    sample_rate: u32,
    last_update: Instant,
    synthesis_enabled: bool,
    voice_states: Vec<VoiceState>,
    envelope_config: EnvelopeConfig,
}

impl SynthesisManager {
    pub fn new() -> anyhow::Result<Self> {
        let audio_manager = match AudioSynthManager::new(Some(44100)) {
            Ok(manager) => Some(manager),
            Err(e) => {
                warn!("Failed to create GStreamer audio manager: {e}");
                None
            }
        };

        Ok(Self {
            audio_manager,
            sample_rate: 44100,
            last_update: Instant::now(),
            synthesis_enabled: false,
            voice_states: vec![VoiceState::default(); 16],
            envelope_config: EnvelopeConfig::default(),
        })
    }

    pub fn start_gpu_synthesis(&mut self) -> anyhow::Result<()> {
        if let Some(ref mut manager) = self.audio_manager {
            manager.start()?;
            self.synthesis_enabled = true;
        }
        Ok(())
    }

    pub fn stop_gpu_synthesis(&mut self) -> anyhow::Result<()> {
        if let Some(ref mut manager) = self.audio_manager {
            manager.stop()?;
            self.synthesis_enabled = false;
        }
        Ok(())
    }

    /// Set the global envelope configuration
    pub fn set_envelope(&mut self, config: EnvelopeConfig) {
        self.envelope_config = config;
        if let Some(ref mut manager) = self.audio_manager {
            manager.set_envelope_config(config);
        }
    }

    /// Set envelope from ADSR values
    pub fn set_adsr(&mut self, attack: f32, decay: f32, sustain: f32, release: f32) {
        self.set_envelope(EnvelopeConfig {
            attack_time: attack,
            decay_time: decay,
            sustain_level: sustain,
            release_time: release,
        });
    }

    pub fn update_frequency(&mut self, frequency: f32) {
        self.set_voice(0, frequency, 0.3, true);
    }

    pub fn update_waveform(&mut self, waveform_type: u32) {
        if let Some(ref mut manager) = self.audio_manager {
            let gst_waveform = match waveform_type {
                0 => AudioWaveform::Sine,
                1 => AudioWaveform::Square,
                2 => AudioWaveform::Saw,
                3 => AudioWaveform::Triangle,
                _ => AudioWaveform::Sine,
            };
            let _ = manager.set_waveform(gst_waveform);
        }
    }

    pub fn set_voice(&mut self, voice_id: usize, frequency: f32, amplitude: f32, active: bool) {
        if voice_id >= self.voice_states.len() {
            return;
        }

        let prev_state = &self.voice_states[voice_id];
        let was_active = prev_state.active && prev_state.amplitude > 0.001;
        let should_be_active = active && amplitude > 0.001;

        if let Some(ref mut manager) = self.audio_manager {
            if should_be_active && !was_active {
                // Note on: start with envelope
                let _ = manager.play_frequency_with_config(
                    frequency as f64,
                    voice_id,
                    self.envelope_config,
                );
            } else if !should_be_active && was_active {
                // Note off: release (don't stop immediately)
                let _ = manager.release_voice(voice_id);
            } else if should_be_active && was_active {
                // Update frequency if changed significantly
                if (frequency - prev_state.frequency).abs() > 0.1 {
                    let _ = manager.update_voice_frequency(
                        voice_id,
                        frequency as f64,
                        amplitude as f64,
                    );
                }
            }
        }

        // Update state
        self.voice_states[voice_id] = VoiceState {
            frequency,
            amplitude,
            active: should_be_active,
        };
    }

    /// Set master volume
    pub fn set_master_volume(&mut self, volume: f64) {
        if let Some(ref mut manager) = self.audio_manager {
            let _ = manager.set_master_volume(volume);
        }
    }

    /// Get the current envelope level for a voice (for visualization)
    pub fn get_voice_level(&self, voice_id: usize) -> f32 {
        if let Some(ref manager) = self.audio_manager {
            manager.get_voice_level(voice_id)
        } else {
            0.0
        }
    }

    pub fn stream_gpu_samples(&mut self, _samples: &[f32]) {
        // This method is kept for compatibility (my impl on CPAL approach) but not used in GStreamer implementation
    }

    /// Must be called every frame to update envelopes
    pub fn update(&mut self) {
        self.last_update = Instant::now();
        if let Some(ref mut manager) = self.audio_manager {
            manager.update();
        }
    }

    pub fn get_buffer_info(&self) -> Option<(usize, bool)> {
        Some((0, self.synthesis_enabled))
    }

    pub fn is_gpu_synthesis_enabled(&self) -> bool {
        self.synthesis_enabled
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SynthesisUniform {
    pub note_frequencies: [[f32; 4]; 4],
    pub note_amplitudes: [[f32; 4]; 4],
    pub master_volume: f32,
    pub waveform_type: u32,
    pub active_note_count: u32,
}

unsafe impl bytemuck::Pod for SynthesisUniform {}
unsafe impl bytemuck::Zeroable for SynthesisUniform {}

impl crate::UniformProvider for SynthesisUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

impl Default for SynthesisUniform {
    fn default() -> Self {
        Self::new()
    }
}

impl SynthesisUniform {
    pub fn new() -> Self {
        Self {
            note_frequencies: [[0.0; 4]; 4],
            note_amplitudes: [[0.0; 4]; 4],
            master_volume: 0.3,
            waveform_type: 0,
            active_note_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthesisWaveform {
    Sine,
    Square,
    Saw,
    Triangle,
}

impl SynthesisWaveform {
    pub fn to_u32(self) -> u32 {
        match self {
            SynthesisWaveform::Sine => 0,
            SynthesisWaveform::Square => 1,
            SynthesisWaveform::Saw => 2,
            SynthesisWaveform::Triangle => 3,
        }
    }
}
