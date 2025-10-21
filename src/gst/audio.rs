use anyhow::{anyhow, Result};
use gst::glib::ControlFlow;
use gst::prelude::*;
use gstreamer as gst;
use log::{debug, info, warn};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Individual audio voice for polyphonic synthesis
/// Based: https://gstreamer.freedesktop.org/documentation/audiotestsrc/index.html?gi-language=c
/// Sample rate: 44100
struct AudioVoice {
    /// The audiotestsrc element for this 'voice'
    audiotestsrc: gst::Element,
    /// The volume control element for this voice
    volume: gst::Element,
    /// Current frequency being generated
    frequency: f64,
    /// Current volume (0.0 to 1.0)
    volume_level: f64,
    /// Whether this voice is active
    is_active: bool,
    /// Musical note this voice is playing
    note: Option<MusicalNote>,
}

/// Audio synthesis manager that can generate multiple tones simultaneously using GStreamer
pub struct AudioSynthManager {
    /// The GStreamer pipeline for audio generation
    pipeline: gst::Pipeline,
    /// The audio mixer element
    _audiomixer: gst::Element,
    /// Multiple voices for polyphonic synthesis
    voices: Vec<AudioVoice>,
    /// Current waveform type
    current_waveform: AudioWaveform,
    /// Master volume (0.0 to 1.0)
    master_volume: Arc<Mutex<f64>>,
    /// Sample rate for audio generation
    sample_rate: u32,
    /// Last update time
    last_update: Instant,
    /// Currently playing notes
    active_notes: Arc<Mutex<Vec<MusicalNote>>>,
}

impl AudioSynthManager {
    pub fn new(sample_rate: Option<u32>) -> Result<Self> {
        let sample_rate = sample_rate.unwrap_or(44100);
        // Support up to 8 simultaneous notes
        let max_voices = 8;

        info!(
            "Creating polyphonic audio synthesis manager with {} voices at {} Hz",
            max_voices, sample_rate
        );

        let pipeline = gst::Pipeline::new();

        // Create audio mixer for combining multiple voices
        let audiomixer = gst::ElementFactory::make("audiomixer")
            .name("audio_mixer")
            .build()
            .map_err(|_| anyhow!("Failed to create audiomixer element"))?;

        // Create final audio conversion and output chain
        let final_convert = gst::ElementFactory::make("audioconvert")
            .name("final_convert")
            .build()
            .map_err(|_| anyhow!("Failed to create final audioconvert element"))?;

        let final_resample = gst::ElementFactory::make("audioresample")
            .name("final_resample")
            .build()
            .map_err(|_| anyhow!("Failed to create final audioresample element"))?;

        let master_volume = gst::ElementFactory::make("volume")
            .name("master_volume")
            .property("volume", 0.3f64)
            .build()
            .map_err(|_| anyhow!("Failed to create master volume element"))?;

        let audiosink = gst::ElementFactory::make("autoaudiosink")
            .name("audio_sink")
            .build()
            .map_err(|_| anyhow!("Failed to create autoaudiosink element"))?;

        // A mixer and output chain to pipeline
        pipeline
            .add_many(&[
                &audiomixer,
                &final_convert,
                &final_resample,
                &master_volume,
                &audiosink,
            ])
            .map_err(|_| anyhow!("Failed to add output elements to pipeline"))?;

        gst::Element::link_many(&[
            &audiomixer,
            &final_convert,
            &final_resample,
            &master_volume,
            &audiosink,
        ])
        .map_err(|_| anyhow!("Failed to link output elements"))?;

        // Create voices
        let mut voices = Vec::new();
        for i in 0..max_voices {
            let voice = Self::create_voice(&pipeline, &audiomixer, i, sample_rate)?;
            voices.push(voice);
        }

        // bus watch for error handling stuff
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

        let audio_synth = Self {
            pipeline,
            _audiomixer: audiomixer,
            voices,
            current_waveform: AudioWaveform::Sine,
            master_volume: Arc::new(Mutex::new(0.3)),
            sample_rate,
            last_update: Instant::now(),
            active_notes: Arc::new(Mutex::new(Vec::new())),
        };

        info!("Polyphonic audio synthesis manager created successfully");
        Ok(audio_synth)
    }

    fn create_voice(
        pipeline: &gst::Pipeline,
        mixer: &gst::Element,
        voice_id: usize,
        _sample_rate: u32,
    ) -> Result<AudioVoice> {
        // Create audiotestsrc for this voice
        let audiotestsrc = gst::ElementFactory::make("audiotestsrc")
            .name(&format!("voice_{}_source", voice_id))
            .property("freq", 440.0f64)
            .property("volume", 0.0f64) // Start silent
            .property("samplesperbuffer", 512i32)
            .property("is-live", true)
            .build()
            .map_err(|_| anyhow!("Failed to create audiotestsrc for voice {}", voice_id))?;

        //default waveform:
        audiotestsrc.set_property_from_str("wave", "sine");

        // lets create voice-specific elements
        let audioconvert = gst::ElementFactory::make("audioconvert")
            .name(&format!("voice_{}_convert", voice_id))
            .build()
            .map_err(|_| anyhow!("Failed to create audioconvert for voice {}", voice_id))?;

        let audioresample = gst::ElementFactory::make("audioresample")
            .name(&format!("voice_{}_resample", voice_id))
            .build()
            .map_err(|_| anyhow!("Failed to create audioresample for voice {}", voice_id))?;

        let volume = gst::ElementFactory::make("volume")
            .name(&format!("voice_{}_volume", voice_id))
            .property("volume", 0.0f64)
            .build()
            .map_err(|_| anyhow!("Failed to create volume for voice {}", voice_id))?;

        //voice elements to our pipeline...
        pipeline
            .add_many(&[&audiotestsrc, &audioconvert, &audioresample, &volume])
            .map_err(|_| anyhow!("Failed to add voice {} elements to pipeline", voice_id))?;

        // link:
        gst::Element::link_many(&[&audiotestsrc, &audioconvert, &audioresample, &volume])
            .map_err(|_| anyhow!("Failed to link voice {} elements", voice_id))?;

        // Connect to mixer
        volume
            .link(mixer)
            .map_err(|_| anyhow!("Failed to link voice {} to mixer", voice_id))?;

        Ok(AudioVoice {
            audiotestsrc,
            volume,
            frequency: 440.0,
            volume_level: 0.0,
            is_active: false,
            note: None,
        })
    }

    /// audio generation
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

    // Stop all voices
    pub fn stop(&mut self) -> Result<()> {
        info!("Stopping audio synthesis");
        for voice in &mut self.voices {
            voice.volume.set_property("volume", 0.0f64);
            voice.is_active = false;
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

        // Find master volume element
        if let Some(master_vol) = self.pipeline.by_name("master_volume") {
            master_vol.set_property("volume", clamped_volume);
        }

        debug!("Set master volume to {:.3}", clamped_volume);
        Ok(())
    }

    /// Set the waveform type for all voices
    pub fn set_waveform(&mut self, wave_type: AudioWaveform) -> Result<()> {
        let wave_str = match wave_type {
            AudioWaveform::Sine => "sine",
            AudioWaveform::Square => "square",
            AudioWaveform::Saw => "saw",
            AudioWaveform::Triangle => "triangle",
        };

        self.current_waveform = wave_type;

        // Update all voices
        for voice in &mut self.voices {
            voice.audiotestsrc.set_property_from_str("wave", wave_str);
        }

        debug!("Set waveform to {:?}", wave_type);
        Ok(())
    }

    /// Play a frequency (polyphonic - can play multiple frequencies simultaneously)
    /// This allows arbitrary frequencies, not just predefined musical notes
    pub fn play_frequency(&mut self, frequency: f64, voice_id: usize) -> Result<()> {
        if voice_id >= self.voices.len() {
            return Err(anyhow!("Voice ID {} out of range", voice_id));
        }

        let voice = &mut self.voices[voice_id];
        let volume_level = 0.4; // Individual voice volume

        // Configure voice with arbitrary frequency
        voice.audiotestsrc.set_property("freq", frequency);
        voice.audiotestsrc.set_property("volume", volume_level);
        voice.volume.set_property("volume", volume_level);
        voice.frequency = frequency;
        voice.volume_level = volume_level;
        voice.is_active = true;
        // Not tied to a specific musical note
        voice.note = None;

        info!(
            "Playing frequency {:.2} Hz on voice {}",
            frequency, voice_id
        );
        Ok(())
    }

    /// Stop a specific voice by ID
    pub fn stop_voice(&mut self, voice_id: usize) -> Result<()> {
        if voice_id >= self.voices.len() {
            return Err(anyhow!("Voice ID {} out of range", voice_id));
        }

        let voice = &mut self.voices[voice_id];
        voice.audiotestsrc.set_property("volume", 0.0f64);
        voice.volume.set_property("volume", 0.0f64);
        voice.is_active = false;
        voice.note = None;

        info!("Stopped voice {}", voice_id);
        Ok(())
    }

    /// Update frequency and amplitude for an already active voice
    pub fn update_voice_frequency(
        &mut self,
        voice_id: usize,
        frequency: f64,
        amplitude: f64,
    ) -> Result<()> {
        if voice_id >= self.voices.len() {
            return Err(anyhow!("Voice ID {} out of range", voice_id));
        }

        let voice = &mut self.voices[voice_id];
        if voice.is_active {
            voice.audiotestsrc.set_property("freq", frequency);
            voice.volume.set_property("volume", amplitude);
            voice.frequency = frequency;
            voice.volume_level = amplitude;
        }

        Ok(())
    }

    /// Play a note (polyphonic - can play multiple notes simultaneously)
    /// Little bit tricky, I tried to up to 8 notes at once
    pub fn play_note(&mut self, note: MusicalNote) -> Result<()> {
        // Check if note is already playing
        let mut active_notes = self.active_notes.lock().unwrap();
        if active_notes.contains(&note) {
            return Ok(()); // Note already playing
        }

        // Find an available voice
        if let Some(voice) = self.voices.iter_mut().find(|v| !v.is_active) {
            let freq = note.to_frequency();
            let volume_level = 0.4; // Individual voice volume

            // cfg voice
            voice.audiotestsrc.set_property("freq", freq);
            voice.audiotestsrc.set_property("volume", volume_level);
            voice.volume.set_property("volume", volume_level);
            voice.frequency = freq;
            voice.volume_level = volume_level;
            voice.is_active = true;
            voice.note = Some(note);

            active_notes.push(note);
            info!("Playing note {:?} ({:.2} Hz) on voice", note, freq);
        } else {
            warn!("No available voice for note {:?}", note);
        }

        Ok(())
    }

    /// Stop a specific note. this more likely for testing purposes for myself
    pub fn stop_note(&mut self, note: MusicalNote) -> Result<()> {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.note == Some(note)) {
            voice.audiotestsrc.set_property("volume", 0.0f64);
            voice.volume.set_property("volume", 0.0f64);
            voice.is_active = false;
            voice.note = None;

            // Remove from active notes
            let mut active_notes = self.active_notes.lock().unwrap();
            active_notes.retain(|&n| n != note);

            info!("Stopped note {:?}", note);
        }
        Ok(())
    }

    pub fn stop_all_notes(&mut self) -> Result<()> {
        for voice in &mut self.voices {
            voice.audiotestsrc.set_property("volume", 0.0f64);
            voice.volume.set_property("volume", 0.0f64);
            voice.is_active = false;
            voice.note = None;
        }
        self.active_notes.lock().unwrap().clear();
        info!("Stopped all notes");
        Ok(())
    }

    /// Get current waveform
    pub fn waveform(&self) -> AudioWaveform {
        self.current_waveform
    }

    /// Get master volume
    pub fn master_volume(&self) -> f64 {
        *self.master_volume.lock().unwrap()
    }

    /// Check if any notes are playing
    pub fn is_active(&self) -> bool {
        !self.active_notes.lock().unwrap().is_empty()
    }

    /// Get currently playing notes
    pub fn active_notes(&self) -> Vec<MusicalNote> {
        self.active_notes.lock().unwrap().clone()
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Update method to be called in the main loop if needed
    pub fn update(&mut self) {
        self.last_update = Instant::now();
        //maybe per-frame updates?
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
/// I took these from here:
/// https://www.liutaiomottola.com/formulae/freqtab.htm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MusicalNote {
    C4,      // 261.63 Hz
    CSharp4, // 277.18 Hz
    D4,      // 293.66 Hz
    DSharp4, // 311.13 Hz
    E4,      // 329.63 Hz
    F4,      // 349.23 Hz
    FSharp4, // 369.99 Hz
    G4,      // 392.00 Hz
    GSharp4, // 415.30 Hz
    A4,      // 440.00 Hz
    ASharp4, // 466.16 Hz
    B4,      // 493.88 Hz
    C5,      // 523.25 Hz
}

impl MusicalNote {
    /// Convert musical note to frequency in Hz
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

    /// Get note from keyboard number (1-9 maps to different notes)
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

    /// Get the name of the note as a string
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

    /// Update with current audio synthesis parameters
    pub fn update(&mut self, _active_notes: &[MusicalNote], _master_volume: f64) {
        self.sample_count += 1;
    }

    /// Generate audio data array for visualization based on active notes
    /// Maps the 9 keyboard notes (1-9) directly to visualization data
    pub fn generate_audio_data(
        &self,
        active_notes: &[MusicalNote],
        master_volume: f64,
    ) -> [[f32; 4]; 32] {
        let mut audio_data = [[0.0f32; 4]; 32];

        // Map notes 1-9 to positions across the audio data array
        // We'll use the first 9 positions to represent keys 1-9
        let note_mapping = [
            MusicalNote::C4,      // Key 1
            MusicalNote::D4,      // Key 2
            MusicalNote::E4,      // Key 3
            MusicalNote::F4,      // Key 4
            MusicalNote::G4,      // Key 5
            MusicalNote::A4,      // Key 6
            MusicalNote::B4,      // Key 7
            MusicalNote::C5,      // Key 8
            MusicalNote::CSharp4, // Key 9
        ];

        if master_volume > 0.0 {
            // Check each of the 9 notes and set corresponding data
            for (note_index, &mapped_note) in note_mapping.iter().enumerate() {
                if active_notes.contains(&mapped_note) {
                    // Calculate the position in the audio_data array for this note
                    // Spread 9 notes across 32*4=128 positions
                    let positions_per_note = 128 / 9; // ~14 positions per note
                    let start_pos = note_index * positions_per_note;

                    let amplitude = (master_volume * 0.9) as f32;

                    // Fill multiple positions for this note to create a "region"
                    for i in 0..positions_per_note.min(10) {
                        let pos = start_pos + i;
                        if pos < 128 {
                            let array_index = pos / 4;
                            let component_index = pos % 4;

                            if array_index < 32 {
                                // Create a peak with falloff from center
                                let distance_from_center =
                                    (i as f32 - positions_per_note as f32 / 2.0).abs();
                                let falloff = (1.0
                                    - distance_from_center / (positions_per_note as f32 / 2.0))
                                    .max(0.0);
                                let final_amplitude = amplitude * falloff;

                                match component_index {
                                    0 => {
                                        audio_data[array_index][0] =
                                            (audio_data[array_index][0] + final_amplitude).min(1.0)
                                    }
                                    1 => {
                                        audio_data[array_index][1] =
                                            (audio_data[array_index][1] + final_amplitude).min(1.0)
                                    }
                                    2 => {
                                        audio_data[array_index][2] =
                                            (audio_data[array_index][2] + final_amplitude).min(1.0)
                                    }
                                    3 => {
                                        audio_data[array_index][3] =
                                            (audio_data[array_index][3] + final_amplitude).min(1.0)
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        audio_data
    }
}

/// Dedicated uniform structure for audio synthesis data (separate from audio visualization)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AudioSynthUniform {
    /// Active note frequencies (up to 16 simultaneous notes) - stored as vec4s for alignment
    pub note_frequencies: [[f32; 4]; 4], // 16 frequencies in 4 vec4s
    /// Individual note amplitudes - stored as vec4s for alignment  
    pub note_amplitudes: [[f32; 4]; 4], // 16 amplitudes in 4 vec4s
    /// Master volume level
    pub master_volume: f32,
    /// Current waveform type (0=sine, 1=square, 2=saw, 3=triangle)
    pub waveform_type: u32,
    /// Number of currently active notes
    pub active_note_count: u32,
    /// Padding for proper alignment
    pub _padding: u32,
}

// Safe to transmute to bytes for GPU upload
unsafe impl bytemuck::Pod for AudioSynthUniform {}
unsafe impl bytemuck::Zeroable for AudioSynthUniform {}

impl crate::UniformProvider for AudioSynthUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
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

    /// Update with current synthesis state
    pub fn update_from_synthesis(
        &mut self,
        active_notes: &[MusicalNote],
        master_volume: f64,
        waveform: AudioWaveform,
    ) {
        // Clear previous data
        for vec4 in &mut self.note_frequencies {
            vec4.fill(0.0);
        }
        for vec4 in &mut self.note_amplitudes {
            vec4.fill(0.0);
        }

        // Update current state
        self.master_volume = master_volume as f32;
        self.waveform_type = match waveform {
            AudioWaveform::Sine => 0,
            AudioWaveform::Square => 1,
            AudioWaveform::Saw => 2,
            AudioWaveform::Triangle => 3,
        };
        self.active_note_count = active_notes.len().min(16) as u32;

        // Copy active note data (map 16 notes to 4 vec4s)
        for (i, note) in active_notes.iter().take(16).enumerate() {
            let vec4_index = i / 4;
            let component_index = i % 4;
            self.note_frequencies[vec4_index][component_index] = note.to_frequency() as f32;
            self.note_amplitudes[vec4_index][component_index] = master_volume as f32;
        }
    }
}

impl Default for AudioDataProvider {
    fn default() -> Self {
        Self::new()
    }
}

// High-level synthesis manager that bridges GPU-computed parameters to GStreamer audio
pub struct SynthesisManager {
    audio_manager: Option<AudioSynthManager>,
    sample_rate: u32,
    last_update: std::time::Instant,
    synthesis_enabled: bool,
    // Track which voices are active
    active_voices: Vec<bool>,
}

impl SynthesisManager {
    pub fn new() -> anyhow::Result<Self> {
        let audio_manager = match AudioSynthManager::new(Some(44100)) {
            Ok(manager) => Some(manager),
            Err(e) => {
                eprintln!("Failed to create GStreamer audio manager: {}", e);
                None
            }
        };

        Ok(Self {
            audio_manager,
            sample_rate: 44100,
            last_update: std::time::Instant::now(),
            synthesis_enabled: false,
            active_voices: vec![false; 9], // Track 9 voices
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
        if voice_id >= self.active_voices.len() {
            return;
        }

        let should_be_active = active && amplitude > 0.001;

        if let Some(ref mut manager) = self.audio_manager {
            if should_be_active && !self.active_voices[voice_id] {
                // Start the voice
                let _ = manager.play_frequency(frequency as f64, voice_id);
                self.active_voices[voice_id] = true;
            } else if !should_be_active && self.active_voices[voice_id] {
                // Stop the voice
                let _ = manager.stop_voice(voice_id);
                self.active_voices[voice_id] = false;
            } else if should_be_active && self.active_voices[voice_id] {
                // Voice is already active, but update frequency and amplitude
                let _ =
                    manager.update_voice_frequency(voice_id, frequency as f64, amplitude as f64);
            }
        }
    }

    pub fn stream_gpu_samples(&mut self, _samples: &[f32]) {
        // This method is kept for compatibility (my impl on CPAL approach) but not used in GStreamer implementation
    }

    pub fn update(&mut self) {
        self.last_update = std::time::Instant::now();
        // Update GStreamer manager if needed
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
