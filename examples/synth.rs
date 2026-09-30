use cuneus::audio::PcmStreamManager;
use cuneus::compute::*;
use cuneus::prelude::*;
use log::error;
use std::ops::RangeInclusive;
use winit::keyboard::{KeyCode, PhysicalKey};

const MAX_SAMPLES_PER_FRAME: u32 = 1024;
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
        _pad1: f32,
        key_on: [[u32; 4]; 4],
        key_off: [[u32; 4]; 4],
    }
}

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
}

impl SynthManager {
    /// First sample not generated yet, so a note always starts with its full attack; 0 means none
    fn next_sample(&self) -> u32 {
        let written = self.pcm_stream.as_ref().map_or(0, |s| s.samples_written());
        ((written + self.last_samples_generated as u64) as u32).max(1)
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
            _pad1: 0.0,
            key_on: [[0; 4]; 4],
            key_off: [[0; 4]; 4],
        };

        let audio_buffer_size = (MAX_SAMPLES_PER_FRAME * 2) as usize;
        // lets add a new buffer called dsp. Persistent DSP state: filter integrators, delay/chorus/reverb lines and the
        // scope ring. Must be >= the shader's layout (~62,110 floats); 65,536 leaves headroom.
        let dsp_buffer_size = (65536 * std::mem::size_of::<f32>()) as u64;

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<SynthParams>()
            .with_audio(audio_buffer_size)
            .with_storage_buffer(StorageBufferSpec::new("dsp", dsp_buffer_size))
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

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

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

        if changed {
            // Preserve audio fields that are managed by update()
            params.sample_offset = self.current_params.sample_offset;
            params.samples_to_generate = self.current_params.samples_to_generate;
            params.sample_rate = self.current_params.sample_rate;
            params.play_sample = self.current_params.play_sample;
            params.key_on = self.current_params.key_on;
            params.key_off = self.current_params.key_off;
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
