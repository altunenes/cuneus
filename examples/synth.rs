use cuneus::audio::{EnvelopeConfig, SynthesisManager};
use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

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
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    key_states: [[f32; 4]; 3],
    key_decay: [[f32; 4]; 3]}
}

struct SynthManager {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SynthParams,
    gpu_synthesis: Option<SynthesisManager>,
    // Track which keys are currently held down
    keys_held: [bool; 9]}

impl SynthManager {
    fn set_key_state(&mut self, key_index: usize, state: f32) {
        if key_index < 9 {
            let vec_idx = key_index / 4;
            let comp_idx = key_index % 4;
            self.current_params.key_states[vec_idx][comp_idx] = state;
        }
    }

    fn set_key_decay(&mut self, key_index: usize, decay: f32) {
        if key_index < 9 {
            let vec_idx = key_index / 4;
            let comp_idx = key_index % 4;
            self.current_params.key_decay[vec_idx][comp_idx] = decay;
        }
    }

    fn get_note_frequency(&self, note_index: usize) -> f32 {
        let notes = [
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33,
        ];
        let octave_multiplier = 2.0_f32.powf(self.current_params.octave - 4.0);
        notes[note_index] * octave_multiplier
    }

    fn update_envelope_config(&mut self) {
        if let Some(ref mut synth) = self.gpu_synthesis {
            synth.set_adsr(
                self.current_params.attack_time,
                self.current_params.decay_time,
                self.current_params.sustain_level,
                self.current_params.release_time,
            );
        }
    }
}

impl ShaderManager for SynthManager {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        let initial_params = SynthParams {
            tempo: 120.0,
            waveform_type: 1,
            octave: 4.0,
            volume: 0.7,
            beat_enabled: 0,
            reverb_mix: 0.15,
            delay_time: 0.3,
            delay_feedback: 0.3,
            filter_cutoff: 0.9,
            filter_resonance: 0.1,
            distortion_amount: 0.0,
            chorus_rate: 2.0,
            chorus_depth: 0.1,
            attack_time: 0.02,
            decay_time: 0.15,
            sustain_level: 0.7,
            release_time: 0.4,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
            key_states: [[0.0; 4]; 3],
            key_decay: [[0.0; 4]; 3]};

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<SynthParams>()
            .with_audio(2048)
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Synth Unified")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/synth.wgsl", config);

        compute_shader.set_custom_params(initial_params, &core.queue);

        // Initialize audio synthesis with envelope configuration
        let gpu_synthesis = match SynthesisManager::new() {
            Ok(mut synth) => {
                // Set initial envelope config
                synth.set_envelope(EnvelopeConfig {
                    attack_time: initial_params.attack_time,
                    decay_time: initial_params.decay_time,
                    sustain_level: initial_params.sustain_level,
                    release_time: initial_params.release_time});

                if let Err(_e) = synth.start_gpu_synthesis() {
                    None
                } else {
                    Some(synth)
                }
            }
            Err(_e) => None};

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            gpu_synthesis,
            keys_held: [false; 9]}
    }

    fn update(&mut self, core: &Core) {

        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Update GPU shader params for visualization
        // The audio backend handles the actual envelope - we just need to track key states for visuals
        let mut needs_update = false;
        for i in 0..9 {
            if self.keys_held[i] {
                // Key is held - show full visualization
                self.set_key_state(i, 1.0);
                self.set_key_decay(i, 1.0);
                needs_update = true;
            } else {
                // Key released - get actual envelope level from audio backend for smooth visual
                let current_decay = self.current_params.key_decay[i / 4][i % 4];
                if current_decay > 0.01 {
                    // Smooth visual fade (the audio backend handles actual audio envelope)
                    let new_decay = current_decay * 0.92;
                    self.set_key_decay(i, new_decay);
                    if new_decay < 0.01 {
                        self.set_key_state(i, 0.0);
                    }
                    needs_update = true;
                }
            }
        }

        if needs_update {
            self.compute_shader
                .set_custom_params(self.current_params, &core.queue);
        }

        // Update audio synthesis
        if let Some(ref mut synth) = self.gpu_synthesis {
            // Update waveform
            synth.update_waveform(self.current_params.waveform_type);
            synth.set_master_volume(self.current_params.volume as f64);

            // The audio manager handles envelopes internally
            // We just need to call update() to process envelope states
            synth.update();
        }
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = false;
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Cuneus GPU Synth")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("About")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("🎹 GPU-powered polyphonic synthesizer");
                                ui.label("• Press keys 1-9 for musical notes");
                                ui.label("• Smooth ADSR envelopes prevent clicks");
                                ui.label("• Real-time effects processing");
                                ui.label("• Visual feedback with spectrum bars");
                            });

                        egui::CollapsingHeader::new("Playback")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Keys:");
                                    ui.label("1-9 for C D E F G A B C D");
                                });

                                let mut beat_enabled = params.beat_enabled > 0;
                                if ui.checkbox(&mut beat_enabled, "Background Beat").changed() {
                                    params.beat_enabled = if beat_enabled { 1 } else { 0 };
                                    changed = true;
                                }

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.tempo, 60.0..=180.0)
                                            .text("Tempo"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.octave, 2.0..=7.0)
                                            .text("Octave"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.volume, 0.0..=1.0)
                                            .text("Master Volume"),
                                    )
                                    .changed();

                                ui.horizontal(|ui| {
                                    ui.label("Waveform:");
                                    let waveform_names = ["Sin", "Saw", "Sqr", "Tri", "Nse"];
                                    for (i, name) in waveform_names.iter().enumerate() {
                                        let selected = params.waveform_type == i as u32;
                                        if ui.selectable_label(selected, *name).clicked() {
                                            params.waveform_type = i as u32;
                                            changed = true;
                                        }
                                    }
                                });
                            });

                        egui::CollapsingHeader::new("Envelope (ADSR)")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Controls click-free sound transitions");
                                ui.separator();

                                let attack_changed = ui
                                    .add(
                                        egui::Slider::new(&mut params.attack_time, 0.001..=0.5)
                                            .logarithmic(true)
                                            .text("Attack")
                                            .suffix("s"),
                                    )
                                    .changed();
                                let decay_changed = ui
                                    .add(
                                        egui::Slider::new(&mut params.decay_time, 0.01..=1.0)
                                            .logarithmic(true)
                                            .text("Decay")
                                            .suffix("s"),
                                    )
                                    .changed();
                                let sustain_changed = ui
                                    .add(
                                        egui::Slider::new(&mut params.sustain_level, 0.0..=1.0)
                                            .text("Sustain"),
                                    )
                                    .changed();
                                let release_changed = ui
                                    .add(
                                        egui::Slider::new(&mut params.release_time, 0.01..=2.0)
                                            .logarithmic(true)
                                            .text("Release")
                                            .suffix("s"),
                                    )
                                    .changed();

                                if attack_changed
                                    || decay_changed
                                    || sustain_changed
                                    || release_changed
                                {
                                    changed = true;
                                }

                                ui.separator();
                                if ui.small_button("Piano Preset").clicked() {
                                    params.attack_time = 0.01;
                                    params.decay_time = 0.3;
                                    params.sustain_level = 0.5;
                                    params.release_time = 0.8;
                                    changed = true;
                                }
                                ui.horizontal(|ui| {
                                    if ui.small_button("Pad Preset").clicked() {
                                        params.attack_time = 0.2;
                                        params.decay_time = 0.5;
                                        params.sustain_level = 0.8;
                                        params.release_time = 1.5;
                                        changed = true;
                                    }
                                    if ui.small_button("Pluck Preset").clicked() {
                                        params.attack_time = 0.005;
                                        params.decay_time = 0.1;
                                        params.sustain_level = 0.3;
                                        params.release_time = 0.2;
                                        changed = true;
                                    }
                                });
                            });

                        egui::CollapsingHeader::new("Filter")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.filter_cutoff, 0.0..=1.0)
                                            .text("Cutoff"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.filter_resonance, 0.0..=0.9)
                                            .text("Resonance"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Effects")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.reverb_mix, 0.0..=0.8)
                                            .text("Reverb"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.delay_time, 0.01..=1.0)
                                            .text("Delay Time"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.delay_feedback, 0.0..=0.8)
                                            .text("Delay Feedback"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.distortion_amount, 0.0..=0.9)
                                            .text("Distortion"),
                                    )
                                    .changed();
                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.chorus_rate, 0.1..=10.0)
                                            .text("Chorus Rate"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.chorus_depth, 0.0..=0.5)
                                            .text("Chorus Depth"),
                                    )
                                    .changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
            // Update envelope config in audio backend
            self.update_envelope_config();
        }

        self.base.apply_control_request(controls_request);

        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self
            .base
            .egui_state
            .on_window_event(core.window(), event)
            .consumed
        {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let winit::keyboard::Key::Character(ref s) = event.logical_key {
                if let Some(key_index) = s.chars().next().and_then(|c| c.to_digit(10)) {
                    if (1..=9).contains(&key_index) {
                        let index = (key_index - 1) as usize;
                        let frequency = self.get_note_frequency(index);

                        if event.state == winit::event::ElementState::Pressed {
                            // Only trigger if not already held (prevent retriggering on key repeat)
                            if !self.keys_held[index] {
                                self.keys_held[index] = true;

                                // Update visual state
                                self.set_key_state(index, 1.0);
                                self.set_key_decay(index, 1.0);
                                self.compute_shader
                                    .set_custom_params(self.current_params, &core.queue);

                                // Trigger note in audio backend (handles envelope automatically)
                                if let Some(ref mut synth) = self.gpu_synthesis {
                                    let amplitude = self.current_params.volume * 0.4;
                                    synth.set_voice(index, frequency, amplitude, true);
                                }
                            }
                        } else if event.state == winit::event::ElementState::Released {
                            self.keys_held[index] = false;

                            // Release note in audio backend (will fade out with envelope)
                            if let Some(ref mut synth) = self.gpu_synthesis {
                                synth.set_voice(index, frequency, 0.0, false);
                            }

                            // Visual state will fade in update()
                        }
                        return true;
                    }
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
