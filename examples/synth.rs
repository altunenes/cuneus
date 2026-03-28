use cuneus::audio::PcmStreamManager;
use cuneus::compute::*;
use cuneus::prelude::*;
use log::error;

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
        key_states: [[f32; 4]; 3],
        key_decay: [[f32; 4]; 3],
    }
}

struct SynthManager {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SynthParams,
    pcm_stream: Option<PcmStreamManager>,
    keys_held: [bool; 9],
    audio_start: std::time::Instant,
    last_samples_generated: u32,
}

impl SynthManager {
    /// key_states stores note-on time (>0 = pressed at this time, 0 = never pressed)
    fn set_key_press_time(&mut self, key_index: usize, time: f32) {
        if key_index < 9 {
            self.current_params.key_states[key_index / 4][key_index % 4] = time;
        }
    }

    /// key_decay stores release time (>0 = released at this time, 0 = still held or idle)
    fn set_key_release_time(&mut self, key_index: usize, time: f32) {
        if key_index < 9 {
            self.current_params.key_decay[key_index / 4][key_index % 4] = time;
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
            sample_offset: 0,
            samples_to_generate: MAX_SAMPLES_PER_FRAME,
            sample_rate: SAMPLE_RATE,
            key_states: [[0.0; 4]; 3],
            key_decay: [[0.0; 4]; 3],
        };

        let audio_buffer_size = (MAX_SAMPLES_PER_FRAME * 2) as usize;

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<SynthParams>()
            .with_audio(audio_buffer_size)
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
            keys_held: [false; 9],
            audio_start: std::time::Instant::now(),
            last_samples_generated: 0,
        }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

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
            let elapsed = self.audio_start.elapsed().as_secs_f64();
            let target_samples = (elapsed * SAMPLE_RATE as f64) as u64;
            let written = stream.samples_written();
            let needed = (target_samples.saturating_sub(written) as u32).min(MAX_SAMPLES_PER_FRAME);
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
                        egui::CollapsingHeader::new("Playback")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Keys 1-9: C D E F G A B C D");

                                let mut beat_enabled = params.beat_enabled > 0;
                                if ui.checkbox(&mut beat_enabled, "Background Beat").changed() {
                                    params.beat_enabled = u32::from(beat_enabled);
                                    changed = true;
                                }

                                changed |= ui
                                    .add(egui::Slider::new(&mut params.tempo, 60.0..=180.0).text("Tempo"))
                                    .changed();
                                changed |= ui
                                    .add(egui::Slider::new(&mut params.octave, 2.0..=7.0).text("Octave"))
                                    .changed();
                                changed |= ui
                                    .add(egui::Slider::new(&mut params.volume, 0.0..=1.0).text("Volume"))
                                    .changed();

                                ui.horizontal(|ui| {
                                    ui.label("Wave:");
                                    for (i, name) in ["Sin", "Saw", "Sqr", "Tri", "Nse"].iter().enumerate() {
                                        if ui.selectable_label(params.waveform_type == i as u32, *name).clicked() {
                                            params.waveform_type = i as u32;
                                            changed = true;
                                        }
                                    }
                                });
                            });

                        egui::CollapsingHeader::new("Envelope (ADSR)")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.attack_time, 0.001..=0.5).logarithmic(true).text("Attack").suffix("s")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.decay_time, 0.01..=1.0).logarithmic(true).text("Decay").suffix("s")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.sustain_level, 0.0..=1.0).text("Sustain")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.release_time, 0.01..=2.0).logarithmic(true).text("Release").suffix("s")).changed();

                                ui.separator();
                                if ui.small_button("Piano").clicked() {
                                    params.attack_time = 0.01; params.decay_time = 0.3;
                                    params.sustain_level = 0.5; params.release_time = 0.8;
                                    changed = true;
                                }
                                ui.horizontal(|ui| {
                                    if ui.small_button("Pad").clicked() {
                                        params.attack_time = 0.2; params.decay_time = 0.5;
                                        params.sustain_level = 0.8; params.release_time = 1.5;
                                        changed = true;
                                    }
                                    if ui.small_button("Pluck").clicked() {
                                        params.attack_time = 0.005; params.decay_time = 0.1;
                                        params.sustain_level = 0.3; params.release_time = 0.2;
                                        changed = true;
                                    }
                                });
                            });

                        egui::CollapsingHeader::new("Filter")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.filter_cutoff, 0.0..=1.0).text("Cutoff")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.filter_resonance, 0.0..=0.9).text("Resonance")).changed();
                            });

                        egui::CollapsingHeader::new("Effects")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.reverb_mix, 0.0..=0.8).text("Reverb")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.delay_time, 0.01..=1.0).text("Delay")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.delay_feedback, 0.0..=0.8).text("Feedback")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.distortion_amount, 0.0..=0.9).text("Distortion")).changed();
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.chorus_rate, 0.1..=10.0).text("Chorus Rate")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.chorus_depth, 0.0..=0.5).text("Chorus Depth")).changed();
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
            params.key_states = self.current_params.key_states;
            params.key_decay = self.current_params.key_decay;
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

                        let current_time = self.base.controls.get_time(&self.base.start_time);
                        if event.state == winit::event::ElementState::Pressed && !self.keys_held[index] {
                            self.keys_held[index] = true;
                            let has_previous = self.current_params.key_states[index / 4][index % 4] > 0.0;
                            let in_release = self.current_params.key_decay[index / 4][index % 4] > 0.0;
                            if has_previous && in_release {
                                // Retrigger: just cancel the release, note continues from current level
                                self.set_key_release_time(index, 0.0);
                            } else {
                                // Fresh note
                                self.set_key_press_time(index, current_time);
                                self.set_key_release_time(index, 0.0);
                            }
                            self.compute_shader
                                .set_custom_params(self.current_params, &core.queue);
                        } else if event.state == winit::event::ElementState::Released {
                            self.keys_held[index] = false;
                            // Store release time — shader ADSR handles the fade
                            self.set_key_release_time(index, current_time);
                            self.compute_shader
                                .set_custom_params(self.current_params, &core.queue);
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
