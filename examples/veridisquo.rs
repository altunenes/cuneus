use cuneus::audio::{EnvelopeConfig, SynthesisManager};
use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

cuneus::uniform_params! {
    struct SongParams {
    volume: f32,
    octave_shift: f32,
    tempo_multiplier: f32,
    waveform_type: u32,
    crossfade: f32,
    reverb_mix: f32,
    chorus_rate: f32,
    _padding: f32}
}

struct VeridisQuo {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SongParams,
    audio_synthesis: Option<SynthesisManager>}

impl ShaderManager for VeridisQuo {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        let initial_params = SongParams {
            volume: 0.5,
            octave_shift: 0.0,
            tempo_multiplier: 1.0,
            waveform_type: 1,
            crossfade: 0.0,
            reverb_mix: 0.0,
            chorus_rate: 0.0,
            _padding: 0.0};

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<SongParams>()
            .with_fonts()
            .with_audio(4096)
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Veridis Quo Unified")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/veridisquo.wgsl", config);

        compute_shader.set_custom_params(initial_params, &core.queue);

        let audio_synthesis = match SynthesisManager::new() {
            Ok(mut synth) => {
                // For continuous song playback, use minimal envelope
                // Quick attack, full sustain, quick release for smooth note transitions
                synth.set_envelope(EnvelopeConfig {
                    attack_time: 0.005, // 5ms - very quick attack
                    decay_time: 0.01,   // 10ms decay
                    sustain_level: 1.0, // Full sustain for continuous playback
                    release_time: 0.05, // 50ms release for smooth transitions
                });

                if let Err(e) = synth.start_gpu_synthesis() {
                    eprintln!("Failed to start audio synthesis: {e}");
                    None
                } else {
                    println!("Audio synthesis started.");
                    Some(synth)
                }
            }
            Err(e) => {
                eprintln!("Failed to create audio synthesis: {e}");
                None
            }
        };

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            audio_synthesis}
    }

    fn update(&mut self, core: &Core) {

        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Read GPU audio parameters and update synthesis
        if let Some(ref mut synth) = self.audio_synthesis {
            // Update waveform type
            synth.update_waveform(self.current_params.waveform_type);
            synth.set_master_volume(self.current_params.volume as f64);

            // Read melody and bass frequencies from GPU's audio buffer
            if let Ok(audio_data) = pollster::block_on(
                self.compute_shader
                    .read_audio_buffer(&core.device, &core.queue),
            ) {
                if audio_data.len() >= 7 {
                    let final_melody_freq = audio_data[3];
                    let melody_amp = audio_data[4];
                    let final_bass_freq = audio_data[5];
                    let bass_amp = audio_data[6];

                    // Voice 0: Melody
                    let melody_active = melody_amp > 0.01 && final_melody_freq > 10.0;
                    synth.set_voice(0, final_melody_freq, melody_amp * 0.8, melody_active);

                    // Voice 1: Bass
                    let bass_active = bass_amp > 0.01 && final_bass_freq > 10.0;
                    synth.set_voice(1, final_bass_freq, bass_amp * 0.6, bass_active);
                }
            }

            // Must call update() every frame to process envelopes
            synth.update();
        }
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

                egui::Window::new("Veridis Quo")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Audio Controls")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.volume, 0.0..=1.0)
                                            .text("Volume"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.octave_shift, -2.0..=2.0)
                                            .text("Octave"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.tempo_multiplier, 0.5..=4.0)
                                            .text("Tempo"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Waveforms")
                            .default_open(false)
                            .show(ui, |ui| {
                                let waveform_names =
                                    [("Sine", 0), ("Square", 1), ("Saw", 2), ("Triangle", 3)];
                                for (name, wave_type) in waveform_names.iter() {
                                    let selected = params.waveform_type == *wave_type;
                                    if ui.selectable_label(selected, *name).clicked() {
                                        params.waveform_type = *wave_type;
                                        changed = true;
                                    }
                                }
                            });

                        egui::CollapsingHeader::new("Effects")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.crossfade, 0.0..=1.0)
                                            .text("Legato"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.reverb_mix, 0.0..=1.0)
                                            .text("Reverb"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.chorus_rate, 0.1..=8.0)
                                            .text("Chorus Rate"),
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
        }

        self.base.apply_control_request(controls_request);


        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
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
            if event.state == winit::event::ElementState::Pressed {
                if let winit::keyboard::Key::Character(ref s) = event.logical_key {
                    match s.as_str() {
                        "r" | "R" => {
                            self.base.start_time = std::time::Instant::now();
                            return true;
                        }
                        _ => {}
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

    let (app, event_loop) = ShaderApp::new("Veridis Quo", 800, 600);

    app.run(event_loop, VeridisQuo::init)
}
