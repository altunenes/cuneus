use cuneus::audio::SynthesisManager;
use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
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
    fade_speed_initial: f32,
    fade_speed_sustain: f32,
    fade_speed_tail: f32,
    _padding1: u32,
    key_states: [[f32; 4]; 3],
    key_decay: [[f32; 4]; 3],
}

impl UniformProvider for SynthParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct SynthManager {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SynthParams,
    gpu_synthesis: Option<SynthesisManager>,
    key_press_times: [Option<std::time::Instant>; 9],
}

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
}

impl ShaderManager for SynthManager {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let initial_params = SynthParams {
            tempo: 120.0,
            waveform_type: 1,
            octave: 4.0,
            volume: 0.85,
            beat_enabled: 1,
            reverb_mix: 0.15,
            delay_time: 0.3,
            delay_feedback: 0.4,
            filter_cutoff: 0.8,
            filter_resonance: 0.1,
            distortion_amount: 0.0,
            chorus_rate: 2.0,
            chorus_depth: 0.15,
            attack_time: 0.015,
            decay_time: 0.6,
            sustain_level: 0.6,
            release_time: 1.2,
            fade_speed_initial: 0.92,
            fade_speed_sustain: 0.96,
            fade_speed_tail: 0.98,
            _padding1: 0,
            key_states: [[0.0; 4]; 3],
            key_decay: [[0.0; 4]; 3],
        };

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<SynthParams>()
            .with_audio(2048) // Audio buffer in Group 2
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Synth Unified")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/synth.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/synth.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Synth Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/synth.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for synth shader: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        // Initialize audio synthesis system
        let gpu_synthesis = match SynthesisManager::new() {
            Ok(mut synth) => {
                if let Err(_e) = synth.start_gpu_synthesis() {
                    None
                } else {
                    Some(synth)
                }
            }
            Err(_e) => None,
        };

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            gpu_synthesis,
            key_press_times: [None; 9],
        }
    }

    fn update(&mut self, core: &Core) {
        self.base.fps_tracker.update();

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);

        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Update key states for GPU shader envelope computation
        let mut keys_updated = false;
        for i in 0..9 {
            if let Some(_press_time) = self.key_press_times[i] {
                // Key is currently held - let GPU handle envelope progression
                self.set_key_decay(i, 1.0);
                keys_updated = true;
            } else {
                // Key released - fade to silence
                let current_decay = self.current_params.key_decay[i / 4][i % 4];
                if current_decay > 0.005 {
                    // Use customizable piano fade speeds
                    let fade_speed = if current_decay > 0.8 {
                        self.current_params.fade_speed_initial
                    } else if current_decay > 0.4 {
                        self.current_params.fade_speed_sustain
                    } else {
                        self.current_params.fade_speed_tail
                    };
                    let new_decay = current_decay * fade_speed;
                    self.set_key_decay(i, new_decay);
                    keys_updated = true;
                } else {
                    // Fade complete
                    self.set_key_state(i, 0.0);
                    self.set_key_decay(i, 0.0);
                }
            }
        }

        if keys_updated {
            self.compute_shader
                .set_custom_params(self.current_params, &core.queue);
        }

        // Read GPU shader-generated audio parameters from the audio buffer
        // The GPU writes audio parameters to the buffer every frame
        if self.base.time_uniform.data.frame % 5 == 0 {
            if let Some(ref mut synth) = self.gpu_synthesis {
                // Update waveform type from GPU shader params (the GPU updates this)
                synth.update_waveform(self.current_params.waveform_type);

                // Control individual voices using GPU-computed frequencies and envelopes
                for i in 0..9 {
                    let key_state = self.current_params.key_states[i / 4][i % 4];
                    let key_decay = self.current_params.key_decay[i / 4][i % 4];

                    if key_state > 0.5 || key_decay > 0.001 {
                        // Calculate frequency for this key (C major scale)
                        let notes = [
                            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33,
                        ];
                        let octave_multiplier = 2.0_f32.powf(self.current_params.octave - 4.0);
                        let frequency = notes[i] * octave_multiplier;

                        // Use key_decay as envelope amplitude (fades out when key released)
                        let amplitude = key_decay * self.current_params.volume * 0.15;
                        let active = key_decay > 0.001;

                        synth.set_voice(i, frequency, amplitude, active);
                    } else {
                        synth.set_voice(i, 440.0, 0.0, false);
                    }
                }

                // Background beat if enabled
                if self.current_params.beat_enabled > 0 {
                    let beat_freq = self.current_params.tempo * 2.0;
                    let beat_time = self.base.time_uniform.data.time * beat_freq / 60.0;
                    let beat_amp = if beat_time.fract() < 0.1 { 0.1 } else { 0.0 };
                    synth.set_voice(8, beat_freq, beat_amp, beat_amp > 0.0);
                }
            }
        }

        if let Some(ref mut synth) = self.gpu_synthesis {
            synth.update();
        }
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Synth Render Encoder"),
            });

        let mut params = self.current_params;
        let mut changed = false;
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill =
                        egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style
                        .text_styles
                        .get_mut(&egui::TextStyle::Body)
                        .unwrap()
                        .size = 11.0;
                    style
                        .text_styles
                        .get_mut(&egui::TextStyle::Button)
                        .unwrap()
                        .size = 10.0;
                });

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
                                ui.label("• All audio generated on GPU compute shaders");
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
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.attack_time, 0.001..=2.0)
                                            .logarithmic(true)
                                            .text("Attack"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.decay_time, 0.01..=3.0)
                                            .logarithmic(true)
                                            .text("Decay"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.sustain_level, 0.0..=1.0)
                                            .text("Sustain"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.release_time, 0.01..=5.0)
                                            .logarithmic(true)
                                            .text("Release"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Response")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.fade_speed_initial,
                                            0.5..=0.999,
                                        )
                                        .text("Initial")
                                        .custom_formatter(|n, _| format!("{:.3}", n)),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.fade_speed_sustain,
                                            0.5..=0.999,
                                        )
                                        .text("Sustain")
                                        .custom_formatter(|n, _| format!("{:.3}", n)),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.fade_speed_tail, 0.5..=0.999)
                                            .text("Tail")
                                            .custom_formatter(|n, _| format!("{:.3}", n)),
                                    )
                                    .changed();

                                if ui.small_button("Reset").clicked() {
                                    params.fade_speed_initial = 0.92;
                                    params.fade_speed_sustain = 0.96;
                                    params.fade_speed_tail = 0.98;
                                    changed = true;
                                }
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
        }

        self.base.apply_control_request(controls_request);

        // Single stage dispatch
        self.compute_shader.dispatch(&mut encoder, core);

        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.compute_shader.output_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        self.base
            .handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();

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
            if event.state == winit::event::ElementState::Pressed {
                if let winit::keyboard::Key::Character(ref s) = event.logical_key {
                    if let Some(key_index) = s.chars().next().and_then(|c| c.to_digit(10)) {
                        if key_index >= 1 && key_index <= 9 {
                            let index = (key_index - 1) as usize;

                            // Only start if not already pressed (prevent retriggering)
                            if self.key_press_times[index].is_none() {
                                self.key_press_times[index] = Some(std::time::Instant::now());
                                self.set_key_state(index, 1.0);
                                self.set_key_decay(index, 1.0);
                                self.compute_shader
                                    .set_custom_params(self.current_params, &core.queue);
                            }
                            return true;
                        }
                    }
                }
            } else if event.state == winit::event::ElementState::Released {
                // Handle key release for smooth fade-out
                if let winit::keyboard::Key::Character(ref s) = event.logical_key {
                    if let Some(key_index) = s.chars().next().and_then(|c| c.to_digit(10)) {
                        if key_index >= 1 && key_index <= 9 {
                            let index = (key_index - 1) as usize;

                            // Start fade-out process
                            if self.key_press_times[index].is_some() {
                                self.key_press_times[index] = None; // This triggers fade-out in update()
                                self.set_key_state(index, 0.0);
                                // Keep current decay value for smooth fade
                                self.compute_shader
                                    .set_custom_params(self.current_params, &core.queue);
                            }
                            return true;
                        }
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
    app.run(event_loop, |core| SynthManager::init(core))
}
