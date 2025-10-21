use cuneus::audio::SynthesisManager;
use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SongParams {
    volume: f32,
    octave_shift: f32,
    tempo_multiplier: f32,
    waveform_type: u32,
    crossfade: f32,
    reverb_mix: f32,
    chorus_rate: f32,
    _padding: f32,
}

impl UniformProvider for SongParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct VeridisQuo {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SongParams,
    audio_synthesis: Option<SynthesisManager>,
}

impl ShaderManager for VeridisQuo {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let initial_params = SongParams {
            volume: 0.5,
            octave_shift: 0.0,
            tempo_multiplier: 1.0,
            waveform_type: 1,
            crossfade: 0.0,
            reverb_mix: 0.0,
            chorus_rate: 0.0,
            _padding: 0.0,
        };

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<SongParams>()
            .with_fonts() // Fonts in Group 2
            .with_audio(4096) // Audio buffer in Group 2
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Veridis Quo Unified")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/veridisquo.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/veridisquo.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Veridisquo Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/veridisquo.wgsl").into(),
                    ),
                }),
        ) {
            eprintln!("Failed to enable hot reload for veridisquo shader: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        let audio_synthesis = match SynthesisManager::new() {
            Ok(mut synth) => {
                if let Err(_e) = synth.start_gpu_synthesis() {
                    None
                } else {
                    println!("Audio synthesis started.");
                    Some(synth)
                }
            }
            Err(_e) => None,
        };

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            audio_synthesis,
        }
    }

    fn update(&mut self, core: &Core) {
        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);

        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);
        self.base.fps_tracker.update();

        // Handle GPU audio reading for CPU synthesis
        if self.base.time_uniform.data.frame % 2 == 0 {
            if let Some(ref mut synth) = self.audio_synthesis {
                // Update the waveform type for all voices
                synth.update_waveform(self.current_params.waveform_type);

                // Read melody and bass frequencies from GPU's audio buffer
                // The GPU shader writes: melody_freq, envelope, waveform_type, final_melody_freq, melody_amp, final_bass_freq, bass_amp
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
                        synth.set_voice(0, final_melody_freq, melody_amp, melody_active);

                        // Voice 1: Bass
                        let bass_active = bass_amp > 0.01 && final_bass_freq > 10.0;
                        synth.set_voice(1, final_bass_freq, bass_amp, bass_active);
                    }
                }
            }
        }
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
                                let waveform_names = [
                                    ("Sine", 0),
                                    ("Square", 1),
                                    ("Saw", 2),
                                    ("Triangle", 3),
                                    ("Pulse", 4),
                                ];
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

        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Veridis Quo Render Encoder"),
            });

        // Single stage dispatch
        self.compute_shader.dispatch(&mut encoder, core);

        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Veridis Quo Render Pass"),
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

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
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

    app.run(event_loop, |core| VeridisQuo::init(core))
}
