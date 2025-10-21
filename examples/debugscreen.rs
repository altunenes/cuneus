use cuneus::audio::SynthesisManager;
use cuneus::compute::{ComputeShader, COMPUTE_TEXTURE_FORMAT_RGBA16};
use cuneus::{Core, RenderKit, ShaderApp, ShaderControls, ShaderManager};
use winit::event::*;

struct DebugScreen {
    base: RenderKit,
    compute_shader: ComputeShader,
    audio_synthesis: Option<SynthesisManager>,
    generate_note: bool,
}

impl ShaderManager for DebugScreen {
    fn init(core: &Core) -> Self {
        // Create texture display layout - needed to show compute shader output on screen
        // This layout defines how to bind the texture (binding 0) and sampler (binding 1) for rendering
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        // Entry point configuration
        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_mouse() // Automatically goes to @group(2)
            .with_fonts() // Automatically goes to @group(2)
            .with_audio(1024) // Automatically goes to @group(2)
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Debug Screen")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/debugscreen.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/debugscreen.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Debug Screen Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/debugscreen.wgsl").into(),
                    ),
                }),
        ) {
            eprintln!("Failed to enable hot reload for debugscreen shader: {}", e);
        }

        // init audio synthesis system
        let audio_synthesis = match SynthesisManager::new() {
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
            audio_synthesis,
            generate_note: false,
        }
    }

    fn update(&mut self, core: &Core) {
        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Update mouse data
        if let Some(mouse_uniform) = &mut self.compute_shader.mouse_uniform {
            mouse_uniform.data = self.base.mouse_tracker.uniform;
            mouse_uniform.update(&core.queue);
        }

        self.base.fps_tracker.update();

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);

        // Handle audio generation
        if self.generate_note {
            if self.base.time_uniform.data.frame % 60 == 0 {
                if let Some(ref mut synth) = self.audio_synthesis {
                    let frequency = 220.0 + self.base.mouse_tracker.uniform.position[1] * 440.0;
                    let active = self.base.mouse_tracker.uniform.buttons[0] & 1 != 0;
                    let amp = if active { 0.1 } else { 0.0 };
                    synth.set_voice(0, frequency, amp, active);
                }
            }
        } else if let Some(ref mut synth) = self.audio_synthesis {
            synth.set_voice(0, 440.0, 0.0, false);
        }

        if let Some(ref mut synth) = self.audio_synthesis {
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

        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let mouse_pos = self.base.mouse_tracker.uniform.position;
        let raw_pos = self.base.mouse_tracker.raw_position;
        let mouse_buttons = self.base.mouse_tracker.uniform.buttons[0];
        let mouse_wheel = self.base.mouse_tracker.uniform.wheel;

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill =
                        egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });

                egui::Window::new("Debug Screen").show(ctx, |ui| {
                    ui.heading("Controls");
                    ShaderControls::render_controls_widget(ui, &mut controls_request);

                    ui.separator();
                    ui.heading("Mouse Debug");
                    ui.label(format!(
                        "Position (normalized): {:.3}, {:.3}",
                        mouse_pos[0], mouse_pos[1]
                    ));
                    ui.label(format!(
                        "Position (pixels): {:.1}, {:.1}",
                        raw_pos[0], raw_pos[1]
                    ));
                    ui.label(format!("Buttons: {:#b}", mouse_buttons));
                    ui.label(format!(
                        "Wheel: {:.2}, {:.2}",
                        mouse_wheel[0], mouse_wheel[1]
                    ));

                    ui.separator();
                    ui.heading("Audio Test");
                    if ui.button("Press 5 to generate a simple note").clicked() {
                        self.generate_note = !self.generate_note;
                    }

                    if ui.input(|i| i.key_pressed(egui::Key::Num5)) {
                        self.generate_note = !self.generate_note;
                    }

                    let audio_status = if self.generate_note {
                        "🔊 Note playing"
                    } else {
                        "🔇 No audio"
                    };
                    ui.label(audio_status);

                    if let Some(ref synth) = self.audio_synthesis {
                        if synth.is_gpu_synthesis_enabled() {
                            ui.label("✓ Audio synthesis ready");
                        } else {
                            ui.label("⚠ Audio synthesis not active");
                        }
                    } else {
                        ui.label("❌ Audio synthesis unavailable");
                    }

                    ui.separator();
                    ui.label("Controls:");
                    ui.label("• Scroll wheel");
                    ui.label("• Press 'H' to toggle this UI");
                    ui.label("• Press 'F' to toggle fullscreen");
                    ui.label("• Press '5' to generate audio note");
                });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.apply_control_request(controls_request);

        // Create command encoder
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.compute_shader.dispatch(&mut encoder, core);

        // Render compute output to screen
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Main Render Pass"),
            );

            let compute_texture = self.compute_shader.get_output_texture();
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
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

        if self.base.handle_mouse_input(core, event, false) {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
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

    let (app, event_loop) = ShaderApp::new("Debug Screen", 800, 600);

    app.run(event_loop, |core| DebugScreen::init(core))
}
