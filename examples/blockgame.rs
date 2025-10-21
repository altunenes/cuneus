// Block Game, Enes Altun, 2025, MIT License

use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockGameParams {
    // 0=menu, 1=playing, 2=game_over
    game_state: i32,
    score: u32,
    current_block: u32,
    total_blocks: u32,

    block_x: f32,
    block_y: f32,
    block_z: f32,

    block_width: f32,
    block_height: f32,
    block_depth: f32,

    movement_speed: f32,
    movement_range: f32,
    drop_triggered: i32,

    camera_height: f32,
    camera_angle: f32,
    camera_scale: f32,

    // Game mech
    perfect_placement: i32,
    game_over: i32,

    _padding: [f32; 2],
}

impl Default for BlockGameParams {
    fn default() -> Self {
        Self {
            game_state: 0,
            score: 0,
            current_block: 0,
            total_blocks: 1,

            block_x: 0.0,
            block_y: 1.0,
            block_z: 0.0,

            block_width: 3.0,
            block_height: 0.6,
            block_depth: 3.0,

            movement_speed: 2.0,
            movement_range: 2.5,
            drop_triggered: 0,

            camera_height: 0.0,
            camera_angle: 0.0,
            camera_scale: 65.0,

            perfect_placement: 0,
            game_over: 0,

            _padding: [0.0; 2],
        }
    }
}

impl UniformProvider for BlockGameParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct BlockTowerGame {
    base: RenderKit,
    compute_shader: ComputeShader,
    last_mouse_click: bool,
    game_params: BlockGameParams,
}

impl ShaderManager for BlockTowerGame {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        // Create single-pass compute shader with mouse, fonts, and game storage
        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_mouse()
            .with_fonts()
            .with_audio(1024) // Used for game state storage, not audio
            .with_workgroup_size([8, 8, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Block Tower Game Unified")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/blockgame.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/blockgame.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Block Tower Game Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blockgame.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for blockgame shader: {}", e);
        }

        Self {
            base,
            compute_shader,
            last_mouse_click: false,
            game_params: BlockGameParams::default(),
        }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);
        self.compute_shader
            .update_mouse_uniform(&self.base.mouse_tracker.uniform, &core.queue);
        self.base.fps_tracker.update();

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);

        self.update_camera_in_shader(&core.queue);
        let mouse_buttons = self.base.mouse_tracker.uniform.buttons[0];
        let mouse_pressed = mouse_buttons & 1 != 0;
        self.last_mouse_click = mouse_pressed;
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
                egui::Window::new("Block Tower")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(220.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Camera")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.game_params.camera_height,
                                        0.0..=20.0,
                                    )
                                    .text("Height"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.game_params.camera_angle,
                                        -3.14159..=3.14159,
                                    )
                                    .text("Angle"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.game_params.camera_scale,
                                        20.0..=200.0,
                                    )
                                    .text("Scale"),
                                );

                                ui.separator();
                                ui.label("Controls:");
                                ui.label("Q/E: Move up/down");
                                ui.label("W/S: Rotate left/right");

                                ui.separator();
                                ui.label("Scale presets:");
                                ui.horizontal(|ui| {
                                    if ui.button("1080p").clicked() {
                                        self.game_params.camera_scale = 50.0;
                                    }
                                    if ui.button("1440p").clicked() {
                                        self.game_params.camera_scale = 65.0;
                                    }
                                    if ui.button("4K").clicked() {
                                        self.game_params.camera_scale = 100.0;
                                    }
                                });

                                if ui.button("Reset Camera").clicked() {
                                    self.game_params.camera_height = 8.0;
                                    self.game_params.camera_angle = 0.0;
                                    self.game_params.camera_scale = 65.0;
                                }
                            });
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Block Game Render Encoder"),
            });

        self.compute_shader.dispatch(&mut encoder, core);

        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Block Game Render Pass"),
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
        let ui_handled = self
            .base
            .egui_state
            .on_window_event(core.window(), event)
            .consumed;

        if self.base.handle_mouse_input(core, event, ui_handled) {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let winit::keyboard::PhysicalKey::Code(key_code) = event.physical_key {
                if event.state == ElementState::Pressed {
                    let camera_speed = 0.5;

                    match key_code {
                        winit::keyboard::KeyCode::KeyQ => {
                            self.game_params.camera_height += camera_speed;
                            return true;
                        }
                        winit::keyboard::KeyCode::KeyE => {
                            self.game_params.camera_height -= camera_speed;
                            return true;
                        }
                        winit::keyboard::KeyCode::KeyW => {
                            self.game_params.camera_angle += 0.1;
                            return true;
                        }
                        winit::keyboard::KeyCode::KeyS => {
                            self.game_params.camera_angle -= 0.1;
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

impl BlockTowerGame {
    fn update_camera_in_shader(&self, queue: &wgpu::Queue) {
        if let Some(audio_buffer) = self.compute_shader.get_audio_buffer() {
            let camera_data = [
                self.game_params.camera_height,
                self.game_params.camera_angle,
                self.game_params.camera_scale,
            ];

            let camera_data_bytes = bytemuck::cast_slice(&camera_data);
            let offset = 5 * std::mem::size_of::<f32>();

            queue.write_buffer(audio_buffer, offset as u64, camera_data_bytes);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    cuneus::gst::init()?;

    let (app, event_loop) = ShaderApp::new("Block Tower Game", 600, 800);

    app.run(event_loop, |core| BlockTowerGame::init(core))
}
