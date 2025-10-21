use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

struct CameraMovement {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    speed: f32,
    last_update: std::time::Instant,

    yaw: f32,
    pitch: f32,
    mouse_sensitivity: f32,

    last_mouse_x: f32,
    last_mouse_y: f32,
    mouse_initialized: bool,
    mouse_look_enabled: bool,
}

impl Default for CameraMovement {
    fn default() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            speed: 2.0,
            last_update: std::time::Instant::now(),

            yaw: 0.0,
            pitch: 0.0,
            mouse_sensitivity: 0.005,

            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            mouse_initialized: false,
            mouse_look_enabled: true,
        }
    }
}

impl CameraMovement {
    fn update_camera(&mut self, params: &mut PathTracingParams) -> bool {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;

        let mut changed = false;

        let forward = [
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        ];

        let world_up = [0.0, 1.0, 0.0];
        let right = [
            forward[1] * world_up[2] - forward[2] * world_up[1],
            forward[2] * world_up[0] - forward[0] * world_up[2],
            forward[0] * world_up[1] - forward[1] * world_up[0],
        ];

        let right_len = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
        let right = [
            right[0] / right_len,
            right[1] / right_len,
            right[2] / right_len,
        ];

        let delta = self.speed * dt;
        let mut move_vec = [0.0, 0.0, 0.0];

        if self.forward {
            move_vec[0] += forward[0] * delta;
            move_vec[1] += forward[1] * delta;
            move_vec[2] += forward[2] * delta;
            changed = true;
        }
        if self.backward {
            move_vec[0] -= forward[0] * delta;
            move_vec[1] -= forward[1] * delta;
            move_vec[2] -= forward[2] * delta;
            changed = true;
        }
        if self.right {
            move_vec[0] += right[0] * delta;
            move_vec[1] += right[1] * delta;
            move_vec[2] += right[2] * delta;
            changed = true;
        }
        if self.left {
            move_vec[0] -= right[0] * delta;
            move_vec[1] -= right[1] * delta;
            move_vec[2] -= right[2] * delta;
            changed = true;
        }
        if self.up {
            move_vec[1] += delta;
            changed = true;
        }
        if self.down {
            move_vec[1] -= delta;
            changed = true;
        }

        params.camera_pos_x += move_vec[0];
        params.camera_pos_y += move_vec[1];
        params.camera_pos_z += move_vec[2];

        let look_distance = 1.0;
        params.camera_target_x = params.camera_pos_x + forward[0] * look_distance;
        params.camera_target_y = params.camera_pos_y + forward[1] * look_distance;
        params.camera_target_z = params.camera_pos_z + forward[2] * look_distance;

        changed
    }

    fn handle_mouse_movement(&mut self, x: f32, y: f32) -> bool {
        if !self.mouse_look_enabled {
            return false;
        }

        if !self.mouse_initialized {
            self.last_mouse_x = x;
            self.last_mouse_y = y;
            self.mouse_initialized = true;
            return false;
        }

        let dx = x - self.last_mouse_x;
        let dy = y - self.last_mouse_y;

        self.last_mouse_x = x;
        self.last_mouse_y = y;

        self.yaw += dx * self.mouse_sensitivity;
        self.pitch -= dy * self.mouse_sensitivity;

        self.pitch = self
            .pitch
            .clamp(-std::f32::consts::PI * 0.49, std::f32::consts::PI * 0.49);

        true
    }

    fn toggle_mouse_look(&mut self) {
        self.mouse_look_enabled = !self.mouse_look_enabled;
        self.mouse_initialized = false;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PathTracingParams {
    camera_pos_x: f32,
    camera_pos_y: f32,
    camera_pos_z: f32,
    camera_target_x: f32,
    camera_target_y: f32,
    camera_target_z: f32,
    fov: f32,
    aperture: f32,

    max_bounces: u32,
    samples_per_pixel: u32,
    accumulate: u32,

    num_spheres: u32,
    mouse_x: f32,
    mouse_y: f32,

    rotation_speed: f32,

    exposure: f32,
}

impl UniformProvider for PathTracingParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct PathTracingShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: PathTracingParams,
    camera_movement: CameraMovement,
    frame_count: u32,
    should_reset_accumulation: bool,
}

impl PathTracingShader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
        self.frame_count = 0;
        self.should_reset_accumulation = false;
    }
}

impl ShaderManager for PathTracingShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let initial_params = PathTracingParams {
            camera_pos_x: 0.0,
            camera_pos_y: 1.0,
            camera_pos_z: 6.0,
            camera_target_x: 0.0,
            camera_target_y: 0.0,
            camera_target_z: -1.0,
            fov: 40.0,
            aperture: 0.00,
            max_bounces: 4,
            samples_per_pixel: 2,
            accumulate: 1,
            num_spheres: 15,
            mouse_x: 0.5,
            mouse_y: 0.5,
            rotation_speed: 0.2,
            exposure: 1.5,
        };

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_input_texture() // Enable input texture support for background
            .with_custom_uniforms::<PathTracingParams>()
            .with_storage_buffer(StorageBufferSpec::new(
                "atomic_buffer",
                (core.size.width * core.size.height * 3 * 4) as u64,
            )) // 3 channels * u32 per pixel
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Path Tracing Unified")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/pathtracing.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/pathtracing.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Path Tracing Compute Shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/pathtracing.wgsl").into(),
                    ),
                }),
        ) {
            eprintln!(
                "Failed to enable Path Tracing compute shader hot reload: {}",
                e
            );
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            camera_movement: CameraMovement::default(),
            frame_count: 0,
            should_reset_accumulation: true,
        }
    }

    fn update(&mut self, core: &Core) {
        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Update input textures for background
        self.base.update_current_texture(core, &core.queue);
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            self.compute_shader.update_input_texture(
                &texture_manager.view,
                &texture_manager.sampler,
                &core.device,
            );
        }

        if self.camera_movement.update_camera(&mut self.current_params) {
            self.compute_shader
                .set_custom_params(self.current_params, &core.queue);
            self.should_reset_accumulation = true;
        }

        self.base.fps_tracker.update();
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
        self.should_reset_accumulation = true;
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Path Tracing Render Encoder"),
            });

        // Handle UI and parameter updates
        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let current_fps = self.base.fps_tracker.fps();
        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();

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

                egui::Window::new("Path Tracer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        ui.label("Camera Controls:");
                        ui.label("W/A/S/D - Movements");
                        ui.label("Q/E - down/up");
                        ui.label("Mouse - Look around");
                        ui.label("Right Click - Toggle mouse look");
                        ui.label("Space - Toggle progressive rendering");
                        ui.separator();
                        ShaderControls::render_media_panel(
                            ui,
                            &mut controls_request,
                            using_video_texture,
                            video_info,
                            using_hdri_texture,
                            hdri_info,
                            using_webcam_texture,
                            webcam_info,
                        );
                        ui.separator();

                        egui::CollapsingHeader::new("Render Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                let old_samples = params.samples_per_pixel;
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.samples_per_pixel, 1..=16)
                                            .text("Samples/pixel"),
                                    )
                                    .changed();
                                if params.samples_per_pixel != old_samples {
                                    self.should_reset_accumulation = true;
                                }

                                let old_bounces = params.max_bounces;
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.max_bounces, 1..=16)
                                            .text("Max Bounces"),
                                    )
                                    .changed();
                                if params.max_bounces != old_bounces {
                                    self.should_reset_accumulation = true;
                                }

                                let old_accumulate = params.accumulate;
                                let mut accumulate_bool = params.accumulate > 0;
                                changed |= ui
                                    .checkbox(&mut accumulate_bool, "Progressive Rendering")
                                    .changed();
                                params.accumulate = if accumulate_bool { 1 } else { 0 };
                                if params.accumulate != old_accumulate {
                                    self.should_reset_accumulation = true;
                                }

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.exposure, 0.1..=5.0)
                                            .text("Exposure"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.aperture, 0.0..=0.5)
                                            .text("Depth of Field"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.rotation_speed, 0.0..=2.0)
                                            .text("Animation Speed"),
                                    )
                                    .changed();

                                if ui.button("Reset Accumulation").clicked() {
                                    self.should_reset_accumulation = true;
                                    changed = true;
                                }
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);
                        ui.separator();
                        ui.label(format!("Accumulated Samples: {}", self.frame_count));
                        ui.label(format!(
                            "Resolution: {}x{}",
                            core.size.width, core.size.height
                        ));
                        ui.label(format!("FPS: {:.1}", current_fps));
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        // Apply controls
        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers || self.should_reset_accumulation {
            self.clear_buffers(core);
        }
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        self.base.handle_webcam_requests(core, &controls_request);

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // Update mouse position in params
        params.mouse_x = self.base.mouse_tracker.uniform.position[0];
        params.mouse_y = self.base.mouse_tracker.uniform.position[1];
        changed = true;

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);

        // Set frame count for random number generation
        self.compute_shader.time_uniform.data.frame = self.frame_count;
        self.compute_shader.time_uniform.update(&core.queue);

        // Single stage dispatch
        self.compute_shader.dispatch(&mut encoder, core);

        // Render compute output to screen
        {
            let mut render_pass = Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Path Tracing Display Pass"),
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

        // Increment frame count for progressive rendering and noise generation
        if self.current_params.accumulate > 0 {
            self.frame_count += 1;
        } else {
            self.frame_count = (self.frame_count + 1) % 1000;
        }

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
            if let winit::keyboard::Key::Character(ch) = &event.logical_key {
                match ch.as_str() {
                    "w" | "W" => {
                        self.camera_movement.forward =
                            event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    }
                    "s" | "S" => {
                        self.camera_movement.backward =
                            event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    }
                    "a" | "A" => {
                        self.camera_movement.left =
                            event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    }
                    "d" | "D" => {
                        self.camera_movement.right =
                            event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    }
                    "q" | "Q" => {
                        self.camera_movement.down =
                            event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    }
                    "e" | "E" => {
                        self.camera_movement.up =
                            event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    }
                    " " => {
                        if event.state == winit::event::ElementState::Released {
                            self.current_params.accumulate = 1 - self.current_params.accumulate;
                            self.should_reset_accumulation = true;
                            self.compute_shader
                                .set_custom_params(self.current_params, &core.queue);
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }

        if let WindowEvent::CursorMoved { position, .. } = event {
            let x = position.x as f32;
            let y = position.y as f32;

            if self.camera_movement.handle_mouse_movement(x, y) {
                self.should_reset_accumulation = true;
                return true;
            }
        }

        if let WindowEvent::MouseInput { state, button, .. } = event {
            if *button == winit::event::MouseButton::Right {
                if *state == winit::event::ElementState::Released {
                    self.camera_movement.toggle_mouse_look();
                    return true;
                }
            }
        }

        if let WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                eprintln!("Failed to load dropped file: {:?}", e);
            }
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if self
                .base
                .key_handler
                .handle_keyboard_input(core.window(), event)
            {
                return true;
            }
        }

        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    cuneus::gst::init()?;
    let (app, event_loop) = ShaderApp::new("Path Tracer", 800, 600);

    app.run(event_loop, |core| PathTracingShader::init(core))
}
