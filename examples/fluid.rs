use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

cuneus::uniform_params! {
    struct FluidParams {
    rotation_speed: f32,
    motor_strength: f32,
    distortion: f32,
    feedback: f32,
    particle_size: f32,
    _padding: [f32; 7]}
}

struct FluidShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: FluidParams}

impl ShaderManager for FluidShader {
    fn init(core: &Core) -> Self {
        let initial_params = FluidParams {
            rotation_speed: 2.0,
            motor_strength: 0.01,
            distortion: 10.0,
            feedback: 0.95,
            particle_size: 1.0,
            _padding: [0.0; 7]};
        let base = RenderKit::new(core);

        // Create multipass system: buffer_a (simulation) -> main_image (display)
        let passes = vec![
            PassDescription::new("buffer_a", &[]),
            PassDescription::new("main_image", &["buffer_a"]),
        ];

        // Use input texture in Group 1 for external input
        let config = ComputeShader::builder()
            .with_entry_point("buffer_a")
            .with_multi_pass(&passes)
            .with_channels(1) // Enable channel0 in Group 2 - accessible from all passes!
            .with_custom_uniforms::<FluidParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Fluid Unified")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/fluid.wgsl", config);

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params}
    }

    fn update(&mut self, core: &Core) {
        // Update current texture (video/webcam/static)
        self.base.update_current_texture(core, &core.queue);

        // Update channel0 with external texture (accessible from all passes!)
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            self.compute_shader.update_channel_texture(
                0,
                &texture_manager.view,
                &texture_manager.sampler,
                &core.device,
                &core.queue,
            );
        }

        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Fluid Simulation")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Fluid Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.rotation_speed, -5.0..=5.0)
                                            .text("Rotation Speed"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.motor_strength, -0.2..=0.2)
                                            .text("Motor Strength"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.distortion, 1.0..=20.0)
                                            .text("Distortion"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.feedback, 0.0..=1.01)
                                            .text("Feedback"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Quality")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.particle_size, 0.0..=1.5)
                                            .text("Particle Size"),
                                    )
                                    .changed();
                            });

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
                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();
                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        // Handle controls and clear buffers if requested
        if controls_request.should_clear_buffers {
            // Reset frame count to restart simulation
            self.compute_shader.current_frame = 0;
        }

        // Execute multi-pass compute shader: buffer_a -> main_image
        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        self.base.apply_media_requests(core, &controls_request);

        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Fluid Simulation", 800, 600);
    app.run(event_loop, FluidShader::init)
}
