use cuneus::compute::{ComputeShader, COMPUTE_TEXTURE_FORMAT_RGBA16};
use cuneus::{
    Core, ExportManager, RenderKit, ShaderApp, ShaderControls, ShaderManager, UniformProvider,
};
use log::error;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AudioVisParams {
    red_power: f32,
    green_power: f32,
    blue_power: f32,
    green_boost: f32,
    contrast: f32,
    gamma: f32,
    glow: f32,
    _padding: f32,
}

impl Default for AudioVisParams {
    fn default() -> Self {
        Self {
            red_power: 0.98,
            green_power: 0.85,
            blue_power: 0.90,
            green_boost: 1.62,
            contrast: 1.0,
            gamma: 1.0,
            glow: 0.05,
            _padding: 0.0,
        }
    }
}

impl UniformProvider for AudioVisParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct AudioVisCompute {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: AudioVisParams,
}

impl ShaderManager for AudioVisCompute {
    fn init(core: &Core) -> Self {
        let initial_params = AudioVisParams::default();
        let base = RenderKit::new(core);

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<AudioVisParams>()
            .with_audio_spectrum(69) // 64 spectrum + BPM + 4 energy values
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Audio Visualizer Compute")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/audiovis.wgsl", config);


        // Initialize custom uniform with initial parameters
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Update audio spectrum - energy values are computed in spectrum.rs
        // and included in the buffer at indices 65-68
        self.base.update_audio_spectrum(&core.queue);
        self.compute_shader
            .update_audio_spectrum(&self.base.resolution_uniform.data, &core.queue);
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        // Update video texture (this triggers spectrum data polling!)
        let _video_updated = if self.base.using_video_texture {
            self.base.update_video_texture(core, &core.queue)
        } else {
            false
        };
        let _webcam_updated = if self.base.using_webcam_texture {
            self.base.update_webcam_texture(core, &core.queue)
        } else {
            false
        };

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

                egui::Window::new("Audio Visualizer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        // Media controls
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

                        egui::CollapsingHeader::new("Visual Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Color Power:");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.red_power, 0.1..=2.0)
                                            .text("Red Power"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.green_power, 0.1..=2.0)
                                            .text("Green Power"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.blue_power, 0.1..=2.0)
                                            .text("Blue Power"),
                                    )
                                    .changed();

                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.green_boost, 0.0..=3.0)
                                            .text("Green Boost"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.contrast, 0.1..=3.0)
                                            .text("Contrast"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.1..=3.0)
                                            .text("Gamma"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.glow, 0.0..=1.0).text("Glow"),
                                    )
                                    .changed();
                            });

                        ui.separator();

                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();

                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);

                        ui.separator();
                        ui.label(format!("Frame: {}", self.compute_shader.current_frame));
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        // Apply controls
        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_media_requests(core, &controls_request);

        // Apply parameter changes
        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // Create command encoder

        // Dispatch compute shader
        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.default_handle_input(core, event) {
            return true;
        }
        if let WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                error!("Failed to load dropped file: {e:?}");
            }
            return true;
        }
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    cuneus::gst::init()?;
    let (app, event_loop) = ShaderApp::new("Audio Visualizer", 800, 600);

    app.run(event_loop, AudioVisCompute::init)
}
