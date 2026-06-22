// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Unported License.

use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

cuneus::uniform_params! {
    struct BlackHoleParams {
        disk_inner: f32, disk_outer: f32, disk_thickness: f32, disk_brightness: f32,
        disk_density: f32, noise_scale: f32, swirl_speed: f32, temperature: f32,
        doppler: f32, redshift: f32, beaming: f32, ring_glow: f32,
        cam_x: f32, cam_y: f32, cam_z: f32, cam_pitch: f32,
        cam_yaw: f32, cam_roll: f32, fov: f32, taa_weight: f32,
        exposure: f32, bloom: f32, star_density: f32, gamma: f32,
        spectral_shift: f32, saturation: f32, reddening: f32, sharpen: f32,
        vividness: f32, opacity: f32, highlight: f32, _pad_c: f32,
    }
}

struct BlackHoleShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: BlackHoleParams,
}

impl ShaderManager for BlackHoleShader {
    fn init(core: &Core) -> Self {
        let initial_params = BlackHoleParams {
            disk_inner: 3.0, disk_outer: 14.0, disk_thickness: 0.45, disk_brightness: 1.0,
            disk_density: 1.8, noise_scale: 1.0, swirl_speed: 0.25, temperature: 2.5,
            doppler: 0.55, redshift: 0.0, beaming: 5.0, ring_glow: 3.0,
            cam_x: 0.0, cam_y: 1.6, cam_z: -18.0, cam_pitch: -0.08,
            cam_yaw: 0.0, cam_roll: 0.0, fov: 80.0, taa_weight: 0.8,
            exposure: 1.0, bloom: 0.07, star_density: 1.0, gamma: 1.2,
            spectral_shift: -95.0, saturation: 3.0, reddening: 0.57, sharpen: 0.7,
            vividness: 1.0, opacity: 0.0, highlight: 0.08, _pad_c: 0.0,
        };

        let base = RenderKit::new(core);

        let passes = vec![
            PassDescription::new("bb_lut", &[]),
            PassDescription::new("scene", &["bb_lut"]),
            PassDescription::new("resolve", &["scene", "resolve"]), // tex0=raw, tex1=history
            PassDescription::new("bright", &["resolve"]),
            // note: bloom pyramid: 3 widening levels, each accumulating the previous
            PassDescription::new("blur1_h", &["bright"]),
            PassDescription::new("blur1_v", &["blur1_h"]),
            PassDescription::new("blur2_h", &["blur1_v"]),
            PassDescription::new("blur2_v", &["blur2_h", "blur1_v"]),
            PassDescription::new("blur3_h", &["blur2_v"]),
            PassDescription::new("blur3_v", &["blur3_h", "blur2_v"]),
            PassDescription::new("main_image", &["resolve", "blur3_v"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("scene")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<BlackHoleParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Schwarzschild Black Hole")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/blackhole.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader.set_time(current_time, delta, &core.queue);
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;
        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size,
            self.base.fps_tracker.fps(),
        );

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Schwarzschild")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(330.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Camera").default_open(true).show(ui, |ui| {
                                ui.label("Position:");
                                changed |= ui.add(egui::Slider::new(&mut params.cam_x, -50.0..=50.0).text("X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cam_y, -50.0..=50.0).text("Y")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cam_z, -50.0..=50.0).text("Z")).changed();
                                ui.separator();
                                ui.label("Orientation:");
                                changed |= ui.add(egui::Slider::new(&mut params.cam_pitch, -3.14..=3.14).text("Pitch")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cam_yaw, -3.14..=3.14).text("Yaw")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cam_roll, -3.14..=3.14).text("Roll")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.fov, 15.0..=90.0).text("FOV")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.taa_weight, 0.0..=0.95).text("TAA")).changed();
                            });

                        egui::CollapsingHeader::new("Disk").default_open(true).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.disk_inner, 1.5..=8.0).text("Inner")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.disk_outer, 5.0..=40.0).text("Outer")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.disk_thickness, 0.05..=2.0).text("Thickness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.disk_brightness, 0.1..=5.0).text("Bright")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.disk_density, 0.2..=4.0).text("Density")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.opacity, 0.0..=1.0).text("Opacity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.noise_scale, 0.2..=3.0).text("Turbulence")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.swirl_speed, 0.0..=1.5).text("Swirl")).changed();
                            });

                        egui::CollapsingHeader::new("Optics").default_open(false).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.temperature, 0.3..=2.5).text("Temp")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.vividness, 0.0..=1.0).text("Vividness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.spectral_shift, -120.0..=120.0).text("Shift")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.saturation, 0.0..=3.0).text("Saturation")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.reddening, 0.0..=1.0).text("Reddening")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.doppler, 0.0..=3.0).text("Doppler")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.redshift, 0.0..=3.0).text("Redshift")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.beaming, 0.0..=5.0).text("Beaming")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.ring_glow, 0.0..=3.0).text("Ring")).changed();
                            });

                        egui::CollapsingHeader::new("Post").default_open(false).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.1..=3.0).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.sharpen, 0.0..=2.0).text("Sharpen")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.highlight, 0.02..=0.25).text("Highlight")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.6..=2.2).text("Gamma")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.bloom, 0.0..=0.4).text("Bloom")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.star_density, 0.0..=2.0).text("Stars")).changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.export_manager.apply_ui_request(export_request);

        // if slider moved? -> so lets wipe TAA buffers so it doesn't smear
        if changed || controls_request.should_clear_buffers {
            self.compute_shader.current_frame = 0;
            self.compute_shader.time_uniform.data.frame = 0;
            self.compute_shader.time_uniform.update(&core.queue);
            self.compute_shader.clear_all_buffers(core);
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        self.base.apply_control_request(controls_request);

        if should_start_export {
            self.base.export_manager.start_export();
        }

        if !self.base.export_manager.is_exporting() {
            self.compute_shader.dispatch(&mut frame.encoder, core);
        }

        self.base.renderer.render_to_view(
            &mut frame.encoder,
            &frame.view,
            &self.compute_shader.get_output_texture().bind_group,
        );
        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Schwarzschild Black Hole", 800, 500);
    app.run(event_loop, |core| BlackHoleShader::init(core))
}
