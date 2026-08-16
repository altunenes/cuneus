use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    struct ExperimentParams {
        twist: f32, rotate: f32, core: f32, spread: f32,
        coherence: f32, fold_balance: f32, anim: f32, iterations: f32,
        hue: f32, spectral: f32, dust: f32, color_var: f32,
        dof: f32, focal: f32, bloom: f32, vignette: f32,
        brightness: f32, exposure: f32, gamma: f32, saturation: f32,
        travel: f32, orbit: f32, zoom: f32, arms: f32,
        bokeh_edge: f32, bokeh_blades: f32, bokeh_fringe: f32, taa_weight: f32,
        sharpen: f32, view_tilt: f32, depth_grade: f32, _pad3: f32,
    }
}

struct ExperimentShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ExperimentParams,
}

impl ShaderManager for ExperimentShader {
    fn init(core: &Core) -> Self {
        let initial_params = ExperimentParams {
            twist: 2.0, rotate: 0.3, core: 1.0, spread: 3.2,
            coherence: 0.3, fold_balance: 1.0, anim: 2.0, iterations: 20.0,
            hue: 0.0, spectral: 1.0, dust: 1.0, color_var: 0.5,
            dof: 2.0, focal: 0.0, bloom: 0.6, vignette: 0.3,
            brightness: 1.5, exposure: 1.0, gamma: 0.8, saturation: 1.0,
            travel: 1.0, orbit: 0.0, zoom: 1.0, arms: 0.0,
            bokeh_edge: 0.0, bokeh_blades: 0.0, bokeh_fringe: 0.5, taa_weight: 0.85,
            sharpen: 0.4, view_tilt: 0.0, depth_grade: 0.0, _pad3: 0.0,
        };

        let base = RenderKit::new(core);
        let passes = vec![
            PassDescription::new("Clear", &[]),
            PassDescription::new("Splat", &[]).with_workgroup_size([8192, 1, 1]),
            PassDescription::new("resolve_raw", &[]),
            PassDescription::new("taa", &["resolve_raw", "taa"]),
            PassDescription::new("main_image", &["taa"]),
        ];
        let config = ComputeShader::builder()
            .with_entry_point("Clear")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<ExperimentParams>()
            .with_atomic_buffer(4)
            .with_label("Spectral Galaxy Attractor")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/galaxy.wgsl", config);
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
        let mut controls_request = self.base.controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Spectral Galaxy").collapsible(true).resizable(true).default_width(300.0).show(ctx, |ui| {
                    egui::CollapsingHeader::new("Shape").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.arms, 0.0..=3.0).text("Arms")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.twist, 1.0..=4.0).text("Twist")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.core, 0.0..=4.0).text("Core")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.fold_balance, 0.0..=1.0).text("Swirl Balance")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.iterations, 10.0..=30.0).step_by(1.0).text("Iterations")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.anim, 0.0..=3.0).text("Anim")).changed();
                    });
                    egui::CollapsingHeader::new("Color").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.hue, 0.0..=1.0).text("Hue")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.spectral, 0.0..=2.0).text("Spectral")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.color_var, 0.0..=3.0).text("Radial")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.coherence, -1.0..=1.0).text("Coherence")).changed();
                    });
                    egui::CollapsingHeader::new("Camera & Optics").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.dof, 0.0..=4.0).text("DoF")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.focal, -2.0..=2.0).text("Focal")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.bokeh_edge, 0.0..=1.0).text("Bokeh Rim")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.bokeh_fringe, 0.0..=1.5).text("Fringe")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.taa_weight, 0.0..=0.99).text("TAA")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.sharpen, 0.0..=1.5).text("Sharpen")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.dust, 0.0..=4.0).text("Dust")).changed();
                    });
                    egui::CollapsingHeader::new("Travel & Depth").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.travel, 0.0..=4.0).text("Travel")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.orbit, 0.0..=3.0).text("Orbit")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.4..=3.0).text("Zoom")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.view_tilt, -1.5..=1.5).text("Tilt")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.depth_grade, 0.0..=1.0).text("Depth Grade")).changed();
                    });
                    egui::CollapsingHeader::new("Post").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.brightness, 0.1..=3.0).text("Brightness")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.1..=4.0).text("Exposure")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.4..=2.2).text("Gamma")).changed();
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
        self.base.apply_control_request(controls_request);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        if !self.base.export_manager.is_exporting() {
            self.compute_shader.dispatch(&mut frame.encoder, core);
        }

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);
        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("ifs Galaxy", 800, 600);
    app.run(event_loop, |core| ExperimentShader::init(core))
}
