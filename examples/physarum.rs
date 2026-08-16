use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    #[allow(non_snake_case)]
    struct PhysarumParams {
        sa: f32, sd: f32, drg: f32, spd: f32, dec: f32, dif: f32, dep: f32, jit: f32,
        rSd: f32, mSc: f32, fSc: f32, sGn: f32, sAt: f32, sRp: f32, str: f32, aSc: f32,
        glw: f32, cSh: f32, spc: f32, gam: f32, aCt: f32, aSp: f32, aRd: f32, aSt: f32,
        cSp: f32, sat: f32, pMx: f32, blr: f32,
        c0r: f32, c0g: f32, c0b: f32, c1r: f32, c1g: f32, c1b: f32, c2r: f32, c2g: f32, c2b: f32,
        sub: f32, tSm: f32, wnd: f32, tur: f32, p1: f32, flu: f32, p3: f32,
    }
}

struct PhysarumShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: PhysarumParams,
}

impl ShaderManager for PhysarumShader {
    fn init(core: &Core) -> Self {
        let initial_params = PhysarumParams {
            sa: 1.50, sd: 40.0, drg: 0.20, spd: 4.95, dec: 0.999, dif: 0.10, dep: 30.0, jit: 0.500,
            rSd: 1.0, mSc: 0.30, fSc: 0.80, sGn: 5.0, sAt: 1.00, sRp: 0.92, str: 0.15, aSc: 1.00,
            glw: 0.00, cSh: -0.02, spc: 0.25, gam: 0.50, aCt: 3.0, aSp: 0.30, aRd: 250.0, aSt: 0.15,
            cSp: 0.30, sat: 1.58, pMx: 0.57, blr: 0.99,
            c0r: 1.0, c0g: 0.2, c0b: 0.2, c1r: 0.2, c1g: 0.8, c1b: 0.3, c2r: 0.2, c2g: 0.4, c2b: 1.0,
            sub: 1.0, tSm: 1.0, wnd: 0.8, tur: 0.0, p1: 0.6, flu: 0.0, p3: 0.0,
        };

        let base = RenderKit::new(core);

        let passes = vec![
            PassDescription::new("agent_update", &["agent_update", "turing_resolve"]).with_resolution(1024, 1024),
            PassDescription::new("process_trails", &["process_trails"]),
            PassDescription::new("diffuse_h", &["process_trails"]),
            PassDescription::new("diffuse_v", &["process_trails", "diffuse_h"]),
            PassDescription::new("inhibitor_down", &["diffuse_v"]).with_resolution_scale(0.125),
            PassDescription::new("turing_resolve", &["process_trails", "diffuse_v", "inhibitor_down"]),
            PassDescription::new("main_image", &["process_trails", "turing_resolve", "inhibitor_down"]),
        ];

        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_custom_uniforms::<PhysarumParams>()
            .with_atomic_buffer(4)
            .with_label("Physarum Simulation")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/physarum.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self { base, compute_shader, current_params: initial_params }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(current_time, 1.0 / 60.0, &core.queue);
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
            &self.base.start_time, &core.size, self.base.fps_tracker.fps()
        );

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Physarum Controls")
                    .collapsible(true).resizable(true).default_width(320.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Behavior Rule").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.rSd, 0.0..=100.0).text("Rule Seed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mSc, 0.0..=1.0).text("Species Diversity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.fSc, 0.0..=3.0).text("Force Scale")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sGn, 0.5..=30.0).text("Sensor Gain")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.str, 0.0..=2.0).text("Strafe Power")).changed();
                        });

                        egui::CollapsingHeader::new("Agent Physics").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.spd, 0.1..=5.0).text("Speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.drg, 0.0..=0.99).text("Momentum")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.wnd, 0.0..=3.0).text("Slipstream Wind")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.flu, 0.0..=2.0).text("Bubble vs Vein Bias")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sd, 1.0..=40.0).text("Sensor Distance")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sa, 0.05..=1.5).text("Sensor Angle")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.jit, 0.0..=0.5).text("Random Jitter")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aSc, 0.05..=1.0).text("Agent Density")).changed();
                        });

                        egui::CollapsingHeader::new("Trail Environment").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.dep, 1.0..=30.0).text("Deposit Amount")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.dec, 0.9..=0.999).text("Decay Rate")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.dif, 0.0..=1.0).text("Diffusion Rate")).changed();
                        });

                        egui::CollapsingHeader::new("Species Interaction").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.sAt, 0.0..=1.0).text("Cross-Attraction")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sRp, 0.0..=1.0).text("Cross-Repulsion")).changed();
                        });

                        egui::CollapsingHeader::new("Attractors").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.aCt, 0.0..=8.0).step_by(1.0).text("Count")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aSp, 0.0..=2.0).text("Orbit Speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aRd, 50.0..=500.0).text("Radius")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aSt, 0.0..=1.0).text("Strength")).changed();
                        });

                        egui::CollapsingHeader::new("Rendering").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.blr, 0.0..=1.0).text("Feedback Fade")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.p1, 0.0..=3.0).text("Vein Relief")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.glw, 0.0..=2.0).text("Glow")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.spc, 0.0..=1.5).text("Specular")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gam, 0.1..=2.2).text("Gamma")).changed();
                        });

                        egui::CollapsingHeader::new("Colors").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.pMx, 0.0..=1.0).text("Palette Mix")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.cSh, -0.5..=0.5).text("Color Shift")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.cSp, 0.0..=1.0).text("Species Hue Spread")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sat, 0.0..=2.5).text("Saturation")).changed();
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

        if controls_request.should_clear_buffers { self.compute_shader.current_frame = 0; }
        if !self.base.export_manager.is_exporting() { self.compute_shader.dispatch(&mut frame.encoder, core); }

        self.base.renderer.render_to_view(
            &mut frame.encoder,
            &frame.view,
            &self.compute_shader.get_output_texture().bind_group,
        );
        self.base.apply_control_request(controls_request);
        self.base.export_manager.apply_ui_request(export_request);

        if should_start_export { self.base.export_manager.start_export(); }
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
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Physarum Engine", 1280, 720);
    app.run(event_loop, PhysarumShader::init)
}