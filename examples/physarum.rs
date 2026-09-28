use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    #[allow(non_snake_case)]
    struct PhysarumParams {
        sa: f32, sd: f32, drg: f32, spd: f32, dec: f32, dif: f32, dep: f32, jit: f32,
        rSd: f32, mSc: f32, fSc: f32, sGn: f32, sAt: f32, _s0: f32, str: f32, aSc: f32,
        glw: f32, cSh: f32, spc: f32, gam: f32, cSp: f32, sat: f32, pal: f32, rel: f32,
        tur: f32, rct: f32, stm: f32, vis: f32, org: f32, osz: f32, shn: f32, ohu: f32,
        bgl: f32, fod: f32, ogl: f32, ovr: f32,
        act: f32, bth: f32, spk: f32, eml: f32,
        mit: f32, fcs: f32, apr: f32, fdr: f32,
        sdf: f32, edb: f32, srd: f32, fcx: f32,
        fcy: f32, mtr: f32, vac: f32, pak: f32,
        cal: f32, cyt: f32, tbe: f32, bgf: f32,
        tml: f32, imm: f32, swl: f32, aAB: f32,
        aBC: f32, aCA: f32, stn: f32, dvs: f32,
        prp: f32, _i0: f32, _i1: f32, _i2: f32,
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
            sa: 1.50, sd: 40.0, drg: 0.20, spd: 4.95, dec: 0.995, dif: 0.40, dep: 30.0, jit: 0.500,
            rSd: 1.0, mSc: 0.30, fSc: 0.80, sGn: 5.0, sAt: 0.08, _s0: 0.0, str: 0.15, aSc: 1.00,
            glw: 1.0, cSh: 0.05, spc: 0.25, gam: 2.0, cSp: 1.0, sat: 0.5, pal: 3.0, rel: 3.0,
            tur: 0.0, rct: 1.0, stm: 0.6, vis: 0.2, org: 0.5, osz: 0.4, shn: 0.4, ohu: 0.5,
            bgl: 0.45, fod: 0.6, ogl: 1.0, ovr: 0.7,
            act: 1.0, bth: 0.8, spk: 1.0, eml: 2.0,
            mit: 0.5, fcs: 0.0, apr: 0.0, fdr: 0.15,
            sdf: 0.0, edb: 0.0, srd: 0.35, fcx: 0.5,
            fcy: 0.5, mtr: 0.4, vac: 0.3, pak: 0.5,
            cal: 0.0, cyt: 1.0, tbe: 0.0, bgf: 1.0,
            tml: 6.0, imm: 0.0, swl: 0.0, aAB: -1.0,
            aBC: -1.0, aCA: -1.0, stn: 0.0, dvs: 0.0,
            prp: 0.0, _i0: 0.0, _i1: 0.0, _i2: 0.0,
        };

        let base = RenderKit::new(core);

        let mut passes = vec![
            PassDescription::new("agent_update", &["agent_update", "field", "flow"]).with_resolution(1024, 2048),
            PassDescription::new("force", &["flow"]).with_resolution_scale(0.5),
        ];
        for _ in 0..16 { passes.push(PassDescription::new("pressure", &["force", "pressure"]).with_resolution_scale(0.5)); }
        passes.extend([
            PassDescription::new("flow", &["force", "pressure"]).with_resolution_scale(0.5),
            PassDescription::new("trail_adv", &["field", "flow"]),
            PassDescription::new("diffuse_h", &["trail_adv"]),
            PassDescription::new("diffuse_v", &["diffuse_h"]),
            PassDescription::new("inhibitor_down", &["diffuse_v"]).with_resolution_scale(0.125),
            PassDescription::new("field", &["trail_adv", "diffuse_v", "inhibitor_down"]),
            PassDescription::new("organ", &["organ", "flow", "field"]).with_resolution(64, 192),
            PassDescription::new("organ_res", &["organ", "flow", "field"]),
            PassDescription::new("spark", &["spark", "organ", "field"]).with_resolution(128, 128),
            PassDescription::new("emit", &["organ_res", "field"]).with_resolution_scale(0.5),
            PassDescription::new("eb2", &["emit"]).with_resolution_scale(0.25),
            PassDescription::new("eb3", &["eb2"]).with_resolution_scale(0.125),
            PassDescription::new("eb4", &["eb3"]).with_resolution_scale(0.0625),
            PassDescription::new("eu3", &["eb3", "eb4"]).with_resolution_scale(0.125),
            PassDescription::new("eu2", &["eb2", "eu3"]).with_resolution_scale(0.25),
            PassDescription::new("eu1", &["emit", "eu2", "inhibitor_down"]).with_resolution_scale(0.5),
            PassDescription::new("cyto", &["cyto", "flow", "field"]),
            PassDescription::new("jinit", &["field"]),
            PassDescription::new("j64", &["jinit"]),
            PassDescription::new("j32", &["j64"]),
            PassDescription::new("j16", &["j32"]),
            PassDescription::new("j8", &["j16"]),
            PassDescription::new("j4", &["j8"]),
            PassDescription::new("j2", &["j4"]),
            PassDescription::new("j1", &["j2"]),
            PassDescription::new("rfield", &["field", "cyto", "j1"]),
            PassDescription::new("compose", &["rfield", "eu1", "organ_res"]),
            PassDescription::new("bd1", &["compose"]).with_resolution_scale(0.5),
            PassDescription::new("bd2", &["bd1"]).with_resolution_scale(0.25),
            PassDescription::new("bd3", &["bd2"]).with_resolution_scale(0.125),
            PassDescription::new("bd4", &["bd3"]).with_resolution_scale(0.0625),
            PassDescription::new("bd5", &["bd4"]).with_resolution_scale(0.03125),
            PassDescription::new("bu4", &["bd4", "bd5"]).with_resolution_scale(0.0625),
            PassDescription::new("bu3", &["bd3", "bu4"]).with_resolution_scale(0.125),
            PassDescription::new("bu2", &["bd2", "bu3"]).with_resolution_scale(0.25),
            PassDescription::new("bu1", &["bd1", "bu2"]).with_resolution_scale(0.5),
            PassDescription::new("db1", &["compose"]).with_resolution_scale(0.5),
            PassDescription::new("db2", &["db1"]).with_resolution_scale(0.25),
            PassDescription::new("db3", &["db2"]).with_resolution_scale(0.125),
            PassDescription::new("db4", &["db3"]).with_resolution_scale(0.0625),
            PassDescription::new("du3", &["db3", "db4"]).with_resolution_scale(0.125),
            PassDescription::new("du2", &["db2", "du3"]).with_resolution_scale(0.25),
            PassDescription::new("du1", &["db1", "du2"]).with_resolution_scale(0.5),
            PassDescription::new("main_image", &["compose", "bu1", "du1"]),
        ]);

        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_custom_uniforms::<PhysarumParams>()
            .with_atomic_buffer(6)
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
                            changed |= ui.add(egui::Slider::new(&mut params.rSd, 0.0..=100.0).step_by(1.0).text("Rule Seed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mSc, 0.0..=1.0).text("Species Diversity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.fSc, 0.0..=3.0).text("Force Scale")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sGn, 0.5..=30.0).text("Sensor Gain")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.str, 0.0..=2.0).text("Strafe Power")).changed();
                        });

                        egui::CollapsingHeader::new("Agent Physics").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.spd, 0.1..=10.0).text("Speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.drg, 0.0..=0.99).text("Momentum")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sd, 1.0..=80.0).text("Sensor Distance")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sa, 0.05..=1.5).text("Sensor Angle")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.jit, 0.0..=0.5).text("Random Jitter")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aSc, 0.05..=1.0).text("Agent Density")).changed();
                        });

                        egui::CollapsingHeader::new("Trail Environment").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.dep, 1.0..=60.0).text("Deposit Amount")).changed();
                            let mut hl = 0.5f32.ln() / (params.dec.ln() * 60.0);
                            if ui.add(egui::Slider::new(&mut hl, 0.1..=30.0).logarithmic(true).suffix(" s").text("Trail Half-life")).changed() { params.dec = 0.5f32.powf(1.0 / (hl * 60.0)); changed = true; }
                            changed |= ui.add(egui::Slider::new(&mut params.dif, 0.0..=1.0).text("Diffusion")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.tur, 0.0..=1.0).text("Turing Inhibitor")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.fod, 0.0..=2.0).text("Foraging")).changed();
                        });

                        egui::CollapsingHeader::new("Streaming").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.stm, 0.0..=3.0).text("Streaming")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.vis, 0.0..=0.95).text("Viscosity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.cyt, 0.0..=12.0).text("Cytoplasm")).changed();
                        });

                        egui::CollapsingHeader::new("Organelles").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.org, 0.0..=1.0).text("Amount")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.osz, 0.0..=1.0).text("Size")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.ohu, 0.0..=1.0).text("Hue (0.5 = complement)")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.ogl, 0.0..=3.0).text("Glow")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.ovr, 0.0..=1.0).text("Variety")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.act, 0.0..=3.0).text("Activity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.spk, 0.0..=3.0).text("Sparks")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mit, 0.0..=1.0).text("Mitosis")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mtr, 0.0..=1.0).text("Transport")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.vac, 0.0..=1.0).text("Vacuoles")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.pak, 0.0..=2.0).text("Packing")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.cal, 0.0..=1.0).text("Calcium Look")).changed();
                        });

                        egui::CollapsingHeader::new("Depth of Field").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.fcs, -1.0..=1.0).text("Focus Depth")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.fdr, 0.0..=0.6).text("Focus Range")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.apr, 0.0..=2.0).text("Blur Strength")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sdf, 0.0..=2.0).text("Scene Blur")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.edb, 0.0..=2.0).text("Edge Blur")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.srd, 0.0..=1.0).text("Sharp Radius")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.fcx, 0.0..=1.0).text("Focus X")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.fcy, 0.0..=1.0).text("Focus Y")).changed();
                        });

                        egui::CollapsingHeader::new("Species Interaction").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.sAt, -1.0..=1.0).text("Cross-Species (- repel, + attract)")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.rct, 0.0..=3.0).text("Cyclic Dominance")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.imm, 0.0..=3.0).text("Contact Strength (bubbles)")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.swl, 0.0..=3.0).text("Swirl")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aAB, -1.0..=1.0).text("Affinity A-B")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aBC, -1.0..=1.0).text("Affinity B-C")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aCA, -1.0..=1.0).text("Affinity C-A")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.stn, -1.0..=1.0).text("Surface Tension")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.dvs, 0.0..=2.0).text("Droplet Division")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.prp, 0.0..=2.0).text("Droplet Propulsion")).changed();
                        });

                        egui::CollapsingHeader::new("Rendering").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.rel, 0.0..=6.0).text("Vein Relief")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.tbe, 0.0..=1.0).text("Tube Shape")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.glw, 0.0..=3.0).text("Bloom")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.bth, 0.0..=3.0).text("Bloom Threshold")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.eml, 0.0..=5.0).text("Emitter Light")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.spc, 0.0..=1.5).text("Specular")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.shn, 0.0..=1.0).text("Gloss")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gam, 0.5..=3.0).text("Gamma")).changed();
                        });

                        egui::CollapsingHeader::new("Colors").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.pal, 0.0..=6.0).step_by(1.0).text("Palette")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.cSh, -0.5..=0.5).text("Hue Shift")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.cSp, 0.0..=2.0).text("Species Contrast")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sat, 0.0..=2.5).text("Saturation")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.bgl, 0.0..=3.0).text("Background")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.bgf, 0.0..=2.0).text("Background Life")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.tml, 0.5..=30.0).logarithmic(true).suffix(" s").text("Trail Memory")).changed();
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

        if controls_request.should_clear_buffers { self.compute_shader.clear_all_buffers(core); self.compute_shader.current_frame = 0; }
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