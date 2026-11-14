use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    struct ExperimentParams {
        arms: f32, twist: f32, bar: f32, core: f32,
        young: f32, nebula: f32, dust: f32, thickness: f32,
        anim: f32, orbit: f32, view: f32, zoom: f32,
        stars: f32, yaw: f32, bloom: f32, vignette: f32,
        brightness: f32, exposure: f32, gamma: f32, saturation: f32,
        hue: f32, spectral: f32, relief: f32, field: f32,
        suns: f32, halo: f32, ion: f32, taa_weight: f32,
        sharpen: f32, roll: f32, log_density: f32, clusters: f32,
        size: f32, seed: f32, noise_scale: f32, core_light: f32,
        sky_density: f32, sky_blue: f32, sky_clouds: f32, sky_hue: f32,
        prev_yaw: f32, prev_view: f32, prev_zoom: f32, prev_time: f32,
        star_glow: f32, bloom_threshold: f32, filaments: f32, filament_scale: f32,
        dust_hardness: f32, dust_rims: f32, lumpiness: f32, nebula2: f32,
        nebula_hue: f32, hii: f32, globulars: f32, multi_scatter: f32,
        distance: f32, prev_distance: f32, ovality: f32, jwst: f32,
        dust_light: f32, shadows: f32, cinematic: f32, core_temp: f32,
        lines: f32, hubble: f32, reddening: f32, aperture: f32,
        focus: f32, bokeh: f32, redshift: f32, doppler: f32,
        excitation: f32, balmer: f32, nii_sii: f32, line_vivid: f32,
        gas_flow: f32, cloud_evolve: f32, cloud_swirl: f32, twinkle: f32,
        bokeh_radius: f32, corotation: f32, lifetimes: f32, supernovae: f32,
        superbubbles: f32, highlight_hue: f32, focus_range: f32, bursts: f32,
        resolved: f32, sn_rate: f32, wind: f32, wind_height: f32,
        wind_opening: f32, pan_x: f32, pan_y: f32, pan_z: f32,
    }
}


// baked gas noise volumes (A: eddies + relief, B: cloud octaves)
const GAS_A: [u32; 3] = [96, 96, 24];
const GAS_B: [u32; 3] = [192, 192, 64];

struct ExperimentShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ExperimentParams,
    dragging: bool,
    prev_pose: (f32, f32, f32, f32),
    last_cursor: Option<(f64, f64)>,
}

impl ShaderManager for ExperimentShader {
    fn init(core: &Core) -> Self {
        let initial_params = ExperimentParams {
            arms: 2.0, twist: 1.0, bar: 0.35, core: 1.0,
            young: 1.0, nebula: 1.0, dust: 1.0, thickness: 1.0,
            anim: 1.0, orbit: 0.15, view: 0.75, zoom: 1.0,
            stars: 1.0, yaw: 0.0, bloom: 0.8, vignette: 0.3,
            brightness: 1.0, exposure: 1.0, gamma: 1.0, saturation: 1.1,
            hue: 0.5, spectral: 1.3, relief: 1.0, field: 1.0,
            suns: 1.0, halo: 1.0, ion: 1.0, taa_weight: 0.9,
            sharpen: 0.2, roll: 0.0, log_density: 0.6, clusters: 1.0,
            size: 1.0, seed: 0.0, noise_scale: 1.0, core_light: 1.0,
            sky_density: 0.6, sky_blue: 1.0, sky_clouds: 1.0, sky_hue: 0.0,
            prev_yaw: 0.0, prev_view: 0.75, prev_zoom: 1.0, prev_time: 0.0,
            star_glow: 1.0, bloom_threshold: 0.15, filaments: 1.0, filament_scale: 1.0,
            dust_hardness: 2.0, dust_rims: 1.0, lumpiness: 0.0, nebula2: 0.8,
            nebula_hue: 0.0, hii: 1.0, globulars: 1.0, multi_scatter: 1.0,
            distance: 3.2, prev_distance: 3.2, ovality: 0.0, jwst: 0.0,
            dust_light: 1.0, shadows: 1.0, cinematic: 0.0, core_temp: 4.5,
            lines: 1.0, hubble: 0.0, reddening: 1.0, aperture: 0.0,
            focus: 1.0, bokeh: 1.0, redshift: 0.0, doppler: 0.0,
            excitation: 0.0, balmer: 1.0, nii_sii: 1.0, line_vivid: 1.0,
            gas_flow: 0.0, cloud_evolve: 0.0, cloud_swirl: 0.0, twinkle: 0.0,
            bokeh_radius: 2.2, corotation: 0.65, lifetimes: 1.0, supernovae: 1.0,
            superbubbles: 1.0, highlight_hue: 0.6, focus_range: 0.0, bursts: 0.6,
            resolved: 1.0, sn_rate: 1.0, wind: 0.0, wind_height: 1.0,
            wind_opening: 0.45, pan_x: 0.0, pan_y: 0.0, pan_z: 0.0,
        };

        let base = RenderKit::new(core);
        let passes = vec![
            // gas noise bake (run_once)
            PassDescription::new("bake_ga", &[]).with_volume(GAS_A[0], GAS_A[1], GAS_A[2]).run_once(),
            PassDescription::new("bake_gb", &[]).with_volume(GAS_B[0], GAS_B[1], GAS_B[2]).run_once(),
            PassDescription::new("shadow", &[]).with_resolution_scale(0.25),
            PassDescription::new("scene", &["shadow"]),
            PassDescription::new("dof", &["scene"]),
            PassDescription::new("taa", &["dof", "taa"]),
            // bloom mip chain
            PassDescription::new("bloom_d1", &["taa"]).with_resolution_scale(0.5),
            PassDescription::new("bloom_d2", &["bloom_d1"]).with_resolution_scale(0.25),
            PassDescription::new("bloom_d3", &["bloom_d2"]).with_resolution_scale(0.125),
            PassDescription::new("bloom_d4", &["bloom_d3"]).with_resolution_scale(0.0625),
            PassDescription::new("bloom_d5", &["bloom_d4"]).with_resolution_scale(0.03125),
            PassDescription::new("bloom_d6", &["bloom_d5"]).with_resolution_scale(0.015625),
            PassDescription::new("bloom_u5", &["bloom_d6", "bloom_d5"]).with_resolution_scale(0.03125),
            PassDescription::new("bloom_u4", &["bloom_u5", "bloom_d4"]).with_resolution_scale(0.0625),
            PassDescription::new("bloom_u3", &["bloom_u4", "bloom_d3"]).with_resolution_scale(0.125),
            PassDescription::new("bloom_u2", &["bloom_u3", "bloom_d2"]).with_resolution_scale(0.25),
            PassDescription::new("bloom_u1", &["bloom_u2", "bloom_d1"]).with_resolution_scale(0.5),
            PassDescription::new("main_image", &["taa", "bloom_u1"]),
        ];
        let config = ComputeShader::builder()
            .with_entry_point("shadow")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<ExperimentParams>()
            .with_label("Spiral Galaxy")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/galaxy.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            dragging: false,
            prev_pose: (0.0, 0.75, 3.2, 0.0),
            last_cursor: None,
        }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = (current_time - self.prev_pose.3).clamp(0.0, 0.25);
        self.compute_shader.set_time(current_time, delta, &core.queue);
        // last frame's camera (TAA)
        let p = &mut self.current_params;
        (p.prev_yaw, p.prev_view, p.prev_distance, p.prev_time) = self.prev_pose;
        self.compute_shader.set_custom_params(*p, &core.queue);
        p.prev_zoom = p.zoom;
        self.prev_pose = (p.yaw, p.view, p.distance, current_time);
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

                egui::Window::new("Spiral Galaxy").collapsible(true).resizable(true).default_width(300.0).show(ctx, |ui| {
                    egui::CollapsingHeader::new("Galaxy").default_open(true).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("Spiral").clicked() { preset(&mut params, 0); changed = true; }
                            if ui.button("Barred").clicked() { preset(&mut params, 1); changed = true; }
                            if ui.button("Flocculent").clicked() { preset(&mut params, 2); changed = true; }
                            if ui.button("Lenticular").clicked() { preset(&mut params, 3); changed = true; }
                        });
                        changed |= ui.add(egui::Slider::new(&mut params.arms, 1.0..=6.0).step_by(1.0).text("Arms")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.twist, 0.2..=2.5).text("Winding")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.bar, 0.0..=1.0).text("Bar")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.core, 0.3..=2.5).text("Bulge")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.thickness, 0.2..=3.0).text("Disk thickness")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.seed, 0.0..=50.0).step_by(1.0).text("Seed")).changed();
                    });
                    egui::CollapsingHeader::new("Physics (density wave)").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.corotation, 0.3..=1.2).text("Corotation radius")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.lifetimes, 0.3..=5.0).logarithmic(true).text("Star lifetimes")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.bursts, 0.0..=1.0).text("Star-formation bursts")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.supernovae, 0.0..=3.0).text("Supernovae")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.sn_rate, 0.2..=4.0).text("Supernova rate")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.wind, 0.0..=3.0).text("Galactic wind")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.wind_height, 0.3..=2.0).text("Wind height")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.wind_opening, 0.15..=0.9).text("Wind opening")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.superbubbles, 0.0..=2.0).text("Superbubbles")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.gas_flow, 0.0..=3.0).text("Differential rotation (gas + stars)")).changed();
                    });
                    egui::CollapsingHeader::new("Gas").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.young, 0.0..=2.5).text("Young blue gas")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.lumpiness, 0.0..=1.0).text("Lumpiness")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.relief, 0.0..=2.0).text("Cloud relief")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.noise_scale, 0.4..=2.5).text("Cloud scale")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.nebula, 0.0..=3.0).text("Star-forming knots")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.ion, 0.0..=2.5).text("Ionised gas (teal, violet)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.nebula2, 0.0..=3.0).text("Diffuse nebula")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.nebula_hue, -2.0..=2.0).text("Diffuse nebula hue")).changed();
                    });
                    egui::CollapsingHeader::new("Dust").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.dust, 0.0..=3.0).text("Dust")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.filaments, 0.0..=3.0).text("Filaments")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.filament_scale, 0.4..=2.5).text("Filament scale")).changed();
                    });
                    egui::CollapsingHeader::new("Stars").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.stars, 0.0..=3.0).text("Resolved stars")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.resolved, 0.05..=1.0).text("Resolved star count")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.twinkle, 0.0..=1.0).text("Twinkle")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.hii, 0.0..=3.0).text("HII regions")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.clusters, 0.0..=3.0).text("Cluster light")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.globulars, 0.0..=3.0).text("Globular clusters")).changed();
                    });
                    egui::CollapsingHeader::new("Light").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.core_light, 0.0..=3.0).text("Core light")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.core_temp, 2.0..=15.0).text("Core colour (1000 K)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.shadows, 0.0..=3.0).text("Cloud shadows")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.multi_scatter, 0.0..=1.5).text("Light softness")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.dust_light, 0.0..=3.0).text("Starlight on clouds")).changed();
                    });
                    egui::CollapsingHeader::new("Spectrum").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.lines, 0.0..=1.0).text("Emission lines (spectral)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.redshift, -0.4..=1.0).text("Redshift z")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.doppler, 0.0..=3.0).text("Rotation Doppler")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.excitation, -1.0..=1.0).text("Excitation (H / O)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.balmer, 0.0..=3.0).text("Balmer H-beta")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.nii_sii, 0.0..=3.0).text("[NII] + [SII]")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.line_vivid, 0.0..=2.0).text("Line vividness")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.hubble, 0.0..=1.0).text("Hubble palette (SHO)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.spectral, 0.0..=2.5).text("Star colour vividness")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.reddening, 0.0..=2.5).text("Dust reddening")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.jwst, 0.0..=1.0).text("Infrared (JWST)")).changed();
                    });
                    egui::CollapsingHeader::new("Lens").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.aperture, 0.0..=3.0).text("Aperture (DoF)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.focus, 0.1..=4.0).logarithmic(true).text("Focus (1 = centre)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.focus_range, 0.0..=1.5).text("Focus range (sharp zone)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.bokeh_radius, 0.3..=6.0).text("Max blur (% height)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.bokeh, 0.0..=4.0).text("Bokeh highlights")).changed();
                    });
                    egui::CollapsingHeader::new("Background").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.sky_density, 0.0..=4.0).text("Star density")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.field, 0.0..=3.0).text("Star brightness")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.suns, 0.0..=3.0).text("Foreground stars")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.sky_blue, 0.0..=4.0).text("Blue")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.sky_clouds, 0.0..=4.0).text("Clouds")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.sky_hue, -1.0..=1.0).text("Cloud hue (teal/blue/violet)")).changed();
                    });
                    egui::CollapsingHeader::new("Motion & View").default_open(false).show(ui, |ui| {
                        ui.label("Drag to rotate, wheel to fly in and out");
                        changed |= ui.add(egui::Slider::new(&mut params.distance, 0.05..=8.0).logarithmic(true).text("Distance")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.anim, 0.0..=5.0).text("Rotation speed")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.orbit, -2.0..=2.0).text("Auto orbit")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.cinematic, 0.0..=2.0).text("Cinematic camera")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.pan_x, -2.0..=2.0).text("Camera X (left / right)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.pan_y, -2.0..=2.0).text("Camera Y (down / up)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.pan_z, -2.0..=2.0).text("Camera Z (back / forward)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.cloud_evolve, 0.0..=3.0).text("Cloud evolution")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.cloud_swirl, 0.0..=3.0).text("Cloud swirl")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.view, 0.0..=1.55).text("Inclination")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.roll, -1.5..=1.5).text("Roll")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.taa_weight, 0.0..=0.95).text("TAA")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.sharpen, 0.0..=1.0).text("Sharpen")).changed();
                    });
                    egui::CollapsingHeader::new("Post").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.brightness, 0.1..=4.0).text("Brightness")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.1..=4.0).text("Exposure")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.4..=2.2).text("Gamma")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.bloom, 0.0..=3.0).text("Cloud bloom")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.halo, 0.0..=2.0).text("Bloom halo")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.bloom_threshold, 0.0..=1.0).text("Bloom threshold")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.star_glow, 0.0..=8.0).text("Star bloom")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.log_density, 0.0..=1.0).text("Log density")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.highlight_hue, 0.0..=1.0).text("Highlight colour (hue-preserving)")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.saturation, 0.0..=2.0).text("Saturation")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.vignette, 0.0..=1.0).text("Vignette")).changed();
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
            let old = &self.current_params;
            if old.seed != params.seed || old.noise_scale != params.noise_scale || old.thickness != params.thickness {
                self.compute_shader.rerun_pass("bake_ga");
                self.compute_shader.rerun_pass("bake_gb");
            }
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
        if self.base.default_handle_input(core, event) {
            return true;
        }
        match event {
            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                self.dragging = *state == winit::event::ElementState::Pressed;
                self.last_cursor = None;
                self.dragging
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(d) => d.y as f32 / 40.0,
                };
                let p = &mut self.current_params;
                p.distance = (p.distance * 1.1f32.powf(-steps)).clamp(0.05, 8.0);
                self.compute_shader.set_custom_params(*p, &core.queue);
                true
            }
            WindowEvent::CursorMoved { position, .. } if self.dragging => {
                if let Some((x, y)) = self.last_cursor {
                    let p = &mut self.current_params;
                    p.yaw = (p.yaw - (position.x - x) as f32 * 0.006).rem_euclid(std::f32::consts::TAU);
                    p.view = (p.view - (position.y - y) as f32 * 0.005).clamp(0.0, 1.55);
                    self.compute_shader.set_custom_params(*p, &core.queue);
                }
                self.last_cursor = Some((position.x, position.y));
                true
            }
            _ => false,
        }
    }
}

// galaxy types (shape, gas, dust)
fn preset(p: &mut ExperimentParams, kind: u32) {
    let (arms, twist, bar, oval, core, thick, young, lump, knots, neb2, dust, fil, hii, clus) = match kind {
        1 => (2.0, 0.8, 0.9, 0.0, 1.1, 0.9, 1.1, 0.5, 1.0, 0.6, 1.3, 1.0, 1.0, 1.0),
        2 => (5.0, 1.4, 0.0, 0.0, 0.8, 1.2, 1.4, 0.95, 1.3, 1.2, 1.0, 1.4, 1.5, 1.3),
        3 => (2.0, 0.6, 0.2, 0.0, 2.0, 1.6, 0.1, 0.8, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0),
        _ => (2.0, 1.0, 0.35, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0),
    };
    (p.arms, p.twist, p.bar, p.ovality, p.core, p.thickness, p.young, p.lumpiness) = (arms, twist, bar, oval, core, thick, young, lump);
    (p.nebula, p.nebula2, p.dust, p.filaments, p.hii, p.clusters) = (knots, neb2, dust, fil, hii, clus);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Spiral Galaxy", 800, 600);
    app.run(event_loop, |core| ExperimentShader::init(core))
}
