use cuneus::compute::{PassDescription, StorageBufferSpec, COMPUTE_TEXTURE_FORMAT_RGBA16};
use cuneus::prelude::ComputeShader;
use cuneus::WindowEvent;
use cuneus::{Core, ExportManager, RenderKit, ShaderApp, ShaderControls, ShaderManager};

cuneus::uniform_params! {
    struct ShaderParams {
        n: u32, colored: u32, invert: u32, relax: f32,
        gamma: f32, pin_h: f32, yaw: f32, elev: f32,
        zoom: f32, light_az: f32, light_el: f32, rough: f32,
        pin_r: f32, expo: f32, gam: f32, bloom: f32,
        orb_spd: f32, light_spd: f32, wave_amp: f32, wave_spd: f32,
        lan_n: f32, lan_pow: f32, lan_spd: f32, key: f32,
        focus: f32, aperture: f32, bokeh_hi: f32, blades: f32,
        refl: f32, coat: f32, reseed: u32, glow: f32,
        metal: f32, aniso: f32, flake: f32, dome: f32,
        shape: u32, studs: f32, plates: f32, palette: u32,
        twinkle: f32, holo: f32, fl_size: f32, fl_dens: f32,
        fl_all: u32, _f0: f32, _f1: f32, _f2: f32,
    }
}

// must match the shader: most points, grid side
const MAXN: u32 = 131_072;
const GRID: u32 = 1024;

// largest window the pins' HDR buffer holds
const MAXPIX: u64 = 3840 * 2400;

struct Pinscreen {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ShaderParams,
    reseed: bool,
    // the other mode's finish; pins start chrome, bricks plastic
    other_finish: [f32; 8],
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Pinscreen", 900, 900);
    app.run(event_loop, Pinscreen::init)
}

impl ShaderManager for Pinscreen {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);
        let current_params = ShaderParams {
            n: 2_000, colored: 1, invert: 0, relax: 0.8,
            gamma: 1.0, pin_h: 40.0, yaw: 20.0, elev: 35.0,
            zoom: 1.0, light_az: 135.0, light_el: 40.0, rough: 0.25,
            pin_r: 1.0, expo: 1.0, gam: 1.0, bloom: 0.6,
            orb_spd: 0.0, light_spd: 0.0, wave_amp: 0.0, wave_spd: 2.0,
            lan_n: 4.0, lan_pow: 1.0, lan_spd: 1.0, key: 0.7,
            focus: 0.0, aperture: 0.0, bokeh_hi: 1.0, blades: 0.0,
            refl: 1.0, coat: 0.5, reseed: 1, glow: 0.0,
            metal: 1.0, aniso: 0.6, flake: 0.0, dome: 0.45,
            shape: 0, studs: 64.0, plates: 9.0, palette: 1,
            twinkle: 0.5, holo: 0.0, fl_size: 1.0, fl_dens: 0.6,
            fl_all: 0, _f0: 0.0, _f1: 0.0, _f2: 0.0,
        };

        // point and grid passes have fixed sizes; smaller screen passes skip what's outside
        let points = [MAXN / 256, 1, 1];
        let cells = [GRID * GRID / 256, 1, 1];
        let mut passes = vec![
            PassDescription::new("init_points", &[]).with_workgroup_size(points),
            PassDescription::new("clear_grid", &[]).with_workgroup_size(cells),
            PassDescription::new("seed", &[]).with_workgroup_size(points),
        ];
        passes.extend((0..10).map(|k| PassDescription::new(&format!("jfa_{k}"), &[]).with_workgroup_size(cells)));
        passes.push(PassDescription::new("accumulate", &[]).with_workgroup_size(cells));
        passes.push(PassDescription::new("relax", &[]).with_workgroup_size(points));
        // up to 256 x 256 studs
        passes.push(PassDescription::new("brick_cells", &[]).with_workgroup_size([256, 1, 1]));
        for name in ["pins_hdr", "dof_pre", "dof_gather", "dof_post", "bloom_down", "bloom_h", "bloom_v", "main_image"] {
            passes.push(PassDescription::new(name, &[]));
        }

        let grid = (GRID * GRID) as u64 * 4;
        let config = ComputeShader::builder()
            .with_entry_point("init_points")
            .with_multi_pass(&passes)
            .with_input_texture()
            .with_custom_uniforms::<ShaderParams>()
            .with_storage_buffer(StorageBufferSpec::new("pts", MAXN as u64 * 32))
            .with_storage_buffer(StorageBufferSpec::new("ga", grid))
            .with_storage_buffer(StorageBufferSpec::new("gb", grid))
            .with_storage_buffer(StorageBufferSpec::new("acc", MAXN as u64 * 7 * 4))
            .with_storage_buffer(StorageBufferSpec::new("hdr", MAXPIX * 8))
            .with_storage_buffer(StorageBufferSpec::new("b1", MAXPIX / 16 * 8 + 64))
            .with_storage_buffer(StorageBufferSpec::new("b2", MAXPIX / 16 * 8 + 64))
            .with_storage_buffer(StorageBufferSpec::new("dof", MAXPIX * 8))
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Pinscreen")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/pinscreen.wgsl", config);
        compute_shader.set_custom_params(current_params, &core.queue);

        Self { base, compute_shader, current_params, reseed: true, other_finish: [0.0, 0.3, 0.0, 0.3, 0.5, 0.45, 0.0, 0.0], dragging: false, last_cursor: None }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(current_time, 1.0 / 60.0, &core.queue);

        self.base.update_current_texture(core, &core.queue);
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            self.compute_shader.update_input_texture(&texture_manager.view, &texture_manager.sampler, &core.device);
        }
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        if self.base.using_video_texture {
            self.base.update_video_texture(core, &core.queue);
        }
        if self.base.using_webcam_texture {
            self.base.update_webcam_texture(core, &core.queue);
        }

        let mut params = self.current_params;
        let mut changed = false;
        let mut reseed = false;
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
                egui::Window::new("Pinscreen")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
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
                        ui.horizontal(|ui| {
                            for (i, name) in ["Pins", "Bricks"].iter().enumerate() {
                                changed |= ui.selectable_value(&mut params.shape, i as u32, *name).changed();
                            }
                        });
                        let pins = params.shape == 0;
                        if pins {
                            egui::CollapsingHeader::new("Points").default_open(true).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.n, 1_000..=MAXN).logarithmic(true).text("Pins")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.relax, 0.05..=1.0).text("Relax speed")).changed();
                                let mut colored = params.colored != 0;
                                if ui.checkbox(&mut colored, "Image colours").changed() {
                                    params.colored = colored as u32;
                                    changed = true;
                                }
                                if ui.button("Scatter again").clicked() {
                                    reseed = true;
                                }
                            });
                        } else {
                            egui::CollapsingHeader::new("Bricks").default_open(true).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.studs, 16.0..=256.0).step_by(1.0).text("Studs across")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.plates, 1.0..=24.0).step_by(1.0).text("Tallest stack (plates)")).changed();
                                ui.horizontal(|ui| {
                                    for (i, name) in ["Image colours", "LEGO", "LEGO dithered"].iter().enumerate() {
                                        changed |= ui.selectable_value(&mut params.palette, i as u32, *name).changed();
                                    }
                                });
                            });
                        }
                        egui::CollapsingHeader::new("Relief").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.3..=3.0).text("Tone curve")).changed();
                            let mut invert = params.invert != 0;
                            if ui.checkbox(&mut invert, "Tall where light").changed() {
                                params.invert = invert as u32;
                                changed = true;
                            }
                            if pins {
                                changed |= ui.add(egui::Slider::new(&mut params.pin_h, 0.0..=150.0).text("Height")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.pin_r, 0.3..=1.3).text("Pin size")).changed();
                            }
                        });
                        egui::CollapsingHeader::new("Camera & light").default_open(true).show(ui, |ui| {
                            ui.label("Drag to orbit, wheel to zoom");
                            changed |= ui.add(egui::Slider::new(&mut params.yaw, -180.0..=180.0).text("Orbit")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.elev, 5.0..=85.0).text("Tilt")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.5..=4.0).logarithmic(true).text("Zoom")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.light_az, -180.0..=180.0).text("Light angle")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.light_el, 5.0..=85.0).text("Light height")).changed();
                        });
                        egui::CollapsingHeader::new("Finish").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.metal, 0.0..=1.0).text("Metal")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.rough, 0.04..=1.0).text("Roughness")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aniso, 0.0..=1.0).text("Spun grooves")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.coat, 0.0..=1.0).text("Clear coat")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.refl, 0.0..=1.0).text("Mirror neighbours")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.dome, 0.05..=1.0).text(if pins { "Hat height" } else { "Stud height" })).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.flake, 0.0..=2.0).text("Flakes")).changed();
                            if params.flake > 0.0 {
                                changed |= ui.add(egui::Slider::new(&mut params.fl_size, 0.2..=4.0).logarithmic(true).text("Flake size")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.fl_dens, 0.05..=1.0).text("Flake density")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.twinkle, 0.0..=2.0).text("Twinkle")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.holo, 0.0..=1.0).text("Holo colour")).changed();
                                let mut all = params.fl_all != 0;
                                if ui.checkbox(&mut all, "Flakes everywhere").changed() {
                                    params.fl_all = all as u32;
                                    changed = true;
                                }
                            }
                            changed |= ui.add(egui::Slider::new(&mut params.glow, 0.0..=2.0).text("Tip glow")).changed();
                        });
                        egui::CollapsingHeader::new("Lanterns").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.lan_n, 0.0..=32.0).step_by(1.0).text("Lanterns")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.lan_pow, 0.0..=4.0).text("Power")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.lan_spd, 0.0..=4.0).text("Pace")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.key, 0.0..=1.5).text("Key light")).changed();
                        });
                        egui::CollapsingHeader::new("Depth of field").show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.aperture, 0.0..=3.0).text("Aperture")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.focus, -0.5..=0.5).text("Focus")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.bokeh_hi, 0.0..=4.0).text("Highlights")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.blades, 0.0..=8.0).step_by(1.0).text("Blades (0 round)")).changed();
                        });
                        egui::CollapsingHeader::new("Animation").show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.orb_spd, -30.0..=30.0).text("Orbit speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.light_spd, -60.0..=60.0).text("Light speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.wave_amp, 0.0..=1.0).text("Wave")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.wave_spd, 0.0..=8.0).text("Wave speed")).changed();
                        });
                        egui::CollapsingHeader::new("Post").show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.expo, 0.2..=4.0).logarithmic(true).text("Exposure")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gam, 0.4..=3.0).text("Gamma")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.bloom, 0.0..=3.0).text("Bloom")).changed();
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

        if controls_request.load_media_path.is_some() || controls_request.start_webcam {
            self.reseed = true;
        }
        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_media_requests(core, &controls_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }
        if params.shape != self.current_params.shape {
            let p = &mut params;
            let cur = [p.metal, p.rough, p.aniso, p.coat, p.refl, p.dome, p.flake, p.glow];
            [p.metal, p.rough, p.aniso, p.coat, p.refl, p.dome, p.flake, p.glow] = self.other_finish;
            self.other_finish = cur;
        }
        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }
        self.reseed |= reseed;

        // the init pass only scatters on a frame with this flag up
        if self.current_params.reseed != self.reseed as u32 {
            self.current_params.reseed = self.reseed as u32;
            self.compute_shader.set_custom_params(self.current_params, &core.queue);
        }
        self.reseed = false;
        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);
        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if let WindowEvent::DroppedFile(_) = event {
            self.reseed = true;
        }
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
                p.zoom = (p.zoom * 1.1f32.powf(steps)).clamp(0.5, 4.0);
                self.compute_shader.set_custom_params(*p, &core.queue);
                true
            }
            WindowEvent::CursorMoved { position, .. } if self.dragging => {
                if let Some((x, y)) = self.last_cursor {
                    let p = &mut self.current_params;
                    p.yaw = (p.yaw + (position.x - x) as f32 * 0.3 + 180.0).rem_euclid(360.0) - 180.0;
                    p.elev = (p.elev + (position.y - y) as f32 * 0.2).clamp(5.0, 85.0);
                    self.compute_shader.set_custom_params(*p, &core.queue);
                }
                self.last_cursor = Some((position.x, position.y));
                true
            }
            _ => false,
        }
    }
}
