use cuneus::compute::{
    ComputeShader, PassDescription, StorageBufferSpec, COMPUTE_TEXTURE_FORMAT_RGBA16};
use cuneus::{wgpu, Core, ExportManager, RenderKit, ShaderControls, ShaderManager};
use log::error;
use cuneus::WindowEvent;
use std::ops::RangeInclusive;

cuneus::uniform_params! {
    struct FFTParams {
    filter_type: i32,
    cutoff: f32,
    order: f32,
    band_center: f32,
    band_octaves: f32,
    orientation: f32,
    orient_width: f32,
    view: i32,
    resolution: u32,
    is_bw: i32,
    window: i32,
    keep_mean: i32,
    spec_decades: f32,
    show_radial: i32,
    taper: f32,
    phase_amount: f32,
    phase_seed: u32,
    slope_on: i32,
    slope_target: f32,
    _padding: u32}
}

// the stats buffer: radial profiles, then the fits (alpha_in, b_in, alpha_out, b_out) at this float
const S_FIT: u64 = 2048;

// pipeline stages, in dispatch order
const MEAN: usize = 0;
const INIT: usize = 1;
const FFT_H: usize = 2;
const FFT_V: usize = 3;
const RADIAL_IN: usize = 4;
const FIT_IN: usize = 5;
const MODIFY: usize = 6;
const RADIAL_OUT: usize = 7;
const FIT_OUT: usize = 8;
const IFFT_H: usize = 9;
const IFFT_V: usize = 10;
const MAIN: usize = 11;

struct FFTShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    should_initialize: bool,
    current_params: FFTParams,
    // 1/f fits read back from the GPU: alpha of the input and of the result
    alpha: [f32; 2],
    fit_staging: wgpu::Buffer,
    fit_pending: bool,
}

impl FFTShader {
    fn read_fit(&mut self, core: &Core) {
        let slice = self.fit_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = core.device.poll(wgpu::PollType::wait_indefinitely());
        if let Ok(Ok(())) = rx.recv() {
            if let Ok(data) = slice.get_mapped_range() {
                let fit: &[f32] = bytemuck::cast_slice(&data);
                self.alpha = [fit[0], fit[2]];
            }
        }
        self.fit_staging.unmap();
    }
}

fn slider(ui: &mut egui::Ui, v: &mut f32, range: RangeInclusive<f32>, label: &str) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

fn toggle(ui: &mut egui::Ui, v: &mut i32, label: &str) -> bool {
    let mut on = *v != 0;
    let changed = ui.checkbox(&mut on, label).changed();
    *v = on as i32;
    changed
}

impl ShaderManager for FFTShader {
    fn init(core: &Core) -> Self {
        let initial_params = FFTParams {
            filter_type: 2,
            cutoff: 32.0,
            order: 4.0,
            band_center: 16.0,
            band_octaves: 1.0,
            orientation: 0.0,
            orient_width: 30.0,
            view: 0,
            resolution: 1024,
            is_bw: 0,
            window: 1,
            keep_mean: 1,
            spec_decades: 5.0,
            show_radial: 1,
            taper: 0.1,
            phase_amount: 0.0,
            phase_seed: 1,
            slope_on: 0,
            slope_target: 1.0,
            _padding: 0};
        let base = RenderKit::new(core);

        let passes = vec![
            PassDescription::new("image_mean", &[]),
            PassDescription::new("initialize_data", &["image_mean"]),
            PassDescription::new("fft_horizontal", &["initialize_data"]),
            PassDescription::new("fft_vertical", &["fft_horizontal"]),
            PassDescription::new("radial_in", &["fft_vertical"]),
            PassDescription::new("fit_in", &["radial_in"]),
            PassDescription::new("modify_frequencies", &["fit_in"]),
            PassDescription::new("radial_out", &["modify_frequencies"]),
            PassDescription::new("fit_out", &["radial_out"]),
            PassDescription::new("ifft_horizontal", &["fit_out"]),
            PassDescription::new("ifft_vertical", &["ifft_horizontal"]),
            PassDescription::new("main_image", &["ifft_vertical"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("image_mean")
            .with_multi_pass(&passes)
            .with_input_texture()
            .with_custom_uniforms::<FFTParams>()
            // FFT working memory at the largest resolution, 3 complex channels
            .with_storage_buffer(StorageBufferSpec::new("image_data", 2048 * 2048 * 3 * 8))
            .with_storage_buffer(StorageBufferSpec::new("stats", 4096 * 4))
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("FFT Multi-Pass")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/fft.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        let fit_staging = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT fit readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            base,
            compute_shader,
            should_initialize: true,
            current_params: initial_params,
            alpha: [0.0; 2],
            fit_staging,
            fit_pending: false}
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // the fit written by the last FFT run
        if self.fit_pending {
            self.read_fit(core);
            self.fit_pending = false;
        }

        // Update input textures for image proc.
        self.base.update_current_texture(core, &core.queue);
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            self.compute_shader.update_input_texture(
                &texture_manager.view,
                &texture_manager.sampler,
                &core.device,
            );
        }
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
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
        let alpha = self.alpha;
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("fourier workflow")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(260.0)
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
                            for (i, name) in ["Filtered", "Original", "Spectrum"].iter().enumerate() {
                                changed |= ui.selectable_value(&mut params.view, i as i32, *name).changed();
                            }
                        });
                        ui.label(format!("Amplitude slope α: input {:.2}, result {:.2}", alpha[0], alpha[1]));

                        let half = (params.resolution / 2) as f32;
                        egui::CollapsingHeader::new("Filter").default_open(true).show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                for (i, name) in ["None", "Low-pass", "High-pass", "Band-pass", "Orientation"].iter().enumerate() {
                                    changed |= ui.selectable_value(&mut params.filter_type, i as i32, *name).changed();
                                }
                            });
                            match params.filter_type {
                                1 | 2 => {
                                    changed |= ui.add(egui::Slider::new(&mut params.cutoff, 1.0..=half).logarithmic(true).text("Cutoff").suffix(" c/img")).changed();
                                    changed |= slider(ui, &mut params.order, 1.0..=10.0, "Order");
                                }
                                3 => {
                                    changed |= ui.add(egui::Slider::new(&mut params.band_center, 1.0..=half).logarithmic(true).text("Centre").suffix(" c/img")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.band_octaves, 0.25..=4.0).text("Width").suffix(" oct")).changed();
                                }
                                4 => {
                                    changed |= ui.add(egui::Slider::new(&mut params.orientation, 0.0..=180.0).text("Orientation").suffix("°")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.orient_width, 5.0..=90.0).text("Width").suffix("°")).changed();
                                }
                                _ => {}
                            }
                            changed |= toggle(ui, &mut params.keep_mean, "Keep mean luminance");
                        });

                        egui::CollapsingHeader::new("Phase & amplitude").show(ui, |ui| {
                            changed |= slider(ui, &mut params.phase_amount, 0.0..=1.0, "Phase scramble");
                            ui.horizontal(|ui| {
                                ui.label(format!("Seed {}", params.phase_seed));
                                if ui.button("New").clicked() {
                                    params.phase_seed = params.phase_seed.wrapping_add(1);
                                    changed = true;
                                }
                            });
                            changed |= toggle(ui, &mut params.slope_on, "Set 1/f slope");
                            if params.slope_on != 0 {
                                changed |= ui.add(egui::Slider::new(&mut params.slope_target, 0.0..=3.0).text("Amplitude α (0 = whitened)")).changed();
                            }
                        });

                        egui::CollapsingHeader::new("Analysis").show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Resolution");
                                for n in [256u32, 512, 1024, 2048] {
                                    changed |= ui.radio_value(&mut params.resolution, n, n.to_string()).changed();
                                }
                            });
                            changed |= toggle(ui, &mut params.window, "Window image edges");
                            if params.window != 0 {
                                changed |= slider(ui, &mut params.taper, 0.02..=0.5, "Taper");
                            }
                            changed |= toggle(ui, &mut params.show_radial, "Radial amplitude plot");
                            changed |= slider(ui, &mut params.spec_decades, 2.0..=8.0, "Spectrum decades");
                            changed |= toggle(ui, &mut params.is_bw, "Black & White");
                        });

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

        self.base.apply_media_requests(core, &controls_request);

        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        if controls_request.load_media_path.is_some() || controls_request.start_webcam {
            self.should_initialize = true;
        }

        if changed {
            params.cutoff = params.cutoff.min((params.resolution / 2) as f32);
            params.band_center = params.band_center.min((params.resolution / 2) as f32);
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
            self.should_initialize = true;
        }

        // a still image runs once per change; video and webcam every frame
        let run = self.should_initialize || self.base.using_video_texture || self.base.using_webcam_texture;
        let n = params.resolution;
        if run && self.base.get_current_texture_manager().is_some() {
            let tiles = [n.div_ceil(16), n.div_ceil(16), 1];
            let cs = &mut self.compute_shader;
            let enc = &mut frame.encoder;
            cs.dispatch_stage_with_workgroups(enc, MEAN, [1, 1, 1]);
            cs.dispatch_stage_with_workgroups(enc, INIT, tiles);
            cs.dispatch_stage_with_workgroups(enc, FFT_H, [n, 1, 1]);
            cs.dispatch_stage_with_workgroups(enc, FFT_V, [n, 1, 1]);
            cs.dispatch_stage_with_workgroups(enc, RADIAL_IN, [n / 2, 1, 1]);
            cs.dispatch_stage_with_workgroups(enc, FIT_IN, [1, 1, 1]);
            cs.dispatch_stage_with_workgroups(enc, MODIFY, tiles);
            cs.dispatch_stage_with_workgroups(enc, RADIAL_OUT, [n / 2, 1, 1]);
            cs.dispatch_stage_with_workgroups(enc, FIT_OUT, [1, 1, 1]);
            // the spectrum view shows the modified spectrum, so the inverse is only needed for the picture
            if params.view == 0 {
                cs.dispatch_stage_with_workgroups(enc, IFFT_H, [n, 1, 1]);
                cs.dispatch_stage_with_workgroups(enc, IFFT_V, [n, 1, 1]);
            }
            enc.copy_buffer_to_buffer(&cs.storage_buffers[1], S_FIT * 4, &self.fit_staging, 0, 16);
            self.fit_pending = true;
            self.should_initialize = false;
        }

        self.compute_shader.dispatch_stage(&mut frame.encoder, core, MAIN);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if let WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                error!("Failed to load dropped file: {e:?}");
            } else {
                self.should_initialize = true;
            }
            return true;
        }
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("FFT", 800, 600);
    app.run(event_loop, FFTShader::init)
}
