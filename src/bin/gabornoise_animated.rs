use cuneus::{
    Core, Renderer, ShaderApp, ShaderManager, UniformProvider, UniformBinding, BaseShader,
    ExportSettings, ExportError, ExportManager, ShaderHotReload, ShaderControls,
};
use winit::event::*;
use image::ImageError;
use std::path::PathBuf;

//------------------------------------------------------------------------------
// The shader uniform parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    width: f32,
    height: f32,
    steps: f32,
    _pad1: f32,
    
    kernel_size: f32,
    num_kernels: f32,
    frequency: f32,
    frequency_var: f32,
    
    seed: f32,
    animation_speed: f32,
    gamma: f32,
    _pad2: f32,
}

impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

//------------------------------------------------------------------------------
// A structure to hold flags for each parameter indicating whether it should be animated.
#[derive(Copy, Clone, Debug)]
struct ParamAnimFlags {
    width: bool,
    height: bool,
    steps: bool,
    kernel_size: bool,
    num_kernels: bool,
    frequency: bool,
    frequency_var: bool,
    seed: bool,
    animation_speed: bool,
    gamma: bool,
}

impl Default for ParamAnimFlags {
    fn default() -> Self {
        Self {
            width: false,
            height: false,
            steps: false,
            kernel_size: false,
            num_kernels: false,
            frequency: false,
            frequency_var: false,
            seed: false,
            animation_speed: false,
            gamma: false,
        }
    }
}

//------------------------------------------------------------------------------
// Main entry point.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("gabornoise", 800, 600);
    let shader = SpiralShader::init(app.core());
    app.run(event_loop, shader)
}

//------------------------------------------------------------------------------
// Our shader struct.
struct SpiralShader {
    base: BaseShader,
    params_uniform: UniformBinding<ShaderParams>,
    hot_reload: ShaderHotReload,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    resolution_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    // Per-parameter animation flags.
    param_anim_flags: ParamAnimFlags,
}

impl SpiralShader {
    fn capture_frame(&mut self, core: &Core, time: f32) -> Result<Vec<u8>, wgpu::SurfaceError> {
        let settings = self.base.export_manager.settings();
        let (capture_texture, output_buffer) = self.base.create_capture_texture(
            &core.device,
            settings.width,
            settings.height,
        );
        let align = 256;
        let unpadded_bytes_per_row = settings.width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;
        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Capture Encoder"),
        });
        self.base.time_uniform.data.time = time;
        self.base.time_uniform.update(&core.queue);
        self.base.resolution_uniform.data.dimensions = [settings.width as f32, settings.height as f32];
        self.base.resolution_uniform.update(&core.queue);
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Capture Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &capture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            if self.base.using_video_texture {
                if let Some(video_manager) = &self.base.video_texture_manager {
                    render_pass.set_bind_group(0, &video_manager.texture_manager().bind_group, &[]);
                }
            } else if let Some(texture_manager) = &self.base.texture_manager {
                render_pass.set_bind_group(0, &texture_manager.bind_group, &[]);
            }
            // Time (group 1)
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            // Parameters (group 2)
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            // Resolution (group 3)
            render_pass.set_bind_group(3, &self.base.resolution_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(settings.height),
                },
            },
            wgpu::Extent3d {
                width: settings.width,
                height: settings.height,
                depth_or_array_layers: 1,
            },
        );
        core.queue.submit(Some(encoder.finish()));
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        core.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let padded_data = buffer_slice.get_mapped_range().to_vec();
        let mut unpadded_data = Vec::with_capacity((settings.width * settings.height * 4) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        Ok(unpadded_data)
    }

    #[allow(unused_mut)]
    fn save_frame(&self, mut data: Vec<u8>, frame: u32, settings: &ExportSettings) -> Result<(), ExportError> {
        let frame_path = settings.export_path.join(format!("frame_{:05}.png", frame));
        if let Some(parent) = frame_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        #[cfg(target_os = "macos")]
        {
            for chunk in data.chunks_mut(4) {
                chunk.swap(0, 2);
            }
        }
        let image = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
            settings.width,
            settings.height,
            data,
        ).ok_or_else(|| ImageError::Parameter(
            image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::Generic("Failed to create image buffer".to_string())
            )
        ))?;
        image.save(&frame_path)?;
        Ok(())
    }

    fn handle_export(&mut self, core: &Core) {
        if let Some((frame, time)) = self.base.export_manager.try_get_next_frame() {
            if let Ok(data) = self.capture_frame(core, time) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = self.save_frame(data, frame, settings) {
                    eprintln!("Error saving frame: {:?}", e);
                }
            }
        } else {
            self.base.export_manager.complete_export();
        }
    }
}

//------------------------------------------------------------------------------
// Implement the ShaderManager trait for SpiralShader.
impl ShaderManager for SpiralShader {
    fn init(core: &Core) -> Self {
        let time_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Uniform, 
                    has_dynamic_offset: false, 
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("time_bind_group_layout"),
        });
        let resolution_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Uniform, 
                    has_dynamic_offset: false, 
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("resolution_bind_group_layout"),
        });
        let params_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Uniform, 
                    has_dynamic_offset: false, 
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("params_bind_group_layout"),
        });
        let params_uniform = UniformBinding::new(
            &core.device,
            "Params Uniform",
            ShaderParams {
                width: 320.0,
                height: 240.0,
                steps: 12.0,
                _pad1: 0.0,
                kernel_size: 3.05,
                num_kernels: 48.0,
                frequency: 10.0,
                frequency_var: 33.5,
                seed: 12345.6789,
                animation_speed: 1.0,
                gamma: 1.0,
                _pad2: 0.0,
            },
            &params_bind_group_layout,
            0,
        );
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { 
                        multisampled: false, 
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });
        let bind_group_layouts = vec![
            &texture_bind_group_layout,    // group 0
            &time_bind_group_layout,       // group 1 
            &params_bind_group_layout,     // group 2
            &resolution_bind_group_layout, // group 3
        ];
        let vs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex.wgsl").into()),
        });
        let fs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/gabornoise.wgsl").into()),
        });
        let shader_paths = vec![
            PathBuf::from("shaders/vertex.wgsl"),
            PathBuf::from("shaders/gabornoise.wgsl"),
        ];
        let hot_reload = ShaderHotReload::new(
            core.device.clone(),
            shader_paths,
            vs_module,
            fs_module,
        ).expect("Failed to initialize hot reload");
        let base = BaseShader::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/gabornoise.wgsl"),
            &bind_group_layouts,
            None,
        );
        Self {
            base,
            params_uniform,
            hot_reload,
            texture_bind_group_layout,
            time_bind_group_layout,
            resolution_bind_group_layout,
            params_bind_group_layout,
            param_anim_flags: ParamAnimFlags::default(),
        }
    }

    fn update(&mut self, core: &Core) {
        if let Some((new_vs, new_fs)) = self.hot_reload.check_and_reload() {
            println!("Reloading shaders at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &self.texture_bind_group_layout,
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.resolution_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            self.base.renderer = Renderer::new(
                &core.device,
                new_vs,
                new_fs,
                core.config.format,
                &pipeline_layout,
                None,
            );
        }
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
        let current_time = self.base.controls.get_time(&self.base.start_time);
        // Global auto-animation enabled? Then update each parameter if its flag is true.
        if self.base.auto_animation {
            let mut params = self.params_uniform.data;
            if self.param_anim_flags.width {
                params.width = 320.0 + 50.0 * current_time.sin();
            }
            if self.param_anim_flags.height {
                params.height = 240.0 + 50.0 * current_time.cos();
            }
            if self.param_anim_flags.steps {
                params.steps = 12.0 + 6.0 * current_time.sin();
            }
            if self.param_anim_flags.kernel_size {
                params.kernel_size = 3.05 + 0.5 * current_time.cos();
            }
            if self.param_anim_flags.num_kernels {
                params.num_kernels = 48.0 + 10.0 * current_time.sin();
            }
            if self.param_anim_flags.frequency {
                params.frequency = 10.0 + 5.0 * current_time.sin();
            }
            if self.param_anim_flags.frequency_var {
                params.frequency_var = 33.5 + 10.0 * current_time.cos();
            }
            if self.param_anim_flags.seed {
                params.seed = 12345.6789 + current_time;
            }
            if self.param_anim_flags.animation_speed {
                params.animation_speed = 1.0 + 0.5 * current_time.sin();
            }
            if self.param_anim_flags.gamma {
                params.gamma = 1.0 + 0.2 * current_time.cos();
            }
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if self.base.using_video_texture {
            self.base.update_video_texture(core, &core.queue);
        }
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size);
        let using_video_texture = self.base.using_video_texture;
        let video_info = self.base.get_video_info();
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                egui::Window::new("Gabor Noise")
                    .collapsible(true)
                    .default_size([300.0, 100.0])
                    .show(ctx, |ui| {
                        ui.collapsing("Media", |ui: &mut egui::Ui| {
                            ShaderControls::render_media_panel(ui, &mut controls_request, using_video_texture, video_info);
                        });
                        ui.separator();
                        ui.collapsing("Resolution", |ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.width, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.width, 10.0..=640.0).text("Width")).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.height, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.height, 10.0..=480.0).text("Height")).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.steps, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.steps, 2.0..=24.0).text("Steps")).changed();
                            });
                        });
                        ui.collapsing("Gabor Noise", |ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.kernel_size, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.kernel_size, 0.1..=10.0).text("Kernel Size")).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.num_kernels, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.num_kernels, 1.0..=100.0).text("Num Kernels")).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.frequency, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.frequency, 1.0..=100.0).text("Frequency")).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.frequency_var, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.frequency_var, 0.0..=100.0).text("Freq Variation")).changed();
                            });
                        });
                        ui.collapsing("Animation", |ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.seed, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.seed, 1.0..=99999.0).text("Seed")).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.animation_speed, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.animation_speed, 0.0..=5.0).text("Anim Speed")).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.param_anim_flags.gamma, "Anim");
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.3..=1.5).text("Gamma")).changed();
                            });
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
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.update(&core.queue);
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            if self.base.using_video_texture {
                if let Some(video_manager) = &self.base.video_texture_manager {
                    render_pass.set_bind_group(0, &video_manager.texture_manager().bind_group, &[]);
                }
            } else if let Some(texture_manager) = &self.base.texture_manager {
                render_pass.set_bind_group(0, &texture_manager.bind_group, &[]);
            }
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            render_pass.set_bind_group(3, &self.base.resolution_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }
        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
        false
    }
}

