use cuneus::{Core,Renderer,ShaderApp, ShaderManager, UniformProvider, UniformBinding, BaseShader,ExportSettings, ExportError, ExportManager,ShaderHotReload,ShaderControls};
use winit::event::*;
use image::ImageError;
use std::path::PathBuf;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParams {
    // Colors
    base_color: [f32; 3],
    x: f32,
    rim_color: [f32; 3],
    y: f32,
    accent_color: [f32; 3],
    _pad3: f32,
    _pad4: f32,
    // xy vec2
    
    // Lighting parameters
    iteration: i32,
    col_ext: f32,
    zoom: f32,
    trap_pow: f32,
    
    // Effect parameters
    trap_x: f32,
    trap_y: f32,
    trap_c1: f32,
    aa: i32,


    // Animation parameters
    trap_s1: f32,
    wave_speed: f32,
    fold_intensity: f32,
}


impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct Shader {
    base: BaseShader,
    params_uniform: UniformBinding<ShaderParams>,
    hot_reload: ShaderHotReload,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Cuneus", 800, 600);
    let shader = Shader::init(app.core());
    app.run(event_loop, shader)
}
impl Shader {
    fn capture_frame(&mut self, core: &Core, time: f32) -> Result<Vec<u8>, wgpu::SurfaceError> {
        let settings = self.base.export_manager.settings();
        let (capture_texture, output_buffer) = self.base.create_capture_texture(
            &core.device,
            settings.width,
            settings.height
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
            render_pass.set_bind_group(0, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
        encoder.copy_texture_to_buffer( 
            wgpu::ImageCopyTexture {
                texture: &capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
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

    fn save_frame(&self, mut data: Vec<u8>, frame: u32, settings: &ExportSettings) -> Result<(), ExportError> {
        let frame_path = settings.export_path
            .join(format!("frame_{:05}.png", frame));
        
        if let Some(parent) = frame_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Convert BGRA to RGBA
        for chunk in data.chunks_mut(4) {
            chunk.swap(0, 2);
        }

        let image = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
            settings.width,
            settings.height,
            data
        ).ok_or_else(|| ImageError::Parameter(
            image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::Generic(
                    "Failed to create image buffer".to_string()
                )
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
impl ShaderManager for Shader {
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
                base_color: [0.0, 0.5, 1.0],
                x: 2.14278,
                rim_color: [0.0, 0.5, 1.0],
                y: 2.14278,
                accent_color: [0.018, 0.018, 0.018],
                _pad3: 0.0,
                _pad4: 0.0,
                iteration: 220,
                col_ext: 2.0,
                zoom: 0.0004,
                trap_pow: 2.0,

                
                trap_x: -0.5,
                trap_y: 2.0,
                trap_c1: 0.13,
                aa: 1,
                
                trap_s1: 2.0,
                wave_speed: 0.5,
                fold_intensity: 1.0,
            },
            &params_bind_group_layout,
            0,
        );

        let bind_group_layouts = vec![
            &time_bind_group_layout,
            &params_bind_group_layout,
        ];
        let vs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex.wgsl").into()),
        });

        let fs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/orbits.wgsl").into()),
        });

        let shader_paths = vec![
            PathBuf::from("shaders/vertex.wgsl"),
            PathBuf::from("shaders/orbits.wgsl"),
        ];

        let base = BaseShader::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/orbits.wgsl"),
            &bind_group_layouts,
            None,
        );

        let hot_reload = ShaderHotReload::new(
            core.device.clone(),
            shader_paths,
            vs_module,
            fs_module,
        ).expect("Failed to initialize hot reload");

        Self {
            base,
            params_uniform,
            hot_reload,
            time_bind_group_layout,
            params_bind_group_layout,
        }
    }

    fn update(&mut self, core: &Core) {
        if let Some((new_vs, new_fs)) = self.hot_reload.check_and_reload() {
            println!("Reloading shaders at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
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
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                egui::Window::new("Mandelbrot").show(ctx, |ui| {
                    // Colors
                    changed |= ui.color_edit_button_rgb(&mut params.base_color).changed();
                    ui.label("Base Color");
                    
                    changed |= ui.color_edit_button_rgb(&mut params.rim_color).changed();
                    ui.label("Orbit Color");

                    changed |= ui.color_edit_button_rgb(&mut params.accent_color).changed();
                    ui.label("Ext Color");
                    
                    
                    changed |= ui.add(egui::Slider::new(&mut params.iteration, 1..=500)
                        .text("iteration")).changed();


                        

                        
                    changed |= ui.add(egui::Slider::new(&mut params.wave_speed, 0.0..=12.0)
                        .text("cols")).changed();
                        
                    changed |= ui.add(egui::Slider::new(&mut params.fold_intensity, 0.0..=6.0)
                        .text("escape")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.aa, 1..=8)
                    .text("AA(care!)")).changed(); 


                    ui.collapsing("Trap", |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.trap_x, -12.0..=12.0)
                        .text("trap_x")).changed();
                        
                    changed |= ui.add(egui::Slider::new(&mut params.trap_y, -12.0..=12.0)
                        .text("trap_y")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.col_ext, 0.0..=25.0)
                    .text("c1")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.trap_pow, 0.0..=10.0)
                    .text("Trap Power")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.trap_c1, 0.0..=6.2)
                    .text("c1")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.trap_s1, 0.0..=6.2)
                    .text("s1")).changed();
                    });


                    ui.collapsing("Possitions", |ui| {

                    changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.0001..=1.5)
                        .text("Zoom")).changed();
                    
                    changed |= ui.add(egui::Slider::new(&mut params.x, 0.0..=3.0)
                        .text("X")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.y, 0.0..=6.0)
                        .text("Y")).changed();
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
            render_pass.set_bind_group(0, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
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
