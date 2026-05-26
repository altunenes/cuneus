use cuneus::{Core,Renderer,ShaderApp, ShaderManager, UniformProvider, UniformBinding, RenderKit,ExportManager,ShaderHotReload,ShaderControls,RemoteCommand,RemoteControl};
use winit::event::*;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParams {
    square_size: f32, 
    circle_radius: f32,
    edge_thickness: f32,
    animation_speed: f32,
    
    background_color: [f32; 3],
    edge_color_intensity: f32,
    
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,

}

impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct Shader {
    base: RenderKit,
    params_uniform: UniformBinding<ShaderParams>,
    hot_reload: ShaderHotReload,
    time_bind_group_layout: wgpu::BindGroupLayout,    
    resolution_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    remote_control: Option<RemoteControl>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("rotation_illusion", 800, 600);
    app.run(event_loop, |core| {
        Shader::init(core)
    })
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
            render_pass.set_bind_group(0, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(1, &self.base.resolution_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
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

    fn handle_export(&mut self, core: &Core) {
        if let Some((frame, time)) = self.base.export_manager.try_get_next_frame() {
            if let Ok(data) = self.capture_frame(core, time) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = cuneus::save_frame(data, frame, settings) {
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

        // Initialize with the original ShaderToy values
        let params_uniform = UniformBinding::new(
            &core.device,
            "Params Uniform",
            ShaderParams {
                square_size: 0.2,
                circle_radius: 0.11,
                edge_thickness: 0.003,
                animation_speed: 12.0,
                
                background_color: [0.5, 0.5, 0.5],
                edge_color_intensity: 1.0,
                
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            },
            &params_bind_group_layout,
            0,
        );

        let bind_group_layouts = vec![
            &time_bind_group_layout,
            &resolution_bind_group_layout,
            &params_bind_group_layout,
        ];
        let vs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex.wgsl").into()),
        });

        let fs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/roto.wgsl").into()),
        });

        let shader_paths = vec![
            PathBuf::from("shaders/vertex.wgsl"),
            PathBuf::from("shaders/roto.wgsl"),
        ];

        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/roto.wgsl"),
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
            resolution_bind_group_layout,
            params_bind_group_layout,
            remote_control: RemoteControl::from_env(),
        }
    }

    fn update(&mut self, core: &Core) {
        if let Some((new_vs, new_fs)) = self.hot_reload.check_and_reload() {
            println!("Reloading shaders at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout, 
                    &self.resolution_bind_group_layout,  
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
        self.base.fps_tracker.update();
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut params = self.params_uniform.data;
        let mut changed = false;
        if let Some(remote_control) = &self.remote_control {
            for command in remote_control.drain() {
                match command {
                    RemoteCommand::SetF32 { id, value } => {
                        changed |= apply_remote_f32(&mut params, &id, value);
                    }
                    RemoteCommand::SetColor3 { id, value } => {
                        if id == "background_color" {
                            params.background_color = value;
                            changed = true;
                        }
                    }
                    RemoteCommand::Pulse { .. } |
                    RemoteCommand::Note { .. } |
                    RemoteCommand::Transport { .. } => {}
                }
            }
        }
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });
                egui::Window::new("Rotation Illusion").show(ctx, |ui| {
                    changed |= ui.add(egui::Slider::new(&mut params.square_size, 0.05..=0.5)
                        .text("Square Size")).changed();
                    
                    changed |= ui.add(egui::Slider::new(&mut params.circle_radius, 0.05..=0.2)
                        .text("Circle Radius")).changed();
                    
                    changed |= ui.add(egui::Slider::new(&mut params.edge_thickness, 0.001..=0.01)
                        .text("Edge Thickness")).changed();
                    
                    changed |= ui.add(egui::Slider::new(&mut params.animation_speed, 1.0..=30.0)
                        .text("Animation Speed")).changed();
                    
                    changed |= ui.color_edit_button_rgb(&mut params.background_color).changed();
                    ui.label("Background Color");
                    
                    changed |= ui.add(egui::Slider::new(&mut params.edge_color_intensity, 0.1..=2.0)
                        .text("Edge Brightness")).changed();

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
            render_pass.set_bind_group(1, &self.base.resolution_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
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

fn apply_remote_f32(params: &mut ShaderParams, id: &str, value: f32) -> bool {
    match id {
        "square_size" => params.square_size = value.clamp(0.05, 0.5),
        "circle_radius" => params.circle_radius = value.clamp(0.05, 0.2),
        "edge_thickness" => params.edge_thickness = value.clamp(0.001, 0.01),
        "animation_speed" => params.animation_speed = value.clamp(1.0, 30.0),
        "edge_color_intensity" => params.edge_color_intensity = value.clamp(0.1, 2.0),
        _ => return false,
    }
    true
}
