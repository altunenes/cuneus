use cuneus::{Core,Renderer,ShaderApp, ShaderManager, UniformProvider, UniformBinding, RenderKit,ExportManager,ShaderHotReload,ShaderControls};
use winit::event::*;
use std::path::PathBuf;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParams {
    base_color: [f32; 3],
    x: f32,
    rim_color: [f32; 3],
    y: f32,
    accent_color: [f32; 3],
    gamma_correction: f32,
    travel_speed: f32,
    iteration: i32,
    col_ext: f32,
    zoom: f32,
    trap_pow: f32,
    trap_x: f32,
    trap_y: f32,
    trap_c1: f32,
    aa: i32,
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
    base: RenderKit,
    params_uniform: UniformBinding<ShaderParams>,
    hot_reload: ShaderHotReload,
    time_bind_group_layout: wgpu::BindGroupLayout,    
    resolution_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    mouse_bind_group_layout: wgpu::BindGroupLayout,
    mouse_dragging: bool,
    drag_start: [f32; 2],
    drag_start_pos: [f32; 2],
    zoom_level: f32,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("orbits", 800, 600);
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
            if let Some(mouse_uniform) = &self.base.mouse_uniform {
                render_pass.set_bind_group(3, &mouse_uniform.bind_group, &[]);
            }
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
        let _ = core.device.poll(wgpu::PollType::Wait).unwrap();
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
        let mouse_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("mouse_bind_group_layout"),
        });

        let initial_zoom = 0.0004;
        let initial_x = 2.14278;
        let initial_y = 2.14278;

        let params_uniform = UniformBinding::new(
            &core.device,
            "Params Uniform",
            ShaderParams {
                base_color: [0.0, 0.5, 1.0],
                x: initial_x,
                rim_color: [0.0, 0.5, 1.0],
                y: initial_y,
                accent_color: [0.018, 0.018, 0.018],
                gamma_correction: 0.4,
                travel_speed: 1.0,
                iteration: 355,
                col_ext: 2.0,
                zoom: initial_zoom,
                trap_pow: 1.0,
                trap_x: -0.5,
                trap_y: 2.0,
                trap_c1: 0.2,
                aa: 1,
                trap_s1: 0.8,
                wave_speed: 0.1,
                fold_intensity: 1.0,
            },
            &params_bind_group_layout,
            0,
        );

        let bind_group_layouts = vec![
            &time_bind_group_layout,
            &resolution_bind_group_layout,
            &params_bind_group_layout,
            &mouse_bind_group_layout,
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

        let mut base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/orbits.wgsl"),
            &bind_group_layouts,
            None,
        );
        base.setup_mouse_uniform(core);

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
            mouse_bind_group_layout,
            mouse_dragging: false,
            drag_start: [0.0, 0.0],
            drag_start_pos: [initial_x, initial_y],
            zoom_level: initial_zoom,
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
                    &self.mouse_bind_group_layout,
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
        self.base.update_mouse_uniform(&core.queue);
        self.base.fps_tracker.update();
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
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });                
                egui::Window::new("Orbits")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        
                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Base:");
                                    changed |= ui.color_edit_button_rgb(&mut params.base_color).changed();
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Orbit:");
                                    changed |= ui.color_edit_button_rgb(&mut params.rim_color).changed();
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Exterior:");
                                    changed |= ui.color_edit_button_rgb(&mut params.accent_color).changed();
                                });
                            });

                        egui::CollapsingHeader::new("Rendering")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.iteration, 50..=500).text("Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.aa, 1..=4).text("Anti-aliasing")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma_correction, 0.1..=2.0).text("Gamma")).changed();
                            });

                        egui::CollapsingHeader::new("Traps")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.trap_x, -5.0..=5.0).text("Trap X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.trap_y, -5.0..=5.0).text("Trap Y")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.trap_pow, 0.0..=3.0).text("Trap Power")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.trap_c1, 0.0..=1.0).text("Trap Mix")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.trap_s1, 0.0..=2.0).text("Trap Blend")).changed();
                            });

                        egui::CollapsingHeader::new("Animation")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.travel_speed, 0.0..=2.0).text("Travel Speed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.wave_speed, 0.0..=2.0).text("Wave Speed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.fold_intensity, 0.0..=3.0).text("Fold Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.col_ext, 0.0..=10.0).text("Color Extension")).changed();
                            });

                        egui::CollapsingHeader::new("Navigation")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Left-click + drag: Pan view");
                                ui.label("Mouse wheel: Zoom");
                                ui.separator();
                                let old_zoom = params.zoom;
                                changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.0001..=1.0).text("Zoom").logarithmic(true)).changed();
                                if old_zoom != params.zoom {
                                    self.zoom_level = params.zoom;
                                }
                                changed |= ui.add(egui::Slider::new(&mut params.x, 0.0..=3.0).text("X Position")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.y, 0.0..=6.0).text("Y Position")).changed();
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
            render_pass.set_bind_group(1, &self.base.resolution_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            if let Some(mouse_uniform) = &self.base.mouse_uniform {
                render_pass.set_bind_group(3, &mouse_uniform.bind_group, &[]);
            }
            
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
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                match button {
                    MouseButton::Left => {
                        match state {
                            ElementState::Pressed => {
                                if let Some(mouse_uniform) = &self.base.mouse_uniform {
                                    self.mouse_dragging = true;
                                    self.drag_start = mouse_uniform.data.position;
                                    self.drag_start_pos = [self.params_uniform.data.x, self.params_uniform.data.y];
                                }
                                return true;
                            },
                            ElementState::Released => {
                                self.mouse_dragging = false;
                                return true;
                            }
                        }
                    },
                    _ => {}
                }
                false
            },
            WindowEvent::CursorMoved { .. } => {
                if self.mouse_dragging {
                    if let Some(mouse_uniform) = &self.base.mouse_uniform {
                        let current_pos = mouse_uniform.data.position;
                         let dx = (current_pos[0] - self.drag_start[0]) * 3.0 * self.zoom_level;
                         let dy = (current_pos[1] - self.drag_start[1]) * 6.0 * self.zoom_level;
                        let mut new_x = self.drag_start_pos[0] + dx;
                        let mut new_y = self.drag_start_pos[1] + dy;
                        new_x = new_x.clamp(0.0, 3.0);
                        new_y = new_y.clamp(0.0, 6.0);
                        self.params_uniform.data.x = new_x;
                        self.params_uniform.data.y = new_y;
                        self.params_uniform.update(&core.queue);
                    }
                }
                self.base.handle_mouse_input(core, event, false)
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let zoom_delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y * 0.1,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.001,
                };
                
                if zoom_delta != 0.0 {
                    if let Some(mouse_uniform) = &self.base.mouse_uniform {
                        let mouse_pos = mouse_uniform.data.position;
                        
                        let center_x = self.params_uniform.data.x;
                        let center_y = self.params_uniform.data.y;
                        
                        let rel_x = mouse_pos[0] - 0.5;
                        let rel_y = mouse_pos[1] - 0.5;
                        
                        let zoom_factor = if zoom_delta > 0.0 { 0.9 } else { 1.1 };
                        self.zoom_level = (self.zoom_level * zoom_factor).clamp(0.0001, 1.5);
                        
                        let scale_change = 1.0 - zoom_factor;
                        let dx = rel_x * scale_change * 3.0 * self.zoom_level;
                        let dy = rel_y * scale_change * 6.0 * self.zoom_level;
                        self.params_uniform.data.zoom = self.zoom_level;
                        self.params_uniform.data.x = (center_x + dx).clamp(0.0, 3.0);
                        self.params_uniform.data.y = (center_y + dy).clamp(0.0, 6.0);
                        self.params_uniform.update(&core.queue);
                    }
                }
                self.base.handle_mouse_input(core, event, false)
            },
            
            _ => self.base.handle_mouse_input(core, event, false),
        }
    }
}