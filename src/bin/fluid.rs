use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, BaseShader, TextureManager, create_feedback_texture_pair};
use winit::event::WindowEvent;
use cuneus::ShaderApp;
use cuneus::Renderer;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeUniform {
    time: f32,
    frame: u32,
}
impl UniformProvider for TimeUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FluidParams {
    rotation_speed: f32,
    motor_strength: f32,
    distortion: f32,
    feedback: f32,
}
impl UniformProvider for FluidParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
struct FluidShader {
    base: BaseShader,
    renderer_pass2: Renderer,
    time_uniform: UniformBinding<TimeUniform>,
    params_uniform: UniformBinding<FluidParams>,
    texture_a: Option<TextureManager>,
    texture_b: Option<TextureManager>,
    input_texture: Option<TextureManager>,
    frame_count: u32,
}
impl ShaderManager for FluidShader {
    fn init(core: &Core) -> Self {
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
        let input_texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("input_texture_bind_group_layout"),
        });
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
        let time_uniform = UniformBinding::new(
            &core.device,
            "Time Uniform",
            TimeUniform {
                time: 0.0,
                frame: 0,
            },
            &time_bind_group_layout,
            0,
        );
        let params_uniform = UniformBinding::new(
            &core.device,
            "Params Uniform",
            FluidParams {
                rotation_speed: 1.0,
                motor_strength: 0.01,
                distortion: 10.0,
                feedback: 0.95,
            },
            &params_bind_group_layout,
            0,
        );

        let (texture_a, texture_b) = create_feedback_texture_pair(
            core,
            core.config.width,
            core.config.height,
            &texture_bind_group_layout,
        );
        let vs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex.wgsl").into()),
        });
        let fs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fluid.wgsl").into()),
        });
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &input_texture_bind_group_layout,
                &time_bind_group_layout,
                &params_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let base = BaseShader::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/fluid.wgsl"),
            &[
                &texture_bind_group_layout,
                &input_texture_bind_group_layout,
                &time_bind_group_layout,
                &params_bind_group_layout,
            ],
            Some("fs_pass1"),
        );
        let renderer_pass2 = Renderer::new(
            &core.device,
            &vs_module,
            &fs_module,
            core.config.format,
            &pipeline_layout,
            Some("fs_pass2"),
        );
        Self {
            base,
            renderer_pass2,
            time_uniform,
            params_uniform,
            texture_a: Some(texture_a),
            texture_b: Some(texture_b),
            input_texture: None,
            frame_count: 0,
        }
    }
    fn update(&mut self, core: &Core) {
        self.time_uniform.data.time = self.base.start_time.elapsed().as_secs_f32();
        self.time_uniform.data.frame = self.frame_count;
        self.time_uniform.update(&core.queue);
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        let mut reload_image = false;
        let mut selected_path = None;
        // Local copies of parameters to fight the borrow checker
        let mut params = self.params_uniform.data;
        let mut changed = false;

        let full_output = {
            let base = &mut self.base;
            base.render_ui(core, |ctx| {
                egui::Window::new("Fluid Settings").show(ctx, |ui| {
                    if ui.button("Load Image").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Image", &["png", "jpg", "jpeg"])
                            .pick_file() 
                        {
                            selected_path = Some(path);
                            reload_image = true;
                        }
                    }
                    changed |= ui.add(egui::Slider::new(&mut params.rotation_speed, 0.1..=5.0).text("Rotation Speed")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.motor_strength, 0.001..=0.1).text("Motor Strength")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.distortion, 1.0..=20.0).text("Distortion")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.feedback, 0.0..=1.3).text("Feedback")).changed();
                });
            })
        };
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }

        if reload_image {
            if let Some(path) = selected_path {
                self.base.load_image(core, path);
            }
        }
        if let (Some(ref texture_a), Some(ref texture_b)) = (&self.texture_a, &self.texture_b) {
            let (source_texture, target_texture) = if self.frame_count % 2 == 0 {
                (texture_b, texture_a)
            } else {
                (texture_a, texture_b)
            };
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Fluid Pass 1"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_texture.view,
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
                render_pass.set_bind_group(0, &source_texture.bind_group, &[]);
                if let Some(ref input_texture) = self.input_texture {
                    render_pass.set_bind_group(1, &input_texture.bind_group, &[]);
                } else if let Some(ref default_texture) = self.base.texture_manager {
                    render_pass.set_bind_group(1, &default_texture.bind_group, &[]);
                }
                render_pass.set_bind_group(2, &self.time_uniform.bind_group, &[]);
                render_pass.set_bind_group(3, &self.params_uniform.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Fluid Pass 2"),
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
                render_pass.set_pipeline(&self.renderer_pass2.render_pipeline);
                render_pass.set_vertex_buffer(0, self.renderer_pass2.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &target_texture.bind_group, &[]);
                if let Some(ref input_texture) = self.input_texture {
                    render_pass.set_bind_group(1, &input_texture.bind_group, &[]);
                } else if let Some(ref default_texture) = self.base.texture_manager {
                    render_pass.set_bind_group(1, &default_texture.bind_group, &[]);
                }
                render_pass.set_bind_group(2, &self.time_uniform.bind_group, &[]);
                render_pass.set_bind_group(3, &self.params_uniform.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
            self.frame_count = self.frame_count.wrapping_add(1);
        }
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.egui_state.on_window_event(core.window(), event).consumed
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Fluid Shader", 800, 600);
    let shader = FluidShader::init(app.core());
    app.run(event_loop, shader)
}