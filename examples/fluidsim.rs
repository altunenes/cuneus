// Navier-Stokes Fluid Simulation
// Ported from Pavel Dobryakov's WebGL Fluid Simulation
// https://github.com/PavelDoGreat/WebGL-Fluid-Simulation MIT License
//
// With this example I m trying to demonstrate Cuneus's dispatch_stage() capability for running
// multiple simulation steps per frame, including 20+ pressure iterations. 
// Note that, this shader is storage-buffer based rather than texture based. Because we can easily loop dispatch stage with workgrpus 20 times
// for pressure solving without not 20 separate pressure passes like prs 01 2,3, ... 20.

use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

const SIM_SCALE: u32 = 2;
const PRESSURE_ITERATIONS: u32 = 20;
const INTERNAL_WIDTH: u32 = 2048;
const INTERNAL_HEIGHT: u32 = 1152;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FluidParams {
    sim_width: u32,
    sim_height: u32,
    display_width: u32,
    display_height: u32,
    dt: f32,
    time: f32,
    velocity_dissipation: f32,
    density_dissipation: f32,
    pressure: f32,
    curl_strength: f32,
    splat_radius: f32,
    splat_x: f32,
    splat_y: f32,
    splat_dx: f32,
    splat_dy: f32,
    splat_force: f32,
    splat_color_r: f32,
    splat_color_g: f32,
    splat_color_b: f32,
    // ping-pong tracking for each field
    vel_ping: u32, // which velocity buffer to READ from
    prs_ping: u32, // which pressure buffer to READ from
    dye_ping: u32, // which dye buffer to READ from
    do_splat: u32,
    _pad: u32,
}

impl UniformProvider for FluidParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct FluidSim {
    base: RenderKit,
    compute_shader: ComputeShader,
    params: FluidParams,
    prev_mouse_pos: [f32; 2],
    mouse_initialized: bool,
    current_color: [f32; 3],
    color_timer: f32,
    last_time: std::time::Instant,
    first_frame: bool,
    needs_clear: bool,
}

impl ShaderManager for FluidSim {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);


        let sim_width = INTERNAL_WIDTH / SIM_SCALE;
        let sim_height = INTERNAL_HEIGHT / SIM_SCALE;
        let sim_cells = (sim_width * sim_height) as u64;
        let dye_cells = (INTERNAL_WIDTH * INTERNAL_HEIGHT) as u64;

        // all at fixed internal resolution
        let velocity_size = sim_cells * 2 * 4;
        let pressure_size = sim_cells * 4;
        let divergence_size = sim_cells * 4;
        let curl_size = sim_cells * 4;
        let dye_size = dye_cells * 4 * 4;

        let total_size = velocity_size * 2 + pressure_size * 2 + divergence_size + curl_size + dye_size * 2;

        let passes = vec![
            PassDescription::new("clear_buffers", &[]), // 0
            PassDescription::new("splat_velocity", &[]), // 1
            PassDescription::new("splat_dye", &[]), // 2
            PassDescription::new("curl_compute", &[]), // 3
            PassDescription::new("vorticity_apply", &[]), // 4
            PassDescription::new("divergence_compute", &[]), // 5
            PassDescription::new("pressure_clear", &[]), // 6
            PassDescription::new("pressure_iterate", &[]), // 7
            PassDescription::new("gradient_subtract", &[]), // 8
            PassDescription::new("advect_velocity", &[]), // 9
            PassDescription::new("advect_dye", &[]), // 10
            PassDescription::new("main_image", &[]), // 11
        ];

        // Note: We don't use .with_mouse() because cuneus MouseTracker doesn't provide
        // velocity (dx/dy). Fluid simulation needs velocity for force injection, so we
        // track mouse manually and pass data through FluidParams.
        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_custom_uniforms::<FluidParams>()
            .with_storage_buffer(StorageBufferSpec::new("fluid_data", total_size))
            .with_label("Fluid Simulation")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/fluidsim.wgsl", config);

        let params = FluidParams {
            sim_width,
            sim_height,
            display_width: INTERNAL_WIDTH,
            display_height: INTERNAL_HEIGHT,
            dt: 1.0 / 60.0,
            time: 0.0,
            velocity_dissipation: 0.2,
            density_dissipation: 1.0,
            pressure: 0.8,
            curl_strength: 30.0,
            splat_radius: 0.25,
            splat_x: 0.0,
            splat_y: 0.0,
            splat_dx: 0.0,
            splat_dy: 0.0,
            splat_force: 6000.0,
            splat_color_r: 0.0,
            splat_color_g: 0.0,
            splat_color_b: 0.0,
            vel_ping: 0,
            prs_ping: 0,
            dye_ping: 0,
            do_splat: 0,
            _pad: 0,
        };

        compute_shader.set_custom_params(params, &core.queue);

        Self {
            base,
            compute_shader,
            params,
            prev_mouse_pos: [0.5, 0.5],
            mouse_initialized: false,
            current_color: Self::generate_color(),
            color_timer: 0.0,
            last_time: std::time::Instant::now(),
            first_frame: true,
            needs_clear: true,
        }
    }

    fn update(&mut self, core: &Core) {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_time).as_secs_f32();
        self.last_time = now;
        let dt = dt.min(1.0 / 30.0);

        self.params.time += dt;
        self.params.dt = dt;

        self.color_timer += dt * 10.0;
        if self.color_timer >= 1.0 {
            self.color_timer = 0.0;
            self.current_color = Self::generate_color();
        }


        let current_mouse_pos = self.base.mouse_tracker.uniform.position;
        let mouse_down = (self.base.mouse_tracker.uniform.buttons[0] & 1) != 0; // Left button

        if mouse_down {
            if !self.mouse_initialized {
                self.prev_mouse_pos = current_mouse_pos;
                self.mouse_initialized = true;
            }

            let mut dx = current_mouse_pos[0] - self.prev_mouse_pos[0];
            let mut dy = current_mouse_pos[1] - self.prev_mouse_pos[1];

            let aspect = core.size.width as f32 / core.size.height as f32;
            if aspect < 1.0 {
                dx *= aspect;
            } else {
                dy /= aspect;
            }

            self.params.splat_x = current_mouse_pos[0];
            self.params.splat_y = current_mouse_pos[1];
            self.params.splat_dx = dx * self.params.splat_force;
            self.params.splat_dy = dy * self.params.splat_force;
            self.params.splat_color_r = self.current_color[0];
            self.params.splat_color_g = self.current_color[1];
            self.params.splat_color_b = self.current_color[2];
            self.params.do_splat = if dx.abs() > 0.0001 || dy.abs() > 0.0001 { 1 } else { 0 };

            self.prev_mouse_pos = current_mouse_pos;
        } else if self.first_frame {
            self.first_frame = false;
            let color = Self::generate_color();
            self.params.splat_x = 0.5;
            self.params.splat_y = 0.5;
            self.params.splat_dx = 500.0;
            self.params.splat_dy = 300.0;
            self.params.splat_color_r = color[0] * 10.0;
            self.params.splat_color_g = color[1] * 10.0;
            self.params.splat_color_b = color[2] * 10.0;
            self.params.do_splat = 1;
        } else {
            self.params.do_splat = 0;
            self.mouse_initialized = false;
        }

        self.compute_shader.check_hot_reload(&core.device);
        self.compute_shader.handle_export(core, &mut self.base);
        self.base.fps_tracker.update();
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        // Update params
        self.compute_shader.set_custom_params(self.params, &core.queue);

        let sim_workgroups = [
            self.params.sim_width.div_ceil(16),
            self.params.sim_height.div_ceil(16),
            1,
        ];
        let display_workgroups = [
            self.params.display_width.div_ceil(16),
            self.params.display_height.div_ceil(16),
            1,
        ];
        let output_workgroups = [
            core.size.width.div_ceil(16),
            core.size.height.div_ceil(16),
            1,
        ];

        // Stage indices
        const CLEAR_BUFFERS: usize = 0;
        const SPLAT_VELOCITY: usize = 1;
        const SPLAT_DYE: usize = 2;
        const CURL_COMPUTE: usize = 3;
        const VORTICITY_APPLY: usize = 4;
        const DIVERGENCE_COMPUTE: usize = 5;
        const PRESSURE_CLEAR: usize = 6;
        const PRESSURE_ITERATE: usize = 7;
        const GRADIENT_SUBTRACT: usize = 8;
        const ADVECT_VELOCITY: usize = 9;
        const ADVECT_DYE: usize = 10;
        const MAIN_IMAGE: usize = 11;


        if self.needs_clear {
            self.needs_clear = false;
            let max_workgroups = [
                self.params.display_width.div_ceil(16),
                self.params.display_height.div_ceil(16),
                1,
            ];
            self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, CLEAR_BUFFERS, max_workgroups);
            frame.encoder = core.flush_encoder(frame.encoder);
            self.compute_shader.set_custom_params(self.params, &core.queue);
        }

        // Apply splat (additive, in-place on current read buffer)
        if self.params.do_splat == 1 {
            self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, SPLAT_VELOCITY, sim_workgroups);
            self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, SPLAT_DYE, display_workgroups);
        }

        // Curl: reads vel[vel_ping]
        self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, CURL_COMPUTE, sim_workgroups);

        // Vorticity: reads vel[vel_ping], writes vel[1-vel_ping]
        self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, VORTICITY_APPLY, sim_workgroups);

        // Submit before changing ping
        frame.encoder = core.flush_encoder(frame.encoder);
        self.params.vel_ping = 1 - self.params.vel_ping;
        self.compute_shader.set_custom_params(self.params, &core.queue);

        // Divergence: reads vel[vel_ping] (where vorticity wrote)
        self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, DIVERGENCE_COMPUTE, sim_workgroups);

        // Pressure clear: reads prs[prs_ping], writes prs[1-prs_ping]
        self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, PRESSURE_CLEAR, sim_workgroups);

        frame.encoder = core.flush_encoder(frame.encoder);
        self.params.prs_ping = 1 - self.params.prs_ping;
        self.compute_shader.set_custom_params(self.params, &core.queue);

        // Jacobi solver
        for _ in 0..PRESSURE_ITERATIONS {
            self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, PRESSURE_ITERATE, sim_workgroups);
            frame.encoder = core.flush_encoder(frame.encoder);
            self.params.prs_ping = 1 - self.params.prs_ping;
            self.compute_shader.set_custom_params(self.params, &core.queue);
        }

        // Gradient subtract: reads vel[vel_ping], prs[prs_ping], writes vel[1-vel_ping]
        self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, GRADIENT_SUBTRACT, sim_workgroups);

        frame.encoder = core.flush_encoder(frame.encoder);
        self.params.vel_ping = 1 - self.params.vel_ping;
        self.compute_shader.set_custom_params(self.params, &core.queue);

        // Advect velocity: reads vel[vel_ping], writes vel[1-vel_ping]
        self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, ADVECT_VELOCITY, sim_workgroups);

        frame.encoder = core.flush_encoder(frame.encoder);
        self.params.vel_ping = 1 - self.params.vel_ping;
        self.compute_shader.set_custom_params(self.params, &core.queue);

        // Advect dye: reads vel[vel_ping], dye[dye_ping], writes dye[1-dye_ping]
        self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, ADVECT_DYE, display_workgroups);

        frame.encoder = core.flush_encoder(frame.encoder);
        self.params.dye_ping = 1 - self.params.dye_ping;
        self.compute_shader.set_custom_params(self.params, &core.queue);

        // Display: reads dye[dye_ping]
        self.compute_shader.dispatch_stage_with_workgroups(&mut frame.encoder, MAIN_IMAGE, output_workgroups);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        let mut params = self.params;
        let mut should_clear = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });

                egui::Window::new("Fluid Simulation")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Fluid Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add(egui::Slider::new(&mut params.curl_strength, 0.0..=50.0).text("Vorticity"));
                                ui.add(egui::Slider::new(&mut params.velocity_dissipation, 0.0..=4.0).text("Vel Dissipation"));
                                ui.add(egui::Slider::new(&mut params.density_dissipation, 0.0..=4.0).text("Dye Dissipation"));
                                ui.add(egui::Slider::new(&mut params.pressure, 0.0..=1.0).text("Pressure"));
                            });

                        egui::CollapsingHeader::new("Splat Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.add(egui::Slider::new(&mut params.splat_radius, 0.01..=1.0).text("Radius"));
                                ui.add(egui::Slider::new(&mut params.splat_force, 1000.0..=20000.0).text("Force"));
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);

                        ui.separator();
                        if ui.button("Clear Fluid").clicked() {
                            should_clear = true;
                        }
                        ui.label(format!("Internal: {}x{}", INTERNAL_WIDTH, INTERNAL_HEIGHT));
                        ui.label("Drag mouse to add fluid");
                    });
            })
        } else {
            self.base.render_ui(core, |_| {})
        };

        if controls_request.should_clear_buffers || should_clear {
            self.params.vel_ping = 0;
            self.params.prs_ping = 0;
            self.params.dye_ping = 0;
            self.first_frame = true;
            self.needs_clear = true;
        }

        // Apply UI changes
        self.base.apply_control_request(controls_request);
        self.base.export_manager.apply_ui_request(export_request);
        self.params = params;

        if should_start_export {
            self.base.export_manager.start_export();
        }

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }
        if self.base.handle_mouse_input(core, event, false) {
            if let WindowEvent::MouseInput { state, button, .. } = event {
                if *button == winit::event::MouseButton::Left && state.is_pressed() {
                    self.current_color = Self::generate_color();
                }
            }
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }

        false
    }
}

impl FluidSim {
    fn generate_color() -> [f32; 3] {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let state = RandomState::new();
        let mut hasher = state.build_hasher();
        hasher.write_u64(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64);
        let h = (hasher.finish() as f32) / (u64::MAX as f32);

        let i = (h * 6.0).floor() as i32;
        let f = h * 6.0 - i as f32;
        let q = 1.0 - f;
        let t = f;

        let (r, g, b) = match i % 6 {
            0 => (1.0, t, 0.0),
            1 => (q, 1.0, 0.0),
            2 => (0.0, 1.0, t),
            3 => (0.0, q, 1.0),
            4 => (t, 0.0, 1.0),
            _ => (1.0, 0.0, q),
        };

        [r * 0.2, g * 0.2, b * 0.2]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Fluid Sim", 1024, 768);
    app.run(event_loop, FluidSim::init)
}