use crate::{Core, ShaderManager};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowAttributes,
};

pub struct ShaderApp {
    window_title: String,
    window_size: (u32, u32),
    core: Option<Core>,
}

impl ShaderApp {
    pub fn new(window_title: &str, width: u32, height: u32) -> (Self, EventLoop<()>) {
        let event_loop = EventLoop::builder()
            .build()
            .expect("Failed to create event loop");

        //note: No window creation here - will happen in resumed event
        let app = Self {
            window_title: String::from(window_title),
            window_size: (width, height),
            core: None,
        };

        (app, event_loop)
    }

    pub fn run<S: ShaderManager + 'static>(
        self,
        event_loop: EventLoop<()>,
        shader_creator: impl FnOnce(&Core) -> S + 'static,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut handler = ShaderAppHandler {
            app: self,
            shader_creator: Some(Box::new(shader_creator)),
            shader: None,
            first_render: true,
        };

        Ok(event_loop.run_app(&mut handler)?)
    }

    pub fn core(&self) -> Option<&Core> {
        self.core.as_ref()
    }
}

// This struct implements ApplicationHandler to handle winit events
struct ShaderAppHandler<S: ShaderManager> {
    app: ShaderApp,
    shader_creator: Option<Box<dyn FnOnce(&Core) -> S + 'static>>,
    shader: Option<S>,
    first_render: bool,
}

impl<S: ShaderManager> ApplicationHandler for ShaderAppHandler<S> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = WindowAttributes::default()
            .with_inner_size(LogicalSize::new(
                self.app.window_size.0,
                self.app.window_size.1,
            ))
            .with_title(&self.app.window_title)
            .with_resizable(true);
        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        window.set_window_level(winit::window::WindowLevel::AlwaysOnTop);
        let core = pollster::block_on(Core::new(window));
        // Initialize the shader with the core if it hasn't been initialized yet
        if let Some(shader_creator) = self.shader_creator.take() {
            let shader = shader_creator(&core);
            self.shader = Some(shader);
        }

        self.app.core = Some(core);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Only process events if core and shader are initialized
        if let (Some(core), Some(shader)) = (&self.app.core, &mut self.shader) {
            if window_id == core.window().id() {
                if !shader.handle_input(core, &event) {
                    match event {
                        WindowEvent::CloseRequested => {
                            event_loop.exit();
                        }
                        WindowEvent::Resized(size) => {
                            if let Some(core) = &mut self.app.core {
                                core.resize(size);
                                shader.resize(core);
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            shader.update(core);
                            match shader.render(core) {
                                Ok(_) => {
                                    if self.first_render {
                                        self.first_render = false;
                                    }
                                }
                                Err(wgpu::SurfaceError::Lost) => {
                                    if let Some(core) = &mut self.app.core {
                                        core.resize(core.size);
                                    }
                                }
                                Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                                Err(e) => eprintln!("Render error: {:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(core) = &self.app.core {
            core.window().request_redraw();
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: StartCause) {
        // No special handling needed for new events
    }
}
