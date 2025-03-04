use std::sync::Arc;
use winit::window::Window;
pub use renderer::*;
pub use shader::*;
pub use texture::*;
pub use uniforms::*;
pub use app::*;
pub use base_shader::*;
pub use feedback::*;
pub use keyinputs::KeyInputHandler;
pub use export::{ExportSettings, ExportManager};
pub use export::ExportUiState;
pub use export::ExportError;
pub use hot::ShaderHotReload;
pub use controls::{ControlsRequest, ShaderControls};
pub use atomic::AtomicBuffer;
pub struct Core {
    pub surface: wgpu::Surface<'static>,
    pub device: Arc<wgpu::Device>,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub window: Window,
}
impl Core {
    pub async fn new(window: Window) -> Self {
        let size = window.inner_size();
        let instance_desc = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            backend_options: wgpu::BackendOptions::default(),
            ..Default::default()
        };
        let instance = wgpu::Instance::new(&instance_desc);
        let window_box = Box::new(window);
        let window_ptr = Box::into_raw(window_box);
        // SAFETY: window_ptr is valid as we just created it
        let surface = unsafe { instance.create_surface(&*window_ptr) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();
        let device = Arc::new(device);
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        // SAFETY: window_ptr is still valid and we're taking back ownership
        let window = unsafe { *Box::from_raw(window_ptr) };
        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
        }
    }
    pub fn window(&self) -> &Window {
        &self.window
    }
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        println!("Core resize called with size: {:?}", new_size);
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            println!("Surface reconfigured");
        }
    }
}

mod renderer;
mod shader;
mod texture;
mod uniforms;
mod app;
mod base_shader;
mod feedback; 
mod keyinputs;
mod export;
mod hot;
mod controls;
mod atomic;
#[cfg(not(target_os = "windows"))]
pub mod gst;
