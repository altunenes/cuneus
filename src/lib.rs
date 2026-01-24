use std::sync::Arc;
use winit::window::Window;

pub use anyhow;
pub use bytemuck;
pub use egui;
pub use env_logger;
pub use wgpu;
pub use winit;

pub use bytemuck::{Pod, Zeroable};
pub use wgpu::SurfaceError;
pub use winit::event::WindowEvent;

mod app;
mod atomic;
pub mod compute;
mod controls;
mod export;
mod font;
mod fps;
#[cfg(feature = "media")]
pub mod gst;
pub mod hdri;
mod hot;
mod keyinputs;
mod mouse;
pub mod gaussian;
pub mod ply;
pub mod radix_sort;
mod renderer;
mod renderkit;
mod shader;
mod spectrum;
mod texture;
mod uniforms;
pub use app::*;
pub use atomic::AtomicBuffer;
pub use controls::{ControlsRequest, ShaderControls};
pub use export::{save_frame, ExportError, ExportManager, ExportSettings, ExportUiState};
pub use font::{CharInfo, FontSystem, FontUniforms};
pub use hdri::*;
pub use hot::ShaderHotReload;
pub use keyinputs::KeyInputHandler;
pub use mouse::*;
pub use gaussian::*;
pub use ply::*;
pub use renderer::*;
pub use renderkit::*;
pub use shader::*;
pub use texture::*;
pub use uniforms::*;

#[cfg(feature = "media")]
pub mod audio {
    pub use crate::gst::audio::{
        AudioDataProvider, AudioSynthManager, AudioSynthUniform, AudioWaveform, EnvelopeConfig,
        MusicalNote, SynthesisManager, SynthesisUniform, SynthesisWaveform,
    };
}

pub mod prelude {
    pub use crate::{
        compute::ComputeShader, compute::ComputeShaderBuilder, compute::MultiPassManager,
        save_frame, AtomicBuffer, CharInfo, ControlsRequest, Core, ExportManager, FontSystem,
        FontUniforms, KeyInputHandler, RenderKit, Renderer, ShaderApp, ShaderControls,
        ShaderHotReload, ShaderManager, TextureManager, UniformBinding, UniformProvider,
    };

    #[cfg(feature = "media")]
    pub use crate::{
        audio::{
            AudioWaveform, MusicalNote, SynthesisManager, SynthesisUniform, SynthesisWaveform,
        },
        gst,
    };

    pub use crate::anyhow;
    pub use crate::bytemuck;
    pub use crate::egui;
    pub use crate::wgpu;
    pub use crate::winit;

    pub use crate::SurfaceError;
    pub use crate::WindowEvent;
    pub use env_logger;

    pub use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
    pub use wgpu::{
        BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device, Queue, RenderPipeline,
        ShaderModule, Surface, SurfaceConfiguration, TextureFormat, TextureView,
    };

    pub use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::Window};
}

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
        let power_preference = instance
            .enumerate_adapters(wgpu::Backends::all())
            .iter()
            .find(|p| p.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .map(|_| wgpu::PowerPreference::HighPerformance)
            .unwrap_or(wgpu::PowerPreference::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: wgpu::Trace::default(),
            })
            .await
            .unwrap();
        let device = Arc::new(device);
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb() && *f == CAPTURE_FORMAT)
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
        println!("Core resize called with size: {new_size:?}");
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            println!("Surface reconfigured");
        }
    }

    /// Submit the current encoder and create a new one.
    ///
    /// Useful for multi-pass simulations where you need buffer updates to take effect
    /// before the next dispatch. wgpu batches all write_buffer calls before dispatches
    /// in the same submit, so this forces the GPU to see your changes.
    pub fn flush_encoder(&self, encoder: wgpu::CommandEncoder) -> wgpu::CommandEncoder {
        self.queue.submit(Some(encoder.finish()));
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Continued Encoder"),
        })
    }
}
