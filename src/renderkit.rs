use crate::compute::ComputeShader;
#[cfg(feature = "media")]
use crate::gst::video::VideoTextureManager;
#[cfg(feature = "media")]
use crate::gst::webcam::WebcamTextureManager;
use crate::load_hdri_texture;
use crate::mouse::MouseTracker;
use crate::mouse::MouseUniform;
use crate::spectrum::SpectrumAnalyzer;
use crate::HdriMetadata;
use crate::{
    fps, ControlsRequest, Core, ExportManager, KeyInputHandler, Renderer, ResolutionUniform,
    ShaderControls, TextureManager, UniformBinding, UniformProvider,
};
use egui::ViewportId;
use egui_wgpu::ScreenDescriptor;
#[cfg(feature = "media")]
use log::warn;
use log::{error, info};
use std::path::Path;
use std::time::Instant;
use winit::event::WindowEvent;
#[cfg(target_os = "macos")]
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
#[cfg(not(target_os = "macos"))]
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimeUniform {
    pub time: f32,
    pub frame: u32,
}
impl UniformProvider for TimeUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
pub struct RenderKit {
    pub renderer: Renderer,
    #[cfg(feature = "media")]
    pub video_texture_manager: Option<VideoTextureManager>,
    #[cfg(feature = "media")]
    pub using_video_texture: bool,
    #[cfg(feature = "media")]
    pub webcam_texture_manager: Option<WebcamTextureManager>,
    #[cfg(feature = "media")]
    pub using_webcam_texture: bool,
    pub texture_manager: Option<TextureManager>,
    pub egui_renderer: egui_wgpu::Renderer,
    pub egui_state: egui_winit::State,
    pub context: egui::Context,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub start_time: Instant,
    pub time_uniform: UniformBinding<TimeUniform>,
    pub resolution_uniform: UniformBinding<ResolutionUniform>,
    pub key_handler: KeyInputHandler,
    pub export_manager: ExportManager,
    pub controls: ShaderControls,
    pub spectrum_analyzer: SpectrumAnalyzer,
    pub compute_shader: Option<ComputeShader>,
    pub fps_tracker: fps::FpsTracker,
    pub mouse_tracker: MouseTracker,
    pub mouse_uniform: Option<UniformBinding<MouseUniform>>,
    pub mouse_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub using_hdri_texture: bool,
    pub hdri_metadata: Option<HdriMetadata>,
    pub hdri_file_data: Option<Vec<u8>>,
}

impl RenderKit {
    const VERTEX_SHADER: &'static str = include_str!("../shaders/vertex.wgsl");
    const BLIT_SHADER: &'static str = include_str!("../shaders/blit.wgsl");

    /// Creates a bind group layout with texture (binding 0) and sampler (binding 1) for displaying compute shader output
    pub fn create_standard_texture_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("Standard Texture Layout"),
        })
    }

    /// Create RenderKit with standard texture layout
    pub fn new_with_standard_layout(core: &Core) -> Self {
        let layout = Self::create_standard_texture_layout(&core.device);
        Self::new(core, &layout, None)
    }

    pub fn new(core: &Core, layout: &wgpu::BindGroupLayout, fragment_entry: Option<&str>) -> Self {
        let bind_group_layouts = &[layout];
        let time_bind_group_layout =
            core.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let resolution_bind_group_layout =
            core.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let resolution_uniform = UniformBinding::new(
            &core.device,
            "Resolution Uniform",
            ResolutionUniform {
                dimensions: [core.size.width as f32, core.size.height as f32],
                _padding: [0.0, 0.0],
                audio_data: [[0.0; 4]; 32],
                bpm: 0.0,
                _bpm_padding: [0.0, 0.0, 0.0],
            },
            &resolution_bind_group_layout,
            0,
        );
        let vs_shader = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Vertex Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::VERTEX_SHADER.into()),
            });
        let fs_shader = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Fragment Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::BLIT_SHADER.into()),
            });
        let texture_bind_group_layout =
            core.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let pipeline_layout = core
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts,
                push_constant_ranges: &[],
            });
        let renderer = Renderer::new(
            &core.device,
            &vs_shader,
            &fs_shader,
            core.config.format,
            &pipeline_layout,
            fragment_entry,
        );
        let context = egui::Context::default();
        let egui_state = egui_winit::State::new(
            context.clone(),
            ViewportId::default(),
            core.window(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &core.device,
            core.config.format,
            egui_wgpu::RendererOptions::default(),
        );

        //  default texture manager
        let texture_manager =
            Self::create_default_texture_manager(core, &texture_bind_group_layout);
        let fps_tracker = fps::FpsTracker::new();
        let mouse_tracker = MouseTracker::new();

        Self {
            renderer,
            #[cfg(feature = "media")]
            video_texture_manager: None,
            #[cfg(feature = "media")]
            using_video_texture: false,
            #[cfg(feature = "media")]
            webcam_texture_manager: None,
            #[cfg(feature = "media")]
            using_webcam_texture: false,
            texture_manager: Some(texture_manager),
            egui_renderer,
            egui_state,
            context,
            texture_bind_group_layout,
            start_time: Instant::now(),
            time_uniform,
            resolution_uniform,
            key_handler: KeyInputHandler::new(),
            export_manager: ExportManager::new(),
            controls: ShaderControls::new(),
            spectrum_analyzer: SpectrumAnalyzer::new(),
            compute_shader: None,
            fps_tracker,
            mouse_tracker,
            mouse_uniform: None,
            mouse_bind_group_layout: None,
            using_hdri_texture: false,
            hdri_metadata: None,
            hdri_file_data: None,
        }
    }

    pub fn update_time(&mut self, queue: &wgpu::Queue) {
        self.time_uniform.data.time = self.start_time.elapsed().as_secs_f32();
        self.time_uniform.update(queue);
    }
    pub fn update_resolution(
        &mut self,
        queue: &wgpu::Queue,
        new_size: winit::dpi::PhysicalSize<u32>,
    ) {
        self.resolution_uniform.data.dimensions = [new_size.width as f32, new_size.height as f32];
        self.resolution_uniform.update(queue);
    }
    pub fn create_default_texture_manager(
        core: &Core,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> TextureManager {
        let default_texture = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let default_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = core
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Default Texture Bind Group"),
        });

        TextureManager {
            texture: default_texture,
            view: default_view,
            sampler,
            bind_group,
        }
    }

    pub fn render_ui<F>(&mut self, core: &Core, mut ui_builder: F) -> egui::FullOutput
    where
        F: FnMut(&egui::Context),
    {
        let raw_input = self.egui_state.take_egui_input(core.window());
        self.context.run(raw_input, |ctx| ui_builder(ctx))
    }

    pub fn handle_render_output(
        &mut self,
        core: &Core,
        view: &wgpu::TextureView,
        full_output: egui::FullOutput,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [core.config.width, core.config.height],
            pixels_per_point: core.window().scale_factor() as f32,
        };

        let clipped_primitives = self
            .context
            .tessellate(full_output.shapes, screen_descriptor.pixels_per_point);
        // Update egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&core.device, &core.queue, *id, image_delta);
        }

        self.egui_renderer.update_buffers(
            &core.device,
            &core.queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let render_pass = crate::Renderer::begin_render_pass(
                encoder,
                view,
                wgpu::LoadOp::Load,
                Some("Egui Render Pass"),
            );
            let mut render_pass = render_pass.into_inner().forget_lifetime();
            self.egui_renderer
                .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }
        // Cleanup egui textures
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
    pub fn load_media<P: AsRef<Path>>(&mut self, core: &Core, path: P) -> anyhow::Result<()> {
        let path_ref = path.as_ref();
        let extension = path_ref
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        match extension {
            // Image formats
            Some(ext)
                if ["png", "jpg", "jpeg", "bmp", "gif", "tiff", "webp"].contains(&ext.as_str()) =>
            {
                info!("Loading image: {:?}", path_ref);
                if let Ok(img) = image::open(path_ref) {
                    let rgba_image = img.into_rgba8();
                    let new_texture_manager = TextureManager::new(
                        &core.device,
                        &core.queue,
                        &rgba_image,
                        &self.texture_bind_group_layout,
                    );
                    self.texture_manager = Some(new_texture_manager);
                    #[cfg(feature = "media")]
                    {
                        self.using_video_texture = false;
                        self.video_texture_manager = None;
                        self.using_webcam_texture = false;
                        self.webcam_texture_manager = None;
                    }
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Failed to open image"))
                }
            }
            Some(ext) if ["hdr", "exr"].contains(&ext.as_str()) => {
                info!("Loading HDRI: {:?}", path_ref);
                let file_data = std::fs::read(path_ref)?;
                self.hdri_file_data = Some(file_data.clone());
                let default_exposure = 1.0;
                match load_hdri_texture(
                    &core.device,
                    &core.queue,
                    &file_data,
                    &self.texture_bind_group_layout,
                    default_exposure,
                ) {
                    Ok((texture_manager, metadata)) => {
                        self.texture_manager = Some(texture_manager);
                        #[cfg(feature = "media")]
                        {
                            self.using_video_texture = false;
                            self.video_texture_manager = None;
                            self.using_webcam_texture = false;
                            self.webcam_texture_manager = None;
                        }
                        self.using_hdri_texture = true;
                        self.hdri_metadata = Some(metadata);
                        Ok(())
                    }
                    Err(e) => {
                        error!("Failed to load HDRI: {}", e);
                        Err(anyhow::anyhow!("Failed to load HDRI: {}", e))
                    }
                }
            }
            #[cfg(feature = "media")]
            Some(ext) if ["mp4", "avi", "mkv", "mov", "webm"].contains(&ext.as_str()) => {
                info!("Loading video: {:?}", path_ref);
                match VideoTextureManager::new(
                    &core.device,
                    &core.queue,
                    &self.texture_bind_group_layout,
                    path_ref,
                ) {
                    Ok(video_manager) => {
                        self.video_texture_manager = Some(video_manager);
                        self.using_video_texture = true;
                        self.using_webcam_texture = false;
                        self.webcam_texture_manager = None;
                        if let Err(e) = self.play_video() {
                            warn!("Failed to play video: {}", e);
                        }
                        self.set_video_loop(true);

                        Ok(())
                    }
                    Err(e) => {
                        error!("Failed to load video: {}", e);
                        Err(e)
                    }
                }
            }
            _ => Err(anyhow::anyhow!("Unsupported media format: {:?}", path_ref)),
        }
    }
    #[cfg(feature = "media")]
    pub fn update_video_texture(&mut self, core: &Core, queue: &wgpu::Queue) -> bool {
        if self.using_video_texture {
            if let Some(video_manager) = &mut self.video_texture_manager {
                if let Ok(updated) = video_manager.update_texture(
                    &core.device,
                    queue,
                    &self.texture_bind_group_layout,
                ) {
                    return updated;
                }
            }
        }
        false
    }
    #[cfg(feature = "media")]
    pub fn play_video(&mut self) -> anyhow::Result<()> {
        if let Some(video_manager) = &mut self.video_texture_manager {
            video_manager.play()?;
        }
        Ok(())
    }
    #[cfg(feature = "media")]
    pub fn pause_video(&mut self) -> anyhow::Result<()> {
        if let Some(video_manager) = &mut self.video_texture_manager {
            video_manager.pause()?;
        }
        Ok(())
    }
    #[cfg(feature = "media")]
    pub fn seek_video(&mut self, position_seconds: f64) -> anyhow::Result<()> {
        if let Some(video_manager) = &mut self.video_texture_manager {
            let position = gstreamer::ClockTime::from_seconds(position_seconds as u64);
            video_manager.seek(position)?;
        }
        Ok(())
    }

    #[cfg(feature = "media")]
    pub fn set_video_loop(&mut self, should_loop: bool) {
        if let Some(video_manager) = &mut self.video_texture_manager {
            video_manager.set_loop(should_loop);
        }
    }

    #[cfg(feature = "media")]
    pub fn start_webcam(&mut self, core: &Core, device_index: Option<u32>) -> anyhow::Result<()> {
        info!("Starting webcam");
        let webcam_manager = WebcamTextureManager::new(
            &core.device,
            &core.queue,
            &self.texture_bind_group_layout,
            device_index,
        )?;

        let mut manager = webcam_manager;
        manager.start()?;

        self.webcam_texture_manager = Some(manager);
        self.using_webcam_texture = true;
        self.using_video_texture = false;
        self.video_texture_manager = None;
        self.using_hdri_texture = false;

        Ok(())
    }

    #[cfg(feature = "media")]
    pub fn stop_webcam(&mut self) -> anyhow::Result<()> {
        info!("Stopping webcam");
        if let Some(webcam_manager) = &mut self.webcam_texture_manager {
            webcam_manager.stop()?;
        }
        self.using_webcam_texture = false;
        self.webcam_texture_manager = None;
        Ok(())
    }

    #[cfg(feature = "media")]
    pub fn update_webcam_texture(&mut self, core: &Core, queue: &wgpu::Queue) -> bool {
        if self.using_webcam_texture {
            if let Some(webcam_manager) = &mut self.webcam_texture_manager {
                if let Ok(updated) = webcam_manager.update_texture(
                    &core.device,
                    queue,
                    &self.texture_bind_group_layout,
                ) {
                    return updated;
                }
            }
        }
        false
    }
    pub fn load_image(&mut self, core: &Core, path: std::path::PathBuf) {
        if let Ok(img) = image::open(path) {
            let rgba_image = img.into_rgba8();
            let new_texture_manager = TextureManager::new(
                &core.device,
                &core.queue,
                &rgba_image,
                &self.texture_bind_group_layout,
            );
            self.texture_manager = Some(new_texture_manager);
            #[cfg(feature = "media")]
            {
                self.using_video_texture = false;
                self.video_texture_manager = None;
                self.using_webcam_texture = false;
                self.webcam_texture_manager = None;
            }
        }
    }
    pub fn create_capture_texture(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::Buffer) {
        let capture_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Capture Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CAPTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let align = 256;
        let unpadded_bytes_per_row = width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;
        let buffer_size = padded_bytes_per_row * height;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Capture Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        (capture_texture, output_buffer)
    }
    pub fn apply_control_request(&mut self, request: ControlsRequest) {
        if request.should_reset {
            self.start_time = Instant::now();
        }
        self.controls.apply_ui_request(request);
    }
    #[cfg(feature = "media")]
    pub fn update_audio_spectrum(&mut self, queue: &wgpu::Queue) {
        self.spectrum_analyzer.update_spectrum(
            queue,
            &mut self.resolution_uniform,
            &self.video_texture_manager,
            self.using_video_texture,
        );
    }
    #[cfg(feature = "media")]
    pub fn handle_video_requests(&mut self, core: &Core, request: &ControlsRequest) {
        if let Some(path) = &request.load_media_path {
            if let Err(e) = self.load_media(core, path) {
                error!("Failed to load media: {}", e);
            }
        }

        if request.play_video {
            let _ = self.play_video();
        }

        if request.pause_video {
            let _ = self.pause_video();
        }

        if request.restart_video {
            let _ = self.seek_video(0.0);
            let _ = self.play_video();
        }

        if let Some(position) = request.seek_position {
            let _ = self.seek_video(position);
        }

        if let Some(should_loop) = request.set_loop {
            self.set_video_loop(should_loop);
        }

        // Handle audio control requests
        if let Some(volume) = request.set_volume {
            if let Some(vm) = &mut self.video_texture_manager {
                let _ = vm.set_volume(volume);
            }
        }

        if let Some(muted) = request.mute_audio {
            if let Some(vm) = &mut self.video_texture_manager {
                let _ = vm.set_mute(muted);
            }
        }
        if request.toggle_mute {
            if let Some(vm) = &mut self.video_texture_manager {
                let _ = vm.toggle_mute();
            }
        }
    }

    #[cfg(feature = "media")]
    pub fn handle_webcam_requests(&mut self, core: &Core, request: &ControlsRequest) {
        if request.start_webcam {
            if let Err(e) = self.start_webcam(core, request.webcam_device_index) {
                error!("Failed to start webcam: {}", e);
            }
        }

        if request.stop_webcam {
            if let Err(e) = self.stop_webcam() {
                error!("Failed to stop webcam: {}", e);
            }
        }
    }
    pub fn handle_hdri_requests(&mut self, core: &Core, request: &ControlsRequest) -> bool {
        if !self.using_hdri_texture {
            return false;
        }
        let mut updated = false;
        let mut new_exposure = None;
        let mut new_gamma = None;
        if let (Some(exposure), Some(hdri_meta)) = (request.hdri_exposure, &mut self.hdri_metadata)
        {
            if (exposure - hdri_meta.exposure).abs() > 0.001 {
                hdri_meta.exposure = exposure;
                new_exposure = Some(exposure);
                updated = true;
            }
        }
        if let (Some(gamma), Some(hdri_meta)) = (request.hdri_gamma, &mut self.hdri_metadata) {
            if (gamma - hdri_meta.gamma).abs() > 0.001 {
                hdri_meta.gamma = gamma;
                new_gamma = Some(gamma);
                updated = true;
            }
        }
        if updated {
            if let (Some(hdri_data), Some(texture_manager)) =
                (&self.hdri_file_data, &mut self.texture_manager)
            {
                let exposure = new_exposure
                    .unwrap_or_else(|| self.hdri_metadata.map(|meta| meta.exposure).unwrap_or(1.0));
                if let Err(e) = crate::update_hdri_exposure(
                    &core.device,
                    &core.queue,
                    hdri_data,
                    &self.texture_bind_group_layout,
                    texture_manager,
                    exposure,
                    new_gamma,
                ) {
                    error!("Failed to update HDRI parameters: {}", e);
                }
            }
        }
        updated
    }

    pub fn get_hdri_info(&self) -> Option<HdriMetadata> {
        if self.using_hdri_texture {
            self.hdri_metadata.clone()
        } else {
            None
        }
    }
    pub fn create_compute_shader(
        &mut self,
        core: &Core,
        shader_source: &str,
        _entry_point: &str,
        _workgroup_size: [u32; 3],
        _workgroup_count: Option<[u32; 3]>,
        _dispatch_once: bool,
    ) {
        // WIP untill I complete everything in compute folder
        self.compute_shader = Some(ComputeShader::new(core, shader_source));
    }

    pub fn enable_compute_hot_reload(
        &mut self,
        core: &Core,
        shader_path: &Path,
    ) -> Result<(), notify::Error> {
        if let Some(compute_shader) = &mut self.compute_shader {
            let shader_source = std::fs::read_to_string(shader_path)?;
            let shader_module = core
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Compute Shader Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });
            compute_shader.enable_hot_reload(
                core.device.clone(),
                shader_path.to_path_buf(),
                shader_module,
            )?;

            println!(
                "Compute shader hot reload enabled for: {}",
                shader_path.display()
            );
            Ok(())
        } else {
            Err(notify::Error::generic("No compute shader initialized"))
        }
    }

    pub fn dispatch_compute_shader(&mut self, encoder: &mut wgpu::CommandEncoder, core: &Core) {
        if let Some(compute) = &mut self.compute_shader {
            compute.dispatch(encoder, core);
        }
    }

    pub fn get_compute_output_texture(&self) -> Option<&TextureManager> {
        self.compute_shader
            .as_ref()
            .map(|compute| compute.get_output_texture())
    }

    pub fn resize_compute_shader(&mut self, core: &Core) {
        if let Some(compute) = &mut self.compute_shader {
            compute.resize(core, core.size.width, core.size.height);
        }
    }

    pub fn update_compute_shader_time(&mut self, elapsed: f32, delta: f32, queue: &wgpu::Queue) {
        if let Some(compute) = &mut self.compute_shader {
            compute.set_time(elapsed, delta, queue);
        }
    }

    /// Get video information if a video texture is loaded
    #[cfg(feature = "media")]
    pub fn get_video_info(
        &self,
    ) -> Option<(
        Option<f32>,
        f32,
        (u32, u32),
        Option<f32>,
        bool,
        bool,
        f64,
        bool,
    )> {
        if self.using_video_texture {
            if let Some(vm) = &self.video_texture_manager {
                Some((
                    vm.duration().map(|d| d.seconds() as f32),
                    vm.position().seconds() as f32,
                    vm.dimensions(),
                    vm.framerate().map(|(num, den)| num as f32 / den as f32),
                    vm.is_looping(),
                    vm.has_audio(),
                    vm.volume(),
                    vm.is_muted(),
                ))
            } else {
                None
            }
        } else {
            None
        }
    }

    #[cfg(feature = "media")]
    pub fn get_webcam_info(&self) -> Option<(u32, u32)> {
        if self.using_webcam_texture {
            if let Some(wm) = &self.webcam_texture_manager {
                Some(wm.dimensions())
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn setup_mouse_uniform(&mut self, core: &Core) {
        if self.mouse_uniform.is_none() {
            let mouse_bind_group_layout =
                core.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

            let mouse_uniform = UniformBinding::new(
                &core.device,
                "Mouse Uniform",
                self.mouse_tracker.uniform,
                &mouse_bind_group_layout,
                0,
            );

            self.mouse_bind_group_layout = Some(mouse_bind_group_layout);
            self.mouse_uniform = Some(mouse_uniform);
        }
    }

    pub fn update_mouse_uniform(&mut self, queue: &wgpu::Queue) {
        if let Some(mouse_uniform) = &mut self.mouse_uniform {
            mouse_uniform.data = self.mouse_tracker.uniform;
            mouse_uniform.update(queue);
        }
    }

    pub fn handle_mouse_input(
        &mut self,
        core: &Core,
        event: &WindowEvent,
        ui_handled: bool,
    ) -> bool {
        let window_size = [core.size.width as f32, core.size.height as f32];

        self.mouse_tracker
            .handle_mouse_input(event, window_size, ui_handled)
    }

    /// Get current active texture manager (video, webcam, or static image)
    pub fn get_current_texture_manager(&self) -> Option<&TextureManager> {
        #[cfg(feature = "media")]
        {
            if self.using_video_texture {
                return self
                    .video_texture_manager
                    .as_ref()
                    .map(|vm| vm.texture_manager());
            } else if self.using_webcam_texture {
                return self
                    .webcam_texture_manager
                    .as_ref()
                    .map(|wm| wm.texture_manager());
            }
        }
        self.texture_manager.as_ref()
    }

    /// Update current active texture and return whether an external texture update is needed
    pub fn update_current_texture(&mut self, core: &Core, queue: &wgpu::Queue) -> bool {
        #[cfg(feature = "media")]
        {
            if self.using_video_texture {
                return self.update_video_texture(core, queue);
            } else if self.using_webcam_texture {
                return self.update_webcam_texture(core, queue);
            }
        }
        // Static textures don't need updates
        false
    }
}
