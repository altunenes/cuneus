use log::info;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use wgpu;

use super::builder::ComputeConfiguration;
use super::multipass::MultiPassManager;
use super::resource::ResourceLayout;
use crate::{Core, FontSystem, ShaderHotReload, TextureManager, UniformBinding, UniformProvider};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeTimeUniform {
    pub time: f32,
    pub delta: f32,
    pub frame: u32,
    pub _padding: u32,
}

impl UniformProvider for ComputeTimeUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

pub struct ComputeShader {
    // Core resources
    pub pipelines: Vec<wgpu::ComputePipeline>,
    pub output_texture: TextureManager,
    pub time_uniform: UniformBinding<ComputeTimeUniform>,
    pub workgroup_size: [u32; 3],
    pub dispatch_once: bool,
    pub current_frame: u32,

    // Layouts following the 4-group convention
    pub bind_group_layouts: HashMap<u32, wgpu::BindGroupLayout>,
    pub pipeline_layout: wgpu::PipelineLayout,

    // Bind groups organized by convention
    pub group0_bind_group: wgpu::BindGroup, // Per-frame (time)
    pub group1_bind_group: wgpu::BindGroup, // Primary I/O & params
    pub group2_bind_group: Option<wgpu::BindGroup>, // Engine resources
    pub group3_bind_group: Option<wgpu::BindGroup>, // User data

    // Custom uniform parameters (Group 1)
    pub custom_uniform: Option<wgpu::Buffer>,
    pub custom_uniform_size: Option<u64>,

    // Input texture support (Group 1)
    pub placeholder_input_texture: Option<TextureManager>,

    // Multi-pass support
    pub multipass_manager: Option<MultiPassManager>,
    pub pass_dependencies: Option<HashMap<String, Vec<String>>>,
    pub pass_descriptions: Option<Vec<crate::compute::PassDescription>>,

    // Engine resources (Group 2)
    pub font_system: Option<FontSystem>,
    pub atomic_buffer_raw: Option<wgpu::Buffer>,
    pub audio_buffer: Option<wgpu::Buffer>,
    pub audio_staging_buffer: Option<wgpu::Buffer>,
    pub audio_spectrum_buffer: Option<wgpu::Buffer>,
    pub mouse_uniform: Option<UniformBinding<crate::MouseUniform>>,

    // Channel system for external textures (Group 2)
    pub channel_textures: HashMap<u32, Option<(wgpu::TextureView, wgpu::Sampler)>>,
    pub num_channels: u32,

    // User storage buffers (Group 3)
    pub storage_buffers: Vec<wgpu::Buffer>,

    // Empty bind groups for contiguous layout requirement
    pub empty_bind_groups: std::collections::HashMap<u32, wgpu::BindGroup>,

    // Configuration and hot reload
    pub entry_points: Vec<String>,
    pub hot_reload: Option<ShaderHotReload>,
    pub label: String,
    pub has_input_texture: bool,
}

impl ComputeShader {
    /// Create a compute shader from builder configuration
    pub fn from_builder(core: &Core, shader_source: &str, config: ComputeConfiguration) -> Self {
        // Step 1: Create resource layout following 4-group convention
        let mut resource_layout = ResourceLayout::new();

        // Group 0: Always has time uniform
        resource_layout.add_time_uniform();

        // Group 1: Primary I/O & Parameters
        resource_layout.add_output_texture(config.texture_format);
        if let Some(uniform_size) = config.custom_uniform_size {
            resource_layout.add_custom_uniform("params", uniform_size);
        }
        if config.has_input_texture {
            resource_layout.add_input_texture();
        }

        // Group 2: Engine Resources
        if config.has_mouse {
            resource_layout.add_mouse_uniform();
        }
        if config.has_fonts {
            resource_layout.add_font_resources();
        }
        if config.has_audio {
            resource_layout.add_audio_buffer(config.audio_buffer_size);
        }
        if config.has_atomic_buffer {
            // Create buffer with 3 u32s per pixel
            // The shader accesses: atomic_buffer[idx], atomic_buffer[idx + w*h], atomic_buffer[idx + 2*w*h]
            let atomic_size = (core.size.width * core.size.height * 3 * 4) as u64;
            resource_layout.add_atomic_buffer(atomic_size);
        }
        if config.has_audio_spectrum {
            resource_layout.add_audio_spectrum_buffer(config.audio_spectrum_size);
        }
        if let Some(num_channels) = config.num_channels {
            resource_layout.add_channel_textures(num_channels);
        }

        // Group 3: User-defined storage buffers with optional multi-pass input textures
        if !config.storage_buffers.is_empty() {
            // Priority: User storage buffers (needed for FFT algorithm data, etc.) lemme think more generic things for future!
            for buffer_spec in &config.storage_buffers {
                if buffer_spec.read_only {
                    resource_layout
                        .add_readonly_storage_buffer(&buffer_spec.name, buffer_spec.size_bytes);
                } else {
                    resource_layout.add_storage_buffer(&buffer_spec.name, buffer_spec.size_bytes);
                }
            }
        } else if config.passes.is_some() {
            // Fallback: Multi-pass input textures only if no storage buffers requested
            resource_layout.add_multipass_input_textures();
        }

        // Step 2: Create bind group layouts
        let bind_group_layouts = resource_layout.create_bind_group_layouts(&core.device);

        // Step 3: Create pipeline layout - WebGPU requires contiguous bind group indices
        // I need to ensure all groups 0-3 are present, creating empty layouts if needed
        let mut layouts_vec: Vec<wgpu::BindGroupLayout> = Vec::new();

        for i in 0..4 {
            if let Some(layout) = bind_group_layouts.get(&i) {
                layouts_vec.push(layout.clone()); // Clone the existing layout
            } else {
                // Create an empty bind group layout for missing groups
                let empty_layout =
                    core.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some(&format!("Empty Group {} Layout", i)),
                            entries: &[],
                        });
                layouts_vec.push(empty_layout);
            }
        }

        let layout_refs: Vec<&wgpu::BindGroupLayout> = layouts_vec.iter().collect();

        let pipeline_layout = core
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} Pipeline Layout", config.label)),
                bind_group_layouts: &layout_refs,
                push_constant_ranges: &[],
            });

        // Step 4: Create time uniform (Group 0)
        let time_bind_group_layout = bind_group_layouts.get(&0).unwrap();
        let time_uniform = UniformBinding::new(
            &core.device,
            &format!("{} Time Uniform", config.label),
            ComputeTimeUniform {
                time: 0.0,
                delta: 0.0,
                frame: 0,
                _padding: 0,
            },
            time_bind_group_layout,
            0,
        );
        let group0_bind_group = time_uniform.bind_group.clone();

        // Step 5: Create output texture
        let output_texture = Self::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            config.texture_format,
            &format!("{} Output Texture", config.label),
        );

        // Step 5.5: Create custom uniform buffer if needed
        let custom_uniform = if let Some(uniform_size) = config.custom_uniform_size {
            Some(core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Custom Uniform Buffer", config.label)),
                size: uniform_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        // Create placeholder input texture for shaders that need input texture support
        let placeholder_input_texture = if config.has_input_texture {
            Some(Self::create_placeholder_input_texture(
                &core.device,
                &format!("{} Placeholder Input", config.label),
            ))
        } else {
            None
        };

        let group1_bind_group = Self::create_group1_bind_group(
            &core.device,
            bind_group_layouts.get(&1).unwrap(),
            &output_texture,
            &config,
            custom_uniform.as_ref(),
            placeholder_input_texture.as_ref().map(|t| &t.view),
            placeholder_input_texture.as_ref().map(|t| &t.sampler),
        );

        // Step 6: Create engine resources (Group 2) if needed
        let (
            font_system,
            atomic_buffer_raw,
            audio_buffer,
            audio_staging_buffer,
            audio_spectrum_buffer,
            mouse_uniform,
            group2_bind_group,
        ) = Self::create_engine_resources(core, &bind_group_layouts, &config);

        // Step 7: Create user storage buffers (Group 3) if needed
        let (storage_buffers, group3_bind_group) = if !config.storage_buffers.is_empty() {
            // Create storage buffers (works for both single-pass and multi-pass with storage)
            Self::create_user_storage_buffers(core, &bind_group_layouts, &config)
        } else if config.passes.is_some() {
            // Pure multi-pass mode: Group 3 will be managed dynamically by MultiPassManager
            (Vec::new(), None)
        } else {
            // No storage buffers needed
            (Vec::new(), None)
        };

        // Step 7.5: Create empty bind groups for empty layouts (needed when we create contiguous layouts)
        let mut empty_bind_groups: std::collections::HashMap<u32, wgpu::BindGroup> =
            std::collections::HashMap::new();
        for i in 0..4 {
            if !bind_group_layouts.contains_key(&i) {
                // This group was missing and got an empty layout, create an empty bind group
                let empty_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Empty Group {} Bind Group", i)),
                    layout: &layouts_vec[i as usize],
                    entries: &[],
                });
                empty_bind_groups.insert(i, empty_bind_group);
            }
        }

        // Step 8: Create multi-pass manager if needed (only for texture ping-pong, not storage buffers)
        let (multipass_manager, pass_dependencies) = if let Some(passes) = &config.passes {
            if config.storage_buffers.is_empty() {
                // Pure multi-pass mode with texture ping-pong: Group 3 managed by MultiPassManager
                let buffer_names: Vec<String> = passes.iter().map(|p| p.name.clone()).collect();
                let dependencies: HashMap<String, Vec<String>> = passes
                    .iter()
                    .map(|p| (p.name.clone(), p.inputs.clone()))
                    .collect();

                let manager = MultiPassManager::new(
                    core,
                    &buffer_names,
                    config.texture_format,
                    bind_group_layouts.get(&3).unwrap().clone(),
                );

                (Some(manager), Some(dependencies))
            } else {
                // Multi-pass with storage buffers: no texture ping-pong needed
                // Passes use shared storage buffers instead of ping-pong textures
                let dependencies: HashMap<String, Vec<String>> = passes
                    .iter()
                    .map(|p| (p.name.clone(), p.inputs.clone()))
                    .collect();
                (None, Some(dependencies))
            }
        } else {
            (None, None)
        };

        // Step 9: Create compute pipelines
        let shader_module = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Module", config.label)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let mut pipelines = Vec::new();
        for entry_point in &config.entry_points {
            let pipeline = core
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{} Pipeline - {}", config.label, entry_point)),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            pipelines.push(pipeline);
        }

        Self {
            pipelines,
            output_texture,
            time_uniform,
            workgroup_size: config.workgroup_size,
            dispatch_once: config.dispatch_once,
            current_frame: 0,
            bind_group_layouts,
            pipeline_layout,
            group0_bind_group,
            group1_bind_group,
            group2_bind_group,
            group3_bind_group,
            multipass_manager,
            pass_dependencies,
            pass_descriptions: config.passes.clone(),
            font_system,
            atomic_buffer_raw,
            audio_buffer,
            audio_staging_buffer,
            audio_spectrum_buffer,
            mouse_uniform,
            storage_buffers,
            empty_bind_groups,
            custom_uniform,
            custom_uniform_size: config.custom_uniform_size,
            placeholder_input_texture,
            channel_textures: Self::initialize_channel_textures(config.num_channels.unwrap_or(0)),
            num_channels: config.num_channels.unwrap_or(0),
            entry_points: config.entry_points,
            hot_reload: None,
            label: config.label,
            has_input_texture: config.has_input_texture,
        }
    }

    fn create_output_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> TextureManager {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        // Create a dummy bind group for display purposes - this is only used by the display renderer
        let dummy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Display Texture Layout"),
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
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &dummy_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some(&format!("{} Display Bind Group", label)),
        });

        TextureManager {
            texture,
            view,
            sampler,
            bind_group,
        }
    }

    fn create_placeholder_input_texture(device: &wgpu::Device, label: &str) -> TextureManager {
        // Create a minimal 1x1 placeholder texture for initialization
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb, // Match real texture format
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        // Initialize placeholder with black pixels instead of uninitialized data
        // This prevents red artifacts when no real texture is loaded
        // Note: We could write actual data here, but shaders should handle empty textures gracefully

        // Create dummy bind group layout and bind group for placeholder
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} Placeholder Layout", label)),
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
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some(&format!("{} Placeholder Bind Group", label)),
        });

        TextureManager {
            texture,
            view,
            sampler,
            bind_group,
        }
    }

    fn create_group1_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        output_texture: &TextureManager,
        config: &ComputeConfiguration,
        custom_uniform_buffer: Option<&wgpu::Buffer>,
        input_texture_view: Option<&wgpu::TextureView>,
        input_sampler: Option<&wgpu::Sampler>,
    ) -> wgpu::BindGroup {
        // Create a storage view for the compute shader
        let storage_view = output_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&storage_view),
        }];

        // Add custom uniform if present
        if let (Some(buffer), Some(_size)) = (custom_uniform_buffer, config.custom_uniform_size) {
            entries.push(wgpu::BindGroupEntry {
                binding: 1, // Custom uniforms go to binding 1 in Group 1
                resource: buffer.as_entire_binding(),
            });
        }

        // Add input texture and sampler if present (for shaders like FFT): again, this still not "perfect" and generic but let me think more
        if config.has_input_texture {
            // Input textures should always be provided - if not, there's an architecture issue
            if let (Some(view), Some(sampler)) = (input_texture_view, input_sampler) {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2, // Input texture goes to binding 2
                    resource: wgpu::BindingResource::TextureView(view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3, // Input sampler goes to binding 3
                    resource: wgpu::BindingResource::Sampler(sampler),
                });
            } else {
                // This indicates an architecture problem - input texture support needs placeholder handling
                log::error!("Input texture required but not provided during bind group creation");
            }
        }

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &entries,
            label: Some("Group 1 Bind Group"),
        })
    }

    fn create_engine_resources(
        core: &Core,
        layouts: &HashMap<u32, wgpu::BindGroupLayout>,
        config: &ComputeConfiguration,
    ) -> (
        Option<FontSystem>,
        Option<wgpu::Buffer>,
        Option<wgpu::Buffer>,
        Option<wgpu::Buffer>,
        Option<wgpu::Buffer>,
        Option<UniformBinding<crate::MouseUniform>>,
        Option<wgpu::BindGroup>,
    ) {
        let layout = layouts.get(&2);
        if layout.is_none() {
            return (None, None, None, None, None, None, None);
        }
        let layout = layout.unwrap();

        // Create font system if needed
        let font_system = if config.has_fonts {
            Some(FontSystem::new(core))
        } else {
            None
        };

        // Create atomic buffer if needed (raw buffer, not old AtomicBuffer struct)
        // buffer size: 3 u32s * 4 bytes per pixel
        let atomic_buffer_raw = if config.has_atomic_buffer {
            let buffer_size = (core.size.width * core.size.height * 3 * 4) as u64;
            Some(core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Atomic Storage Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        // Create audio buffers if needed
        let (audio_buffer, audio_staging_buffer) = if config.has_audio {
            let buffer_size = config.audio_buffer_size * std::mem::size_of::<f32>();

            let audio_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Audio Buffer", config.label)),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Audio Staging Buffer", config.label)),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            (Some(audio_buffer), Some(staging_buffer))
        } else {
            (None, None)
        };

        // Create audio spectrum buffer if needed
        let audio_spectrum_buffer = if config.has_audio_spectrum {
            let buffer_size = config.audio_spectrum_size * std::mem::size_of::<f32>();

            let buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Audio Spectrum Buffer", config.label)),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            Some(buffer)
        } else {
            None
        };

        // Create mouse uniform if needed
        let mouse_uniform = if config.has_mouse {
            // Create a temporary bind group layout for UniformBinding compatibility
            let temp_layout =
                core.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                        label: Some("Temp Mouse Layout"),
                    });

            Some(UniformBinding::new(
                &core.device,
                "Mouse Uniform",
                crate::MouseUniform::default(),
                &temp_layout,
                0,
            ))
        } else {
            None
        };

        // Create Group 2 bind group
        // Create empty channel textures map for initial bind group creation
        let empty_channels = std::collections::HashMap::new();
        let num_channels = config.num_channels.unwrap_or(0);

        let bind_group = Self::create_group2_bind_group(
            &core.device,
            &core.queue,
            layout,
            &font_system,
            &atomic_buffer_raw,
            &audio_buffer,
            &audio_spectrum_buffer,
            &mouse_uniform,
            &empty_channels,
            num_channels,
        );

        (
            font_system,
            atomic_buffer_raw,
            audio_buffer,
            audio_staging_buffer,
            audio_spectrum_buffer,
            mouse_uniform,
            bind_group,
        )
    }

    fn create_group2_bind_group(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        font_system: &Option<FontSystem>,
        atomic_buffer_raw: &Option<wgpu::Buffer>,
        audio_buffer: &Option<wgpu::Buffer>,
        audio_spectrum_buffer: &Option<wgpu::Buffer>,
        mouse_uniform: &Option<UniformBinding<crate::MouseUniform>>,
        channel_textures: &HashMap<u32, Option<(wgpu::TextureView, wgpu::Sampler)>>,
        num_channels: u32,
    ) -> Option<wgpu::BindGroup> {
        // Create entries based on expected layout from ResourceLayout
        // Order must match ResourceLayout creation order:
        // 1. mouse (if has_mouse) -> binding 0
        // 2. fonts (if has_fonts) -> bindings 1,2,3
        // 3. audio (if has_audio) -> binding N
        // 4. audio_spectrum (if has_audio_spectrum) -> binding N+1
        // 5. atomic_buffer (if has_atomic_buffer) -> binding N+2
        // 6. channels (if num_channels > 0) -> bindings N+3 onwards (texture + sampler pairs)

        // Create a default 1x1 magenta texture for unassigned channels
        let default_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Channel Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Fill with magenta color so we can see when default texture is used
        let magenta_data: [u8; 4] = [255, 0, 255, 255];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &magenta_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let default_texture_view =
            default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let mut entries = Vec::new();
        let mut binding_counter = 0;

        // Add mouse uniform (binding 0)
        if let Some(mouse) = mouse_uniform {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_counter,
                resource: mouse.buffer.as_entire_binding(),
            });
            binding_counter += 1;
        }

        // Add font texture resources (Shadertoy-style)
        if let Some(font_tex) = font_system {
            entries.extend_from_slice(&[
                wgpu::BindGroupEntry {
                    binding: binding_counter,
                    resource: font_tex.font_uniforms.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: binding_counter + 1,
                    resource: wgpu::BindingResource::TextureView(&font_tex.atlas_texture.view),
                },
            ]);
            binding_counter += 2;
        }

        // Add audio buffer
        if let Some(audio) = audio_buffer {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_counter,
                resource: audio.as_entire_binding(),
            });
            binding_counter += 1;
        }

        // Add audio spectrum buffer
        if let Some(audio_spectrum) = audio_spectrum_buffer {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_counter,
                resource: audio_spectrum.as_entire_binding(),
            });
            binding_counter += 1;
        }

        // Add atomic buffer (if provided)
        if let Some(atomic_buf) = atomic_buffer_raw {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_counter,
                resource: atomic_buf.as_entire_binding(),
            });
            binding_counter += 1;
        }

        // Add channel textures (channel0, channel1, etc. with their samplers)
        for i in 0..num_channels {
            // Channel texture binding
            let (texture_view, sampler) = if let Some(Some((view, samp))) = channel_textures.get(&i)
            {
                (view, samp)
            } else {
                (&default_texture_view, &default_sampler)
            };

            entries.push(wgpu::BindGroupEntry {
                binding: binding_counter,
                resource: wgpu::BindingResource::TextureView(texture_view),
            });
            binding_counter += 1;

            // Channel sampler binding
            entries.push(wgpu::BindGroupEntry {
                binding: binding_counter,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
            binding_counter += 1;
        }

        if entries.is_empty() {
            return None;
        }

        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &entries,
            label: Some("Group 2 Bind Group"),
        }))
    }

    fn create_user_storage_buffers(
        core: &Core,
        layouts: &HashMap<u32, wgpu::BindGroupLayout>,
        config: &ComputeConfiguration,
    ) -> (Vec<wgpu::Buffer>, Option<wgpu::BindGroup>) {
        if config.storage_buffers.is_empty() {
            return (Vec::new(), None);
        }

        let layout = layouts.get(&3);
        if layout.is_none() {
            return (Vec::new(), None);
        }
        let layout = layout.unwrap();

        // Create storage buffers and entries in one pass
        let mut storage_buffers = Vec::new();
        let mut entries = Vec::new();

        for (_i, buffer_spec) in config.storage_buffers.iter().enumerate() {
            let buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&buffer_spec.name),
                size: buffer_spec.size_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            storage_buffers.push(buffer);
        }

        // Create entries using references to stored buffers
        for (i, buffer) in storage_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &entries,
            label: Some("Group 3 Bind Group"),
        });

        (storage_buffers, Some(bind_group))
    }

    /// Dispatch single stage of compute shader with custom workgroup count
    pub fn dispatch_stage_with_workgroups(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        stage_index: usize,
        workgroup_count: [u32; 3],
    ) {
        if stage_index >= self.pipelines.len() {
            log::error!(
                "Stage index {} out of bounds (max: {})",
                stage_index,
                self.pipelines.len() - 1
            );
            return;
        }

        if self.dispatch_once && self.current_frame > 0 {
            return;
        }

        let pipeline = &self.pipelines[stage_index];
        let entry_point = &self.entry_points[stage_index];

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!(
                "{} Stage {} - {}",
                self.label, stage_index, entry_point
            )),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);

        // Set bind groups following the 4-group convention
        compute_pass.set_bind_group(0, &self.group0_bind_group, &[]); // Per-frame
        compute_pass.set_bind_group(1, &self.group1_bind_group, &[]); // Primary I/O

        // Group 2: Engine resources
        if let Some(ref group2) = self.group2_bind_group {
            compute_pass.set_bind_group(2, group2, &[]);
        } else if let Some(empty_group2) = self.empty_bind_groups.get(&2) {
            compute_pass.set_bind_group(2, empty_group2, &[]);
        }

        // Group 3: User data
        if let Some(ref group3) = self.group3_bind_group {
            compute_pass.set_bind_group(3, group3, &[]);
        } else if let Some(empty_group3) = self.empty_bind_groups.get(&3) {
            compute_pass.set_bind_group(3, empty_group3, &[]);
        }

        compute_pass.dispatch_workgroups(
            workgroup_count[0],
            workgroup_count[1],
            workgroup_count[2],
        );
    }

    /// Dispatch single stage of compute shader (for fine-grained control like old system)
    pub fn dispatch_stage(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        core: &Core,
        stage_index: usize,
    ) {
        let workgroup_count = [
            core.size.width.div_ceil(self.workgroup_size[0]),
            core.size.height.div_ceil(self.workgroup_size[1]),
            1,
        ];
        self.dispatch_stage_with_workgroups(encoder, stage_index, workgroup_count);
    }

    pub fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder, core: &Core) {
        if self.dispatch_once && self.current_frame > 0 {
            return;
        }

        let workgroup_count = [
            core.size.width.div_ceil(self.workgroup_size[0]),
            core.size.height.div_ceil(self.workgroup_size[1]),
            1,
        ];

        // Handle multi-pass execution
        if self.multipass_manager.is_some() {
            self.dispatch_multipass(encoder, core, workgroup_count);
        } else {
            self.dispatch_single_pass(encoder, core, workgroup_count);
        }

        self.current_frame += 1;
    }

    /// Flip ping-pong buffers for multi-pass rendering (call after render)
    pub fn flip_buffers(&mut self) {
        if let Some(ref mut multipass) = self.multipass_manager {
            multipass.flip_buffers();
        }
    }

    /// Update custom uniform parameters
    pub fn set_custom_params<T: bytemuck::Pod>(&self, params: T, queue: &wgpu::Queue) {
        if let Some(ref buffer) = self.custom_uniform {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&params));
        } else {
            log::warn!("Attempted to set custom params but no custom uniform buffer exists");
        }
    }

    /// Get the custom uniform buffer size (if any)
    pub fn get_custom_uniform_size(&self) -> Option<u64> {
        self.custom_uniform_size
    }

    /// Update input texture for shaders that use input textures (like FFT)
    pub fn update_input_texture(
        &mut self,
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        device: &wgpu::Device,
    ) {
        if !self.has_input_texture {
            log::warn!("Attempted to update input texture but shader was not configured with input texture support");
            return;
        }

        // Update the placeholder texture manager to store the current texture for multipass use
        if let Some(ref mut _placeholder) = self.placeholder_input_texture {
            // Note: We can't directly replace the view/sampler references in TextureManager
            // since they're owned. In practice, fluid.rs calls this method with the texture
            // from base.get_current_texture_manager() which already updates the correct texture.
            // The placeholder serves as the fallback, but in multipass we should use the current one.
        }

        // Recreate Group 1 bind group with new input texture
        let group1_layout = self.bind_group_layouts.get(&1).unwrap();

        // Create a storage view for the compute shader
        let storage_view = self
            .output_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&storage_view),
        }];

        // Add custom uniform if present
        if let Some(ref buffer) = self.custom_uniform {
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            });
        }

        // Add updated input texture and sampler
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(texture_view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Sampler(sampler),
        });

        self.group1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: group1_layout,
            entries: &entries,
            label: Some("Updated Group 1 Bind Group with Input Texture"),
        });
    }

    /// Update a specific channel texture (channel0, channel1, etc.)
    pub fn update_channel_texture(
        &mut self,
        channel_index: u32,
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if channel_index >= self.num_channels {
            log::warn!(
                "Attempted to update channel {} but only {} channels are configured",
                channel_index,
                self.num_channels
            );
            return;
        }

        // Store the channel texture
        self.channel_textures
            .insert(channel_index, Some((texture_view.clone(), sampler.clone())));

        // Recreate Group 2 bind group with updated channel
        self.recreate_group2_bind_group(device, queue);
    }

    fn initialize_channel_textures(
        num_channels: u32,
    ) -> HashMap<u32, Option<(wgpu::TextureView, wgpu::Sampler)>> {
        let mut channel_textures = HashMap::new();
        for i in 0..num_channels {
            channel_textures.insert(i, None);
        }
        channel_textures
    }

    fn recreate_group2_bind_group(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if let Some(layout) = self.bind_group_layouts.get(&2) {
            self.group2_bind_group = Self::create_group2_bind_group(
                device,
                queue,
                layout,
                &self.font_system,
                &self.atomic_buffer_raw,
                &self.audio_buffer,
                &self.audio_spectrum_buffer,
                &self.mouse_uniform,
                &self.channel_textures,
                self.num_channels,
            );
        }
    }

    fn dispatch_single_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        _core: &Core,
        workgroup_count: [u32; 3],
    ) {
        for (i, pipeline) in self.pipelines.iter().enumerate() {
            // Get workgroup count for this specific pass
            let pass_workgroup_count = if let Some(ref pass_descriptions) = self.pass_descriptions {
                if let Some(pass_desc) = pass_descriptions.get(i) {
                    if let Some(custom_size) = pass_desc.workgroup_size {
                        custom_size // Use custom workgroup size from PassDescription
                    } else {
                        workgroup_count // Fall back to default screen-based size
                    }
                } else {
                    workgroup_count // Fall back to default if no pass description
                }
            } else {
                workgroup_count // Fall back to default if no pass descriptions
            };
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Compute Pass {}", self.label, i)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);

            // Set bind groups following the 4-group convention
            compute_pass.set_bind_group(0, &self.group0_bind_group, &[]); // Per-frame
            compute_pass.set_bind_group(1, &self.group1_bind_group, &[]); // Primary I/O

            // Group 2: Engine resources (required - use empty bind group if not available)
            if let Some(ref group2) = self.group2_bind_group {
                compute_pass.set_bind_group(2, group2, &[]); // Engine resources
            } else if let Some(empty_group2) = self.empty_bind_groups.get(&2) {
                compute_pass.set_bind_group(2, empty_group2, &[]);
            } else {
                log::error!("No Group 2 bind group available - this shouldn't happen with contiguous layout");
            }

            // Group 3: User data (required - use empty bind group if not available)
            if let Some(ref group3) = self.group3_bind_group {
                compute_pass.set_bind_group(3, group3, &[]); // User data
            } else if let Some(empty_group3) = self.empty_bind_groups.get(&3) {
                compute_pass.set_bind_group(3, empty_group3, &[]);
            } else {
                log::error!("No Group 3 bind group available - this shouldn't happen with contiguous layout");
            }

            compute_pass.dispatch_workgroups(
                pass_workgroup_count[0],
                pass_workgroup_count[1],
                pass_workgroup_count[2],
            );
        }
    }

    fn dispatch_multipass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        core: &Core,
        workgroup_count: [u32; 3],
    ) {
        let num_passes = self.pipelines.len();

        let sampler = core
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        // Execute each pass in order with proper dependencies
        for pass_idx in 0..num_passes {
            let pipeline = &self.pipelines[pass_idx];
            let entry_point = &self.entry_points[pass_idx];

            // Get workgroup count for this specific pass
            let pass_workgroup_count = if let Some(ref pass_descriptions) = self.pass_descriptions {
                if let Some(pass_desc) = pass_descriptions.get(pass_idx) {
                    if let Some(custom_size) = pass_desc.workgroup_size {
                        custom_size // Use custom workgroup size from PassDescription
                    } else {
                        workgroup_count // Fall back to default screen-based size
                    }
                } else {
                    workgroup_count // Fall back to default if no pass description
                }
            } else {
                workgroup_count // Fall back to default if no pass descriptions
            };

            // Create input bind group for this pass based on its dependencies
            let input_bind_group = if let (Some(multipass), Some(dependencies)) =
                (&self.multipass_manager, &self.pass_dependencies)
            {
                let empty_deps = Vec::new();
                let pass_dependencies = dependencies.get(entry_point).unwrap_or(&empty_deps);
                multipass.create_input_bind_group(&core.device, &sampler, pass_dependencies)
            } else {
                // Fallback for safety
                continue;
            };

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Multi-Pass - {}", self.label, entry_point)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &self.group0_bind_group, &[]); // Time

            // Group 1: Output texture binding - different for each pass type
            if entry_point == "main_image" {
                // main_image writes to the final output texture - use main Group 1
                compute_pass.set_bind_group(1, &self.group1_bind_group, &[]);
            } else {
                // Intermediate passes write to their ping-pong buffers
                // Create a Group 1 compatible bind group with both storage texture and custom uniform
                if let Some(multipass) = &self.multipass_manager {
                    let write_texture = multipass.get_write_texture(entry_point);
                    let write_view =
                        write_texture.create_view(&wgpu::TextureViewDescriptor::default());

                    let mut entries = vec![wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&write_view),
                    }];

                    // Add custom uniform if present (to match Group 1 layout)
                    if let Some(ref uniform_buffer) = self.custom_uniform {
                        entries.push(wgpu::BindGroupEntry {
                            binding: 1,
                            resource: uniform_buffer.as_entire_binding(),
                        });
                    }

                    let group1_layout = self.bind_group_layouts.get(&1).unwrap();
                    let intermediate_bind_group =
                        core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&format!("{} Intermediate Group 1", entry_point)),
                            layout: group1_layout,
                            entries: &entries,
                        });

                    compute_pass.set_bind_group(1, &intermediate_bind_group, &[]);
                } else {
                    // Fallback to main group if no multipass (shouldn't happen)
                    compute_pass.set_bind_group(1, &self.group1_bind_group, &[]);
                    log::warn!("No multipass manager for intermediate pass {}", entry_point);
                }
            }

            // Group 2: Engine resources (required - use empty bind group if not available)
            if let Some(ref group2) = self.group2_bind_group {
                compute_pass.set_bind_group(2, group2, &[]);
            } else if let Some(empty_group2) = self.empty_bind_groups.get(&2) {
                log::warn!(
                    "Using empty Group 2 bind group for pass {} - channels won't work!",
                    entry_point
                );
                compute_pass.set_bind_group(2, empty_group2, &[]);
            } else {
                log::error!("No Group 2 bind group available - this shouldn't happen with contiguous layout");
            }

            // Group 3: Multi-pass input textures (dynamically created per pass)
            compute_pass.set_bind_group(3, &input_bind_group, &[]);

            compute_pass.dispatch_workgroups(
                pass_workgroup_count[0],
                pass_workgroup_count[1],
                pass_workgroup_count[2],
            );

            // Flip buffers after each pass (except the last one) so next pass can read what this pass wrote
            if pass_idx < num_passes - 1 {
                if let Some(ref mut multipass_mut) = self.multipass_manager {
                    multipass_mut.flip_buffers();
                }
            }
        }
    }

    /// Enable hot reload for the shader
    pub fn enable_hot_reload(
        &mut self,
        device: Arc<wgpu::Device>,
        shader_path: PathBuf,
        shader_module: wgpu::ShaderModule,
    ) -> Result<(), notify::Error> {
        let entry_point = self
            .entry_points
            .first()
            .cloned()
            .unwrap_or_else(|| "main".to_string());
        let hot_reload =
            ShaderHotReload::new_compute(device, shader_path, shader_module, &entry_point)?;

        self.hot_reload = Some(hot_reload);
        Ok(())
    }

    /// Check for hot reload updates
    pub fn check_hot_reload(&mut self, device: &wgpu::Device) -> bool {
        if let Some(hot_reload) = &mut self.hot_reload {
            if let Some(new_module) = hot_reload.reload_compute_shader() {
                // Recreate pipelines with updated shader
                let mut new_pipelines = Vec::new();
                for entry_point in &self.entry_points {
                    let new_pipeline =
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(&format!(
                                "Updated {} Pipeline - {}",
                                self.label, entry_point
                            )),
                            layout: Some(&self.pipeline_layout),
                            module: &new_module,
                            entry_point: Some(entry_point),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            cache: None,
                        });
                    new_pipelines.push(new_pipeline);
                }

                self.pipelines = new_pipelines;
                info!(
                    "{} shader hot-reloaded at frame: {}",
                    self.label, self.current_frame
                );
                return true;
            }
        }
        false
    }

    /// Set time uniform data
    pub fn set_time(&mut self, elapsed: f32, delta: f32, queue: &wgpu::Queue) {
        self.time_uniform.data.time = elapsed;
        self.time_uniform.data.delta = delta;
        self.time_uniform.data.frame = self.current_frame;
        self.time_uniform.update(queue);
    }

    /// Update audio spectrum buffer with data from ResolutionUniform
    pub fn update_audio_spectrum(
        &mut self,
        resolution_uniform: &crate::ResolutionUniform,
        queue: &wgpu::Queue,
    ) {
        if let Some(ref buffer) = self.audio_spectrum_buffer {
            // Convert audio_data from [[f32; 4]; 32] to [f32; 64] format
            // Note: Only taking 64 values to match the original gAV function expectation
            let mut spectrum_data = vec![0.0f32; 64];
            for i in 0..64 {
                let vec_idx = i / 4;
                let comp_idx = i % 4;
                if vec_idx < 32 {
                    spectrum_data[i] = resolution_uniform.audio_data[vec_idx][comp_idx];
                }
            }

            // Debug logging to see if we're getting data
            let total_energy: f32 = spectrum_data.iter().sum();
            if total_energy > 0.01 {
                log::info!(
                    "Audio spectrum update: total_energy={:.3}, first_10={:?}",
                    total_energy,
                    &spectrum_data[..10]
                );
            }

            // Write the spectrum data to the buffer
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&spectrum_data));
        }
    }

    /// Get output texture for display
    pub fn get_output_texture(&self) -> &TextureManager {
        &self.output_texture
    }

    /// Resize resources
    pub fn resize(&mut self, core: &Core, width: u32, height: u32) {
        // Recreate output texture
        self.output_texture = Self::create_output_texture(
            &core.device,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            &format!("{} Output Texture", self.label),
        );

        // CRITICAL: Recreate Group 1 bind group with new texture!
        let group1_layout = self.bind_group_layouts.get(&1).unwrap();
        self.group1_bind_group = Self::create_group1_bind_group(
            &core.device,
            group1_layout,
            &self.output_texture,
            &ComputeConfiguration {
                entry_points: self.entry_points.clone(),
                passes: None,
                custom_uniform_size: self.custom_uniform_size,
                has_input_texture: self.has_input_texture,
                has_mouse: true,
                has_fonts: true,
                has_audio: true,
                has_atomic_buffer: false,
                audio_buffer_size: 1024,
                has_audio_spectrum: false,
                audio_spectrum_size: 128,
                storage_buffers: Vec::new(),
                workgroup_size: self.workgroup_size,
                dispatch_once: self.dispatch_once,
                texture_format: wgpu::TextureFormat::Rgba16Float,
                label: self.label.clone(),
                num_channels: Some(self.num_channels),
            },
            self.custom_uniform.as_ref(),
            self.placeholder_input_texture.as_ref().map(|t| &t.view),
            self.placeholder_input_texture.as_ref().map(|t| &t.sampler),
        );

        // Resize multi-pass buffers if present
        if let Some(multipass) = &mut self.multipass_manager {
            multipass.resize(core, width, height);
        }

        // Recreate atomic buffer if present (like clear_atomic_buffer)
        if let Some(atomic_buffer) = &mut self.atomic_buffer_raw {
            let buffer_size = (width * height * 3 * 4) as u64; // 3 u32s * 4 bytes per pixel
            *atomic_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Atomic Storage Buffer (resized)"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Recreate group2 bind group with the new buffer
            if let Some(layout) = self.bind_group_layouts.get(&2) {
                self.group2_bind_group = Self::create_group2_bind_group(
                    &core.device,
                    &core.queue,
                    layout,
                    &self.font_system,
                    &self.atomic_buffer_raw,
                    &self.audio_buffer,
                    &self.audio_spectrum_buffer,
                    &self.mouse_uniform,
                    &self.channel_textures,
                    self.num_channels,
                );
            }
        }

        // Reset frame counter on resize to start fresh
        self.current_frame = 0;
    }

    /// Clear all buffers (atomic or multipass)
    pub fn clear_all_buffers(&mut self, core: &Core) {
        // Clear multipass buffers if present
        if let Some(multipass) = &mut self.multipass_manager {
            multipass.clear_all(core);
        }

        // Clear atomic buffer if present
        self.clear_atomic_buffer(core);

        // Reset frame counter
        self.current_frame = 0;
    }

    /// Clear atomic buffer by recreating it (like old clear_all method)
    pub fn clear_atomic_buffer(&mut self, core: &Core) {
        if self.atomic_buffer_raw.is_some() {
            // Recreate the atomic buffer entirely (more thorough than just writing zeros)
            let buffer_size = (core.size.width * core.size.height * 3 * 4) as u64; // 3 u32s * 4 bytes per pixel
            self.atomic_buffer_raw = Some(core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Atomic Storage Buffer (cleared)"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Recreate group2 bind group with the new buffer
            if let Some(layout) = self.bind_group_layouts.get(&2) {
                self.group2_bind_group = Self::create_group2_bind_group(
                    &core.device,
                    &core.queue,
                    layout,
                    &self.font_system,
                    &self.atomic_buffer_raw,
                    &self.audio_buffer,
                    &self.audio_spectrum_buffer,
                    &self.mouse_uniform,
                    &self.channel_textures,
                    self.num_channels,
                );
            }
        }
    }

    /// Update mouse uniform with data from RenderKit
    pub fn update_mouse_uniform(
        &mut self,
        mouse_uniform_data: &crate::MouseUniform,
        queue: &wgpu::Queue,
    ) {
        if let Some(mouse_uniform) = &mut self.mouse_uniform {
            mouse_uniform.data = *mouse_uniform_data;
            mouse_uniform.update(queue);
        }
    }
    pub fn get_audio_buffer(&self) -> Option<&wgpu::Buffer> {
        self.audio_buffer.as_ref()
    }

    /// Reads audio data from the GPU's audio buffer back to CPU.
    ///
    /// This method copies audio data from the GPU compute shader's audio buffer
    /// to CPU memory for processing or playback. The GPU shader writes audio
    /// parameters (frequencies, amplitudes, waveforms, etc.) to the buffer,
    /// and this method retrieves them asynchronously.
    ///
    /// Returns a Vec<f32> containing the audio buffer data, or an empty vector if no audio buffer exists.
    pub async fn read_audio_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if let (Some(audio_buffer), Some(staging_buffer)) =
            (&self.audio_buffer, &self.audio_staging_buffer)
        {
            // Get buffer size directly from the wgpu buffer itself
            let buffer_size = audio_buffer.size();

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Audio Buffer Copy"),
            });

            encoder.copy_buffer_to_buffer(audio_buffer, 0, staging_buffer, 0, buffer_size as u64);

            queue.submit(std::iter::once(encoder.finish()));

            let buffer_slice = staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            let _ = device.poll(wgpu::PollType::wait_indefinitely());

            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => return Err("Buffer mapping failed".into()),
            }

            let samples = {
                let data = buffer_slice.get_mapped_range();
                let samples: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                samples
            };

            staging_buffer.unmap();

            Ok(samples)
        } else {
            Ok(Vec::new())
        }
    }

    /// Automatic export - call from shader update() method
    pub fn handle_export(&mut self, core: &Core, render_kit: &mut crate::RenderKit) {
        if let Some((frame, time)) = render_kit.export_manager.try_get_next_frame() {
            match self.capture_export_frame(
                core,
                time,
                render_kit,
                None::<fn(&mut Self, &mut wgpu::CommandEncoder, &Core)>,
            ) {
                Ok(data) => {
                    let settings = render_kit.export_manager.settings();
                    if let Err(e) = crate::save_frame(data, frame, settings) {
                        eprintln!("Error saving frame: {:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Error capturing export frame: {:?}", e);
                }
            }
        } else {
            render_kit.export_manager.complete_export();
        }
    }

    /// Automatic export with custom dispatch
    pub fn handle_export_dispatch(
        &mut self,
        core: &Core,
        render_kit: &mut crate::RenderKit,
        custom_dispatch: impl FnOnce(&mut Self, &mut wgpu::CommandEncoder, &Core),
    ) {
        if let Some((frame, time)) = render_kit.export_manager.try_get_next_frame() {
            match self.capture_export_frame(core, time, render_kit, Some(custom_dispatch)) {
                Ok(data) => {
                    let settings = render_kit.export_manager.settings();
                    if let Err(e) = crate::save_frame(data, frame, settings) {
                        eprintln!("Error saving frame: {:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Error capturing export frame: {:?}", e);
                }
            }
        } else {
            render_kit.export_manager.complete_export();
        }
    }

    /// Captures current frame with format conversion and optional custom dispatch
    pub fn capture_export_frame<F>(
        &mut self,
        core: &Core,
        time: f32,
        render_kit: &crate::RenderKit,
        custom_dispatch: Option<F>,
    ) -> Result<Vec<u8>, wgpu::SurfaceError>
    where
        F: FnOnce(&mut Self, &mut wgpu::CommandEncoder, &Core),
    {
        let settings = render_kit.export_manager.settings();
        let (capture_texture, output_buffer) =
            render_kit.create_capture_texture(&core.device, settings.width, settings.height);

        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Export Encoder"),
            });

        self.set_time(time, 0.0, &core.queue);

        // Use custom dispatch if provided, otherwise use default
        if let Some(custom_dispatch) = custom_dispatch {
            custom_dispatch(self, &mut encoder, core);
        } else {
            self.dispatch(&mut encoder, core);
        }

        {
            let mut render_pass = crate::Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Export Capture Pass"),
            );

            render_pass.set_pipeline(&render_kit.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, render_kit.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.output_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        let align = 256;
        let unpadded_bytes_per_row = settings.width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;

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

        let _ = core
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();

        let padded_data = buffer_slice.get_mapped_range().to_vec();
        let mut unpadded_data = Vec::with_capacity((settings.width * settings.height * 4) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }

        Ok(unpadded_data)
    }
}
