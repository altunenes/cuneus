use crate::{Core, TextureManager, UniformBinding, UniformProvider};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

// font system using texture atlas

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FontUniforms {
    pub atlas_size: [f32; 2],
    pub char_size: [f32; 2],
    pub screen_size: [f32; 2],
    pub grid_size: [f32; 2],
}

impl UniformProvider for FontUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CharInfo {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub char_code: u8,
}

pub struct FontSystem {
    pub atlas_texture: TextureManager,
    pub char_map: HashMap<char, CharInfo>,
    pub font_uniforms: UniformBinding<FontUniforms>,
    pub font_bind_group_layout: wgpu::BindGroupLayout,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub grid_size: u32,
    pub char_size: u32,
}

impl FontSystem {
    pub fn new(core: &Core) -> Self {
        //note that: I always use following:
        // _ATLAS_SIZE: u32 = 1024;
        // _CELL_SIZE: u32 = 64;
        // _GRID_SIZE: u32 = 16;
        let font_texture_bytes = include_bytes!("../assets/fonts/fonttexture.png");
        let font_image = image::load_from_memory(font_texture_bytes)
            .expect("Failed to load font texture")
            .into_rgba8();

        let atlas_width = font_image.width();
        let atlas_height = font_image.height();
        let grid_size = 16u32;
        let char_size = atlas_width / grid_size;

        let font_bind_group_layout =
            core.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                    label: Some("Font Bind Group Layout"),
                });

        let font_uniforms_data = FontUniforms {
            atlas_size: [atlas_width as f32, atlas_height as f32],
            char_size: [char_size as f32, char_size as f32],
            screen_size: [core.size.width as f32, core.size.height as f32],
            grid_size: [grid_size as f32, grid_size as f32],
        };

        let font_uniform_layout =
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
                    label: Some("Font Uniforms Layout"),
                });

        let font_uniforms = UniformBinding::new(
            &core.device,
            "Font Uniforms",
            font_uniforms_data,
            &font_uniform_layout,
            0,
        );

        let atlas_texture = Self::create_font_texture(core, &font_image);
        let char_map = Self::generate_character_map(grid_size);

        Self {
            atlas_texture,
            char_map,
            font_uniforms,
            font_bind_group_layout,
            atlas_width,
            atlas_height,
            grid_size,
            char_size,
        }
    }

    fn create_font_texture(core: &Core, font_image: &image::RgbaImage) -> TextureManager {
        let (width, height) = font_image.dimensions();

        let texture = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadertoy Font Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        core.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            font_image.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_bind_group_layout =
            core.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Font Texture Display Layout"),
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

        let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
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
            label: Some("Font Texture Atlas Bind Group"),
        });

        TextureManager {
            texture,
            view,
            sampler,
            bind_group,
        }
    }

    fn generate_character_map(grid_size: u32) -> HashMap<char, CharInfo> {
        let mut char_map = HashMap::new();

        for ascii_code in 32..127 {
            let char = ascii_code as u8 as char;
            let grid_index = ascii_code as usize;

            if grid_index >= 256 {
                break;
            }

            let grid_x = grid_index % (grid_size as usize);
            let grid_y = grid_index / (grid_size as usize);

            let char_info = CharInfo {
                uv_min: [
                    grid_x as f32 / grid_size as f32,
                    grid_y as f32 / grid_size as f32,
                ],
                uv_max: [
                    (grid_x + 1) as f32 / grid_size as f32,
                    (grid_y + 1) as f32 / grid_size as f32,
                ],
                char_code: ascii_code as u8,
            };

            char_map.insert(char, char_info);
        }

        char_map
    }

    pub fn update_screen_size(&mut self, width: u32, height: u32, queue: &wgpu::Queue) {
        self.font_uniforms.data.screen_size = [width as f32, height as f32];
        self.font_uniforms.update(queue);
    }

    pub fn get_char_info(&self, ch: char) -> Option<&CharInfo> {
        self.char_map.get(&ch)
    }

    pub fn create_font_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.font_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.font_uniforms.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.atlas_texture.view),
                },
            ],
            label: Some("Font Bind Group"),
        })
    }

    pub fn get_atlas_dimensions(&self) -> (u32, u32) {
        (self.atlas_width, self.atlas_height)
    }

    pub fn get_char_size(&self) -> u32 {
        self.char_size
    }

    pub fn get_grid_size(&self) -> u32 {
        self.grid_size
    }
}
