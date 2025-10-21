use crate::{Core, TextureManager};
#[derive(Clone)]
pub struct FeedbackTextureConfig {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub sample_count: u32,
    pub mip_level_count: u32,
    pub dimension: wgpu::TextureDimension,
}

impl Default for FeedbackTextureConfig {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            sample_count: 1,
            mip_level_count: 1,
            dimension: wgpu::TextureDimension::D2,
        }
    }
}

pub fn create_feedback_texture(
    core: &Core,
    config: FeedbackTextureConfig,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> TextureManager {
    let texture = core.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Feedback Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: config.mip_level_count,
        sample_count: config.sample_count,
        dimension: config.dimension,
        format: config.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Feedback Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: texture_bind_group_layout,
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
        label: Some("Feedback Texture Bind Group"),
    });

    TextureManager {
        texture,
        view,
        sampler,
        bind_group,
    }
}

// Helper function to create a pair of feedback textures
pub fn create_feedback_texture_pair(
    core: &Core,
    width: u32,
    height: u32,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> (TextureManager, TextureManager) {
    let config = FeedbackTextureConfig {
        width,
        height,
        format: core.config.format,
        ..Default::default()
    };

    let texture_a = create_feedback_texture(core, config.clone(), texture_bind_group_layout);
    let texture_b = create_feedback_texture(core, config, texture_bind_group_layout);

    (texture_a, texture_b)
}
