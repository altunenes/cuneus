use crate::TextureManager;
use image::codecs::hdr::HdrDecoder;
use image::{ImageDecoder, RgbaImage};
use std::io::Cursor;

#[derive(Clone, Debug, Copy)]
pub struct HdriMetadata {
    pub width: u32,
    pub height: u32,
    pub exposure: f32,
    pub gamma: f32,
}

impl Default for HdriMetadata {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            exposure: 1.0,
            gamma: 2.2,
        }
    }
}

pub fn load_hdri_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    layout: &wgpu::BindGroupLayout,
    exposure: f32,
) -> Result<(TextureManager, HdriMetadata), String> {
    let format = detect_format(data)?;
    let gamma = 2.2;
    let hdri_image = match format {
        HdriFormat::Hdr => hdr_to_rgba8(data, exposure, Some(gamma))?,
        HdriFormat::Exr => exr_to_rgba8(data, exposure, Some(gamma))?,
    };
    let dimensions = hdri_image.dimensions();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("HDRI Texture"),
        size: wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
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
        label: Some("HDRI Texture Bind Group"),
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &hdri_image,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * dimensions.0),
            rows_per_image: Some(dimensions.1),
        },
        wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        },
    );

    let metadata = HdriMetadata {
        width: dimensions.0,
        height: dimensions.1,
        exposure,
        gamma,
    };

    Ok((
        TextureManager {
            texture,
            view,
            sampler,
            bind_group,
        },
        metadata,
    ))
}

enum HdriFormat {
    Hdr,
    Exr,
}

fn detect_format(data: &[u8]) -> Result<HdriFormat, String> {
    if data.len() >= 4 && data[0] == 0x76 && data[1] == 0x2f && data[2] == 0x31 && data[3] == 0x01 {
        return Ok(HdriFormat::Exr);
    }
    let start_bytes = if data.len() >= 10 { &data[0..10] } else { data };
    let start_str = String::from_utf8_lossy(start_bytes);
    if start_str.starts_with("#?RADIANCE") || start_str.starts_with("#?RGBE") {
        return Ok(HdriFormat::Hdr);
    }
    // I know looks NOT good to you :-P. This could be improved with more robust format detection
    Ok(HdriFormat::Hdr)
}

fn hdr_to_rgba8(hdr_data: &[u8], exposure: f32, gamma: Option<f32>) -> Result<RgbaImage, String> {
    let cursor = Cursor::new(hdr_data);
    let decoder = HdrDecoder::new(cursor).map_err(|e| e.to_string())?;
    let metadata = decoder.metadata();
    let dynamic_img = image::DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("Failed to decode HDR: {}", e))?;
    let mut rgba8_image = RgbaImage::new(metadata.width, metadata.height);
    let rgb8_image = dynamic_img.to_rgb8();
    let gamma_value = gamma.unwrap_or(2.2);
    let gamma_correction = 1.0 / gamma_value;
    for (x, y, pixel) in rgb8_image.enumerate_pixels() {
        let r_linear = (pixel[0] as f32 / 255.0) * exposure;
        let g_linear = (pixel[1] as f32 / 255.0) * exposure;
        let b_linear = (pixel[2] as f32 / 255.0) * exposure;
        let r = ((r_linear.powf(gamma_correction)).min(1.0) * 255.0) as u8;
        let g = ((g_linear.powf(gamma_correction)).min(1.0) * 255.0) as u8;
        let b = ((b_linear.powf(gamma_correction)).min(1.0) * 255.0) as u8;
        rgba8_image.put_pixel(x, y, image::Rgba([r, g, b, 255]));
    }
    Ok(rgba8_image)
}

fn exr_to_rgba8(exr_data: &[u8], exposure: f32, gamma: Option<f32>) -> Result<RgbaImage, String> {
    use image::codecs::openexr::OpenExrDecoder;

    let cursor = Cursor::new(exr_data);
    let decoder =
        OpenExrDecoder::new(cursor).map_err(|e| format!("Failed to decode EXR: {}", e))?;
    let (width, height) = decoder.dimensions();
    let dynamic_img = image::DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("Failed to create DynamicImage from EXR: {}", e))?;
    let rgba_float = dynamic_img.to_rgba32f();
    let mut rgba8_image = RgbaImage::new(width, height);
    let gamma_value = gamma.unwrap_or(2.2);
    let gamma_correction = 1.0 / gamma_value;
    for (x, y, pixel) in rgba_float.enumerate_pixels() {
        let r_linear = pixel[0] * exposure;
        let g_linear = pixel[1] * exposure;
        let b_linear = pixel[2] * exposure;
        let a = pixel[3];

        let r = ((r_linear.powf(gamma_correction)).min(1.0) * 255.0) as u8;
        let g = ((g_linear.powf(gamma_correction)).min(1.0) * 255.0) as u8;
        let b = ((b_linear.powf(gamma_correction)).min(1.0) * 255.0) as u8;
        let a = (a.min(1.0) * 255.0) as u8;

        rgba8_image.put_pixel(x, y, image::Rgba([r, g, b, a]));
    }

    Ok(rgba8_image)
}

pub fn update_hdri_exposure(
    _device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    _layout: &wgpu::BindGroupLayout,
    texture_manager: &mut TextureManager,
    new_exposure: f32,
    gamma: Option<f32>,
) -> Result<(), String> {
    let format = detect_format(data)?;
    let rgba_image = match format {
        HdriFormat::Hdr => hdr_to_rgba8(data, new_exposure, gamma)?,
        HdriFormat::Exr => exr_to_rgba8(data, new_exposure, gamma)?,
    };
    texture_manager.update(queue, &rgba_image);
    Ok(())
}
