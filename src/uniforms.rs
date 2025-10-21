use wgpu::util::DeviceExt;
pub trait UniformProvider {
    fn as_bytes(&self) -> &[u8];
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResolutionUniform {
    pub dimensions: [f32; 2],
    pub _padding: [f32; 2],
    pub audio_data: [[f32; 4]; 32],
    pub bpm: f32,
    pub _bpm_padding: [f32; 3],
}

impl UniformProvider for ResolutionUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

pub struct UniformBinding<T: UniformProvider> {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub data: T,
}
impl<T: UniformProvider> UniformBinding<T> {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        data: T,
        layout: &wgpu::BindGroupLayout,
        binding: u32,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data.as_bytes(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }],
            label: Some(label),
        });
        Self {
            buffer,
            bind_group,
            data,
        }
    }
    pub fn update(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, self.data.as_bytes());
    }
}
