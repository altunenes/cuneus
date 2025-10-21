pub struct AtomicBuffer {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub size: u32,
}

impl AtomicBuffer {
    pub fn new(device: &wgpu::Device, size: u32, layout: &wgpu::BindGroupLayout) -> Self {
        let buffer_size = (size * 4 * 4) as u64;
        let max_binding_size = device.limits().max_storage_buffer_binding_size as u64;
        let max_size = (max_binding_size / (4 * 4)) as u32;

        let (actual_size, actual_buffer_size) = if buffer_size > max_binding_size {
            println!(
                "Requested buffer size {} exceeds device max_storage_buffer_binding_size {}. Reducing size to {}.",
                buffer_size, max_binding_size, max_size
            );
            (max_size, max_binding_size)
        } else {
            (size, buffer_size)
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atomic Buffer"),
            size: actual_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("Atomic Buffer Bind Group"),
        });

        Self {
            buffer,
            bind_group,
            size: actual_size,
        }
    }

    pub fn clear(&self, queue: &wgpu::Queue) {
        let clear_data = vec![0u32; (self.size * 4) as usize];
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&clear_data));
    }
}
