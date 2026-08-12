use cuneus::GaussianSorter;
use wgpu::util::DeviceExt;

#[test]
fn depth_shift_uses_enough_radix_bits_and_sorts_wide_keys() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        {
            Ok(adapter) => adapter,
            Err(_) => return,
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Gaussian depth-key regression device"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("device");

        const DEPTH_SHIFT: u32 = 8;
        let mut pairs: Vec<_> = (0_u32..4_097)
            .map(|index| {
                let depth = 0.01 + ((index * 2_053) % 4_097) as f32 * 0.01;
                ((u32::MAX - depth.to_bits()) >> DEPTH_SHIFT, index)
            })
            .collect();
        pairs.reverse();
        let count = pairs.len();
        let padded_count = count.div_ceil(3_840) * 3_840;
        let mut keys: Vec<_> = pairs.iter().map(|&(key, _)| key).collect();
        let mut payloads: Vec<_> = pairs.iter().map(|&(_, payload)| payload).collect();
        keys.resize(padded_count, u32::MAX);
        payloads.resize(padded_count, u32::MAX);
        let mut expected = pairs.clone();
        expected.sort_by_key(|&(key, _)| key);
        let mut low_16_bit_order = pairs.clone();
        low_16_bit_order.sort_by_key(|&(key, _)| key & 0xffff);
        assert_ne!(low_16_bit_order, expected, "dataset must expose truncation");

        let keys_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian depth-key regression keys"),
            contents: bytemuck::cast_slice(&keys),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let payloads_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian depth-key regression payloads"),
            contents: bytemuck::cast_slice(&payloads),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let bytes = keys_buffer.size();
        let keys_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian depth-key regression keys readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let payloads_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian depth-key regression payloads readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut sorter = GaussianSorter::for_depth_shift(&device, DEPTH_SHIFT);
        sorter.prepare_with_buffers(&device, &keys_buffer, &payloads_buffer, count as u32);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gaussian depth-key regression encoder"),
        });
        sorter.sort(&mut encoder, count as u32);
        encoder.copy_buffer_to_buffer(&keys_buffer, 0, &keys_readback, 0, bytes);
        encoder.copy_buffer_to_buffer(&payloads_buffer, 0, &payloads_readback, 0, bytes);
        let submission = queue.submit([encoder.finish()]);

        let key_slice = keys_readback.slice(..);
        let (key_sender, key_receiver) = std::sync::mpsc::channel();
        key_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = key_sender.send(result);
        });
        let payload_slice = payloads_readback.slice(..);
        let (payload_sender, payload_receiver) = std::sync::mpsc::channel();
        payload_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = payload_sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("sort completion");
        key_receiver
            .recv()
            .expect("key map callback")
            .expect("key map");
        payload_receiver
            .recv()
            .expect("payload map callback")
            .expect("payload map");
        let sorted_keys: Vec<u32> = bytemuck::cast_slice(&key_slice.get_mapped_range()).to_vec();
        let sorted_payloads: Vec<u32> =
            bytemuck::cast_slice(&payload_slice.get_mapped_range()).to_vec();
        let actual: Vec<_> = sorted_keys
            .into_iter()
            .zip(sorted_payloads)
            .take(count)
            .collect();
        assert_eq!(actual, expected);
    });
}
