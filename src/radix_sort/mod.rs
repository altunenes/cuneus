//! GPU Radix Sort for cuneus
//!
//! A GPU-based radix sort implementation for sorting key-value pairs.
//! Ported from wgpu_sort (BSD 2-Clause License) : https://github.com/KeKsBoTer/wgpu_sort/tree/master/src
//! please see the original repo for code explanations in detail. 


use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

const HISTOGRAM_WG_SIZE: u32 = 256;
const PREFIX_WG_SIZE: u32 = 128;
const SCATTER_WG_SIZE: u32 = 256;
const RS_RADIX_LOG2: u32 = 8;
const RS_RADIX_SIZE: u32 = 1 << RS_RADIX_LOG2;
const RS_KEYVAL_SIZE: u32 = 4;
const RS_HISTOGRAM_BLOCK_ROWS: u32 = 15;
const RS_SCATTER_BLOCK_ROWS: u32 = RS_HISTOGRAM_BLOCK_ROWS;
const HISTO_BLOCK_KVS: u32 = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;
const SCATTER_BLOCK_KVS: u32 = SCATTER_WG_SIZE * RS_SCATTER_BLOCK_ROWS;

/// State structure for the sorter
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SorterState {
    pub num_keys: u32,
    pub padded_size: u32,
    pub even_pass: u32,
    pub odd_pass: u32,
    pub sort_failed: u32,
}

/// GPU Radix Sorter for key-value pairs
pub struct RadixSorter {
    zero_pipeline: wgpu::ComputePipeline,
    histogram_pipeline: wgpu::ComputePipeline,
    prefix_pipeline: wgpu::ComputePipeline,
    scatter_even_pipeline: wgpu::ComputePipeline,
    scatter_odd_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl RadixSorter {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Radix Sort Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let subgroup_size = 1u32;
        let rs_sweep_0_size = RS_RADIX_SIZE / subgroup_size.max(1);
        let rs_sweep_1_size = rs_sweep_0_size / subgroup_size.max(1);
        let _rs_sweep_2_size = rs_sweep_1_size / subgroup_size.max(1);
        let rs_smem_phase_2 = RS_RADIX_SIZE + RS_SCATTER_BLOCK_ROWS * SCATTER_WG_SIZE;
        let rs_mem_dwords = rs_smem_phase_2;

        // Build shader with constants
        let shader_source = format!(
            "const histogram_sg_size: u32 = {}u;\n\
             const histogram_wg_size: u32 = {}u;\n\
             const rs_radix_log2: u32 = {}u;\n\
             const rs_radix_size: u32 = {}u;\n\
             const rs_keyval_size: u32 = {}u;\n\
             const rs_histogram_block_rows: u32 = {}u;\n\
             const rs_scatter_block_rows: u32 = {}u;\n\
             const rs_mem_dwords: u32 = {}u;\n\
             const rs_mem_sweep_0_offset: u32 = 0u;\n\
             const rs_mem_sweep_1_offset: u32 = {}u;\n\
             const rs_mem_sweep_2_offset: u32 = {}u;\n\
             {}",
            subgroup_size.max(1),
            HISTOGRAM_WG_SIZE,
            RS_RADIX_LOG2,
            RS_RADIX_SIZE,
            RS_KEYVAL_SIZE,
            RS_HISTOGRAM_BLOCK_ROWS,
            RS_SCATTER_BLOCK_ROWS,
            rs_mem_dwords,
            rs_sweep_0_size,
            rs_sweep_0_size + rs_sweep_1_size,
            include_str!("shader.wgsl")
        );

        let shader_code = shader_source
            .replace("{histogram_wg_size}", &HISTOGRAM_WG_SIZE.to_string())
            .replace("{prefix_wg_size}", &PREFIX_WG_SIZE.to_string())
            .replace("{scatter_wg_size}", &SCATTER_WG_SIZE.to_string());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Radix Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        let zero_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Radix Sort Zero"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("zero_histograms"),
            compilation_options: Default::default(),
            cache: None,
        });

        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Radix Sort Histogram"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("calculate_histogram"),
            compilation_options: Default::default(),
            cache: None,
        });

        let prefix_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Radix Sort Prefix"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("prefix_histogram"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_even_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Radix Sort Scatter Even"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_even"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_odd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Radix Sort Scatter Odd"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_odd"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            zero_pipeline,
            histogram_pipeline,
            prefix_pipeline,
            scatter_even_pipeline,
            scatter_odd_pipeline,
            bind_group_layout,
        }
    }

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Radix Sort Bind Group Layout"),
            entries: &[
                // State buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(std::mem::size_of::<SorterState>() as u64).unwrap()),
                    },
                    count: None,
                },
                // Internal memory (histograms)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Keys A
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Keys B
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Payload A (indices)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Payload B
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create sort buffers for a given number of elements
    pub fn create_sort_buffers(&self, device: &wgpu::Device, count: u32) -> SortBuffers {
        let padded_size = keys_buffer_size(count);
        let keys_size = (padded_size * RS_KEYVAL_SIZE * 4) as u64;
        let payload_size = (count * 4) as u64;

        let state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Radix Sort State"),
            contents: bytemuck::bytes_of(&SorterState {
                num_keys: count,
                padded_size,
                even_pass: 0,
                odd_pass: 0,
                sort_failed: 0,
            }),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let internal_size = self.internal_buffer_size(count);
        let internal_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Sort Internal"),
            size: internal_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let keys_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Sort Keys A"),
            size: keys_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let keys_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Sort Keys B"),
            size: keys_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let payload_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Sort Payload A"),
            size: payload_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let payload_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Sort Payload B"),
            size: payload_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Radix Sort Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: internal_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: keys_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: keys_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: payload_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: payload_b.as_entire_binding() },
            ],
        });

        SortBuffers {
            state_buffer,
            internal_buffer,
            keys_a,
            keys_b,
            payload_a,
            payload_b,
            bind_group,
            count,
        }
    }

    fn internal_buffer_size(&self, count: u32) -> u32 {
        let scatter_blocks_ru = scatter_blocks_ru(count);
        let histo_size = RS_RADIX_SIZE * 4;
        (RS_KEYVAL_SIZE + scatter_blocks_ru) * histo_size
    }

    /// Sort the keys and payload
    pub fn sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        buffers: &SortBuffers,
        count: u32,
    ) {
        // Update count
        queue.write_buffer(&buffers.state_buffer, 0, bytemuck::bytes_of(&count));

        let hist_blocks = histo_blocks_ru(count);
        let scatter_blocks = scatter_blocks_ru(count);

        // Zero histograms
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Radix Sort Zero"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.zero_pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups(hist_blocks, 1, 1);
        }

        // Calculate histogram
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Radix Sort Histogram"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups(hist_blocks, 1, 1);
        }

        // Prefix sum
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Radix Sort Prefix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prefix_pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups(RS_KEYVAL_SIZE, 1, 1);
        }

        // Scatter passes (4 passes for 32-bit keys)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Radix Sort Scatter"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &buffers.bind_group, &[]);

            pass.set_pipeline(&self.scatter_even_pipeline);
            pass.dispatch_workgroups(scatter_blocks, 1, 1);

            pass.set_pipeline(&self.scatter_odd_pipeline);
            pass.dispatch_workgroups(scatter_blocks, 1, 1);

            pass.set_pipeline(&self.scatter_even_pipeline);
            pass.dispatch_workgroups(scatter_blocks, 1, 1);

            pass.set_pipeline(&self.scatter_odd_pipeline);
            pass.dispatch_workgroups(scatter_blocks, 1, 1);
        }
    }

    /// Create a bind group that directly binds to external depth_keys and sorted_indices buffers
    /// Returns (bind_group, aux_keys, aux_payload, internal_buffer, state_buffer)
    pub fn create_direct_bind_group(
        &self,
        device: &wgpu::Device,
        depth_keys_buffer: &wgpu::Buffer,
        sorted_indices_buffer: &wgpu::Buffer,
        count: u32,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
        let padded_size = keys_buffer_size(count);
        let keys_aux_size = (padded_size * RS_KEYVAL_SIZE * 4) as u64;
        let payload_aux_size = (count * 4) as u64;

        let state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Radix Sort State Direct"),
            contents: bytemuck::bytes_of(&SorterState {
                num_keys: count,
                padded_size,
                even_pass: 0,
                odd_pass: 0,
                sort_failed: 0,
            }),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let internal_size = self.internal_buffer_size(count);
        let internal_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Sort Internal Direct"),
            size: internal_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let keys_aux = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Sort Keys Aux"),
            size: keys_aux_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let payload_aux = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Sort Payload Aux"),
            size: payload_aux_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Radix Sort Direct Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: internal_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: depth_keys_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: keys_aux.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: sorted_indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: payload_aux.as_entire_binding() },
            ],
        });

        (state_buffer, keys_aux, payload_aux, internal_buffer, bind_group)
    }

    /// Sort using a pre-created bind group (no CPU buffer writes during sort)
    pub fn sort_with_bind_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        count: u32,
    ) {
        let hist_blocks = histo_blocks_ru(count);
        let scatter_blocks = scatter_blocks_ru(count);

        // Zero histograms
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Radix Sort Zero"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.zero_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(hist_blocks, 1, 1);
        }

        // Calculate histogram
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Radix Sort Histogram"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(hist_blocks, 1, 1);
        }

        // Prefix sum
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Radix Sort Prefix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prefix_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(RS_KEYVAL_SIZE, 1, 1);
        }

        // Scatter passes (4 passes for 32-bit keys)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Radix Sort Scatter"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bind_group, &[]);

            pass.set_pipeline(&self.scatter_even_pipeline);
            pass.dispatch_workgroups(scatter_blocks, 1, 1);

            pass.set_pipeline(&self.scatter_odd_pipeline);
            pass.dispatch_workgroups(scatter_blocks, 1, 1);

            pass.set_pipeline(&self.scatter_even_pipeline);
            pass.dispatch_workgroups(scatter_blocks, 1, 1);

            pass.set_pipeline(&self.scatter_odd_pipeline);
            pass.dispatch_workgroups(scatter_blocks, 1, 1);
        }
    }
}

/// Buffers for radix sorting
pub struct SortBuffers {
    pub state_buffer: wgpu::Buffer,
    pub internal_buffer: wgpu::Buffer,
    pub keys_a: wgpu::Buffer,
    pub keys_b: wgpu::Buffer,
    pub payload_a: wgpu::Buffer,
    pub payload_b: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub count: u32,
}

impl SortBuffers {
    /// Get the keys buffer (sorted output)
    pub fn keys(&self) -> &wgpu::Buffer {
        &self.keys_a
    }

    /// Get the payload/values buffer (sorted output)
    pub fn values(&self) -> &wgpu::Buffer {
        &self.payload_a
    }
}

fn scatter_blocks_ru(n: u32) -> u32 {
    n.div_ceil(SCATTER_BLOCK_KVS)
}

fn histo_blocks_ru(n: u32) -> u32 {
    (scatter_blocks_ru(n) * SCATTER_BLOCK_KVS).div_ceil(HISTO_BLOCK_KVS)
}

fn keys_buffer_size(n: u32) -> u32 {
    histo_blocks_ru(n) * HISTO_BLOCK_KVS
}
