// An example of a simple audio creation using Cuneus
@group(0) @binding(0) var<uniform> u_time: ComputeTimeUniform;
@group(1) @binding(0) var output_texture: texture_storage_2d<rgba16float, write>;
@group(2) @binding(0) var<uniform> u_audio_synth: AudioSynthUniform;

struct ComputeTimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}

struct AudioSynthUniform {
    note_frequencies: array<vec4<f32>, 4>,
    note_amplitudes: array<vec4<f32>, 4>,
    master_volume: f32,
    waveform_type: u32,
    active_note_count: u32,
    _padding: u32,
}

// Generate waveform value at screen position based on active notes
fn get_synth_waveform_value(x: f32, y: f32) -> f32 {
    if (u_audio_synth.active_note_count == 0u) {
        return 0.0;
    }
    
    var total_amplitude = 0.0;
    
    // Combine all active notes into a waveform
    for (var i = 0u; i < u_audio_synth.active_note_count && i < 16u; i++) {
        let vec4_index = i / 4u;
        let component_index = i % 4u;
        
        var frequency = 0.0;
        var amplitude = 0.0;
        
        if (component_index == 0u) {
            frequency = u_audio_synth.note_frequencies[vec4_index].x;
            amplitude = u_audio_synth.note_amplitudes[vec4_index].x;
        } else if (component_index == 1u) {
            frequency = u_audio_synth.note_frequencies[vec4_index].y;
            amplitude = u_audio_synth.note_amplitudes[vec4_index].y;
        } else if (component_index == 2u) {
            frequency = u_audio_synth.note_frequencies[vec4_index].z;
            amplitude = u_audio_synth.note_amplitudes[vec4_index].z;
        } else {
            frequency = u_audio_synth.note_frequencies[vec4_index].w;
            amplitude = u_audio_synth.note_amplitudes[vec4_index].w;
        }
        
        if (frequency > 0.0 && amplitude > 0.0) {
            // Create waveform based on frequency and time
            let wave_phase = x * frequency * 0.01 + u_time.time * frequency * 0.1;
            
            var wave_value = 0.0;
            if (u_audio_synth.waveform_type == 0u) {
                // Sine wave
                wave_value = sin(wave_phase);
            } else if (u_audio_synth.waveform_type == 1u) {
                // Square wave
                wave_value = select(-1.0, 1.0, sin(wave_phase) > 0.0);
            } else if (u_audio_synth.waveform_type == 2u) {
                // Saw wave
                wave_value = (wave_phase % 6.28318) / 3.14159 - 1.0;
            } else {
                // Triangle wave
                let normalized = (wave_phase % 6.28318) / 6.28318;
                wave_value = select(4.0 * normalized - 1.0, 3.0 - 4.0 * normalized, normalized > 0.5);
            }
            
            total_amplitude += wave_value * amplitude;
        }
    }
    
    return total_amplitude * u_audio_synth.master_volume;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    let coord = vec2<i32>(global_id.xy);
    
    if coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
        return;
    }
    
    let uv = vec2<f32>(coord) / vec2<f32>(dims);
    
    var color = vec3<f32>(0.0, 0.0, 0.0);
    
    let wave_center_y = 0.5;
    let wave_height = 0.2;
    
    //synthesized waveform value at this position
    let synth_value = get_synth_waveform_value(uv.x, uv.y);
    
    // Create waveform visualization
    let wave_y = wave_center_y + synth_value * wave_height;
    
    let distance_to_wave = abs(uv.y - wave_y);
    
    if distance_to_wave < 0.002 {
        let intensity = abs(synth_value) * u_audio_synth.master_volume;
        color = vec3<f32>(intensity, intensity * 0.8, intensity * 0.3);
    } else if distance_to_wave < 0.006 && u_audio_synth.active_note_count > 0u {
        let glow = (1.0 - distance_to_wave / 0.006) * u_audio_synth.master_volume * 0.3;
        color = vec3<f32>(glow * 0.7, glow * 0.9, glow * 0.4);
    }
    
    if u_audio_synth.active_note_count > 0u {
        for (var i = 0u; i < u_audio_synth.active_note_count && i < 16u; i++) {
            let note_x = f32(i) / 16.0;
            let note_distance = abs(uv.x - note_x);
            
            let vec4_index = i / 4u;
            let component_index = i % 4u;
            
            var note_amplitude = 0.0;
            if (component_index == 0u) {
                note_amplitude = u_audio_synth.note_amplitudes[vec4_index].x;
            } else if (component_index == 1u) {
                note_amplitude = u_audio_synth.note_amplitudes[vec4_index].y;
            } else if (component_index == 2u) {
                note_amplitude = u_audio_synth.note_amplitudes[vec4_index].z;
            } else {
                note_amplitude = u_audio_synth.note_amplitudes[vec4_index].w;
            }
            
            if note_distance < 0.02 && note_amplitude > 0.0 {
                let indicator_strength = (1.0 - note_distance / 0.02) * note_amplitude;
                color += vec3<f32>(0.2 * indicator_strength, 0.1 * indicator_strength, 0.4 * indicator_strength);
            }
        }
    }
    
    textureStore(output_texture, coord, vec4<f32>(color, 1.0));
}