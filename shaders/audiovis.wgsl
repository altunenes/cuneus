
//This is an example shader that uses audio data to create a visualizer effect.
//Currently still TODO.
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(1) @binding(0) var<uniform> u_time: TimeUniform;
@group(2) @binding(0) var<uniform> params: Params;
@group(3) @binding(0) var<uniform> u_resolution: ResolutionUniform;

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
    audio_data: array<vec4<f32>, 8>, 
};

struct TimeUniform {
    time: f32,
};

struct Params { 
    red_power: f32,
    green_power: f32,
    blue_power: f32,
    green_boost: f32,
    contrast: f32, 
    gamma: f32,
    glow:f32,

}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let resolution = u_resolution.dimensions;
    // Normalized coordinates
    let uv = tex_coords;
    // vis colors
    let backgroundColor = vec3<f32>(0.05, 0.05, 0.1);
    // Start with background color
    var finalColor = backgroundColor;
    // a subtle grid pattern
    if (fract(uv.x * 20.0) < 0.05 || fract(uv.y * 20.0) < 0.05) {
        finalColor = mix(finalColor, vec3<f32>(0.2, 0.2, 0.25), 0.2);
    }
    //  total audio energy for effects
    var totalEnergy = 0.0;
    // We need to access each value in the vec4 array
    for (var i = 0; i < 8; i++) {
        totalEnergy += (u_resolution.audio_data[i].x + 
                        u_resolution.audio_data[i].y + 
                        u_resolution.audio_data[i].z + 
                        u_resolution.audio_data[i].w) / 32.0;
    }
    // a background pulse based on total energy
    finalColor *= 1.0 + totalEnergy * 0.5;
    // ==================
    // Draw multi-band equalizer
    // ==================
    let eqBottom = 0.15;
    let eqHeight = 0.7;
    let bandWidth = 1.0 / 34.0;  // 32 bands + small margins
    let bandSpacing = bandWidth * 0.1;
    for (var i = 0; i < 32; i++) {
        // Calculate position for this frequency band
        let bandX = (f32(i) + 1.0) * bandWidth;
        // Get the band energy from our packed vec4 array
        let vecIndex = i / 4;
        let vecComponent = i % 4;
        var bandEnergy = 0.0;
        if (vecComponent == 0) {
            bandEnergy = u_resolution.audio_data[vecIndex].x;
        } else if (vecComponent == 1) {
            bandEnergy = u_resolution.audio_data[vecIndex].y;
        } else if (vecComponent == 2) {
            bandEnergy = u_resolution.audio_data[vecIndex].z;
        } else {
            bandEnergy = u_resolution.audio_data[vecIndex].w;
        }
        
        let barHeight = bandEnergy * eqHeight;
        // Draw frequency band bar if we're within its boundaries
        if (uv.x >= bandX && uv.x < bandX + bandWidth - bandSpacing) {
            if (uv.y >= eqBottom && uv.y < eqBottom + barHeight) {
                // Color gradient based on frequency and height
                let freqT = f32(i) / 32.0;  // 0-1 range for frequency
                let heightT = (uv.y - eqBottom) / max(barHeight, 0.001);  // 0-1 range for height within bar
                // Create a color spectrum from red (low) to blue (high) frequencies
                let baseColor = mix(
                    mix(
                        vec3<f32>(0.9, 0.1, 0.1),  // Low/bass (red)
                        vec3<f32>(0.9, 0.6, 0.1),  // Low-mid (orange)
                        min(freqT * 3.0, 1.0)
                    ),
                    mix(
                        vec3<f32>(0.1, 0.8, 0.2),  // Mid (green)
                        vec3<f32>(0.1, 0.2, 0.9),  // High (blue)
                        (freqT - 0.33) * 1.5
                    ),
                    min(max((freqT - 0.1) * 1.5, 0.0), 1.0)
                );
                
                // Add brightness gradient based on height
                let color = mix(
                    baseColor * 0.7,             // Darker at bottom
                    baseColor * 1.3,             // Brighter at top
                    heightT
                );
                
                // Add animated stripes based on frequency
                let stripeSpeed = 1.0 + freqT * 3.0;
                let stripeWidth = 0.1 + bandEnergy * 0.2;
                if (fract(uv.y * 15.0 - u_time.time * stripeSpeed) < stripeWidth) {
                    finalColor = color * 1.3;
                } else {
                    finalColor = color;
                }
                
                // Add highlight at top of bar
                if (uv.y > eqBottom + barHeight - 0.01) {
                    finalColor = mix(finalColor, vec3<f32>(1.0), 0.5);
                }
            }
        }
    }
    
    // ==================
    // Draw baseline
    // ==================
    if (abs(uv.y - eqBottom) < 0.002) {
        finalColor = vec3<f32>(0.4, 0.4, 0.5);
    }
    
    // ==================
    // Draw spectrum analyzer at bottom
    // ==================
    if (uv.y < 0.08 && uv.y > 0.02) {
        let specY = 0.05;
        let specHeight = 0.03;
        
        for (var i = 0; i < 32; i++) {
            let bandX = (f32(i) + 1.0) * bandWidth;
            let freqT = f32(i) / 32.0;
            
            // Get band energy from our packed array
            let vecIndex = i / 4;
            let vecComponent = i % 4;
            
            var bandEnergy = 0.0;
            if (vecComponent == 0) {
                bandEnergy = u_resolution.audio_data[vecIndex].x;
            } else if (vecComponent == 1) {
                bandEnergy = u_resolution.audio_data[vecIndex].y;
            } else if (vecComponent == 2) {
                bandEnergy = u_resolution.audio_data[vecIndex].z;
            } else {
                bandEnergy = u_resolution.audio_data[vecIndex].w;
            }
            
            // Calculate peak height
            let peakHeight = bandEnergy * specHeight;
            
            // Draw peak line
            if (uv.x >= bandX && uv.x < bandX + bandWidth - bandSpacing) {
                // Draw frequency band
                if (uv.y >= specY - peakHeight && uv.y <= specY) {
                    // Color based on frequency
                    let color = mix(
                        vec3<f32>(1.0, 0.2, 0.1),  // Low/bass (red)
                        mix(
                            vec3<f32>(1.0, 0.8, 0.1),  // Mid (yellow)
                            vec3<f32>(0.2, 0.4, 1.0),  // High (blue)
                            min(max((freqT - 0.3) * 1.5, 0.0), 1.0)
                        ),
                        min(freqT * 2.0, 1.0)
                    );
                    
                    finalColor = color;
                }
            }
        }
        
        // Draw baseline for spectrum
        if (abs(uv.y - specY) < 0.001) {
            finalColor = vec3<f32>(0.3, 0.3, 0.3);
        }
    }
    
    // Get alpha from original texture
    let alpha = textureSample(tex, tex_sampler, uv).a;
    
    return vec4<f32>(finalColor, alpha);
}