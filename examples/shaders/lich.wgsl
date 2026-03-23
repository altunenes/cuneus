// Inspired by https://www.shadertoy.com/view/3sl3WH
// Function "wavelength_to_rgb" based on Michael Friendly (2015)

// Group 0: Time uniform
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Group 1: Primary I/O & Parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: LichParams;

// Group 2: Engine Resources (unused for Lich)

// Group 3: Multi-pass Input Textures
@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;

struct LichParams {
    cloud_density: f32,
    lightning_intensity: f32,
    branch_count: f32,
    feedback_decay: f32,
    base_color: vec3<f32>,
    _pad1: f32,
    color_shift: f32,
    spectrum_mix: f32,
    _pad2: vec2<f32>,
};

const ATOMIC_SCALE: f32 = 2048.0;

fn wavelength_to_rgb(wave: f32) -> vec3<f32> {
    let x = (wave - 380.0) / 10.0;
    let factor = select(0.0, 
                       select(1.0 - (x - 750.0) / 50.0,
                             select(1.0, 1.0 - (380.0 - x) / 50.0, x >= 380.0),
                             x <= 750.0),
                       x <= 800.0);
    
    var r = select(0.0,
                  select(1.0,
                        select((x - 440.0) / (510.0 - 440.0),
                              select(1.0,
                                    (750.0 - x) / (750.0 - 610.0),
                                    x >= 610.0),
                              x >= 510.0),
                        x >= 440.0),
                  x >= 380.0);
    
    var g = select(0.0,
                  select((x - 440.0) / (490.0 - 440.0),
                        select(1.0,
                              (580.0 - x) / (580.0 - 510.0),
                              x >= 510.0),
                        x >= 490.0),
                  x >= 440.0);
    
    var b = select(0.0,
                  select(1.0,
                        (490.0 - x) / (490.0 - 440.0),
                        x >= 440.0),
                  x >= 380.0);
    
    return vec3<f32>(r, g, b) * factor;
}

fn IHash(a: i32) -> i32 {
    var x = a;
    x = (x ^ 61) ^ (x >> 16);
    x = x + (x << 3);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb; 
    x = x ^ (x >> 15);
    return x;
}

fn Hash(a: i32) -> f32 {
    return f32(IHash(a)) / f32(0x7FFFFFFF);
}

fn rand4(seed: i32) -> vec4<f32> {
    return vec4<f32>(
        Hash(seed ^ 348593),
        Hash(seed ^ 859375),
        Hash(seed ^ 625384),
        Hash(seed ^ 253625)
    );
}

fn rand2(seed: i32) -> vec2<f32> {
    return vec2<f32>(
        Hash(seed ^ 348593),
        Hash(seed ^ 859375)
    );
}

fn randn(randuniform: vec2<f32>) -> vec2<f32> {
    var r = randuniform;
    r.x = sqrt(-2.0 * log(1e-9 + abs(r.x)));
    r.y = r.y * 6.28318;
    return r.x * vec2<f32>(cos(r.y), sin(r.y));
}

fn lineDist(a: vec2<f32>, b: vec2<f32>, uv: vec2<f32>) -> f32 {
    return length(uv-(a+normalize(b-a)*min(length(b-a),max(0.0,dot(normalize(b-a),(uv-a))))));
}

fn process_color(base_color: vec3<f32>, wave: f32, spectrum_mix: f32) -> vec3<f32> {
    let spectral = wavelength_to_rgb(wave * 380.0 + 400.0);
    let mixed = mix(base_color, spectral, spectrum_mix);
    
    let temp_adjust = vec3<f32>(
        1.0 + 0.0 * 0.2, 
        1.0,                            
        1.0 - 0.0 * 0.1  
    );
    
    return mixed * temp_adjust;
}

// Lightning generation pass
@compute @workgroup_size(16, 16, 1)
fn lightning(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pixel_pos = vec2<i32>(id.xy);
    let dimensions = vec2<f32>(dims);
    let FragCoord = vec2<f32>(id.xy);
    let uv = (FragCoord * 2.0 - dimensions.xy) / dimensions.y;
    var ds = 1e4;
    
    for(var q = 0; q < 1; q = q + 1) {
        let anim_frame = i32(time_data.time * 20.0);
        let f = anim_frame + 123457 * (q + 1); 
        var seed = i32(params.cloud_density);
        
        var a = vec2<f32>(0.0, 1.0);
        var b = vec2<f32>(0.2, 0.7) + 0.4 * randn(rand2(seed ^ 859375)) / 8.0;
        
        let branch_factor = 30.0 * params.branch_count;
        for(var k = 0; k < i32(branch_factor); k = k + 1) {
            let l = length(b - a);
            
            let c = (a + b) / 2.0 + l * randn(rand2(seed ^ 859375)) / 8.0;
            let d = b * 1.9 - a * 0.9 + l * randn(rand2(seed ^ 935375)) / 4.0;
            let e = b * 1.9 - a * 0.9 + l * randn(rand2(seed ^ 687643)) / 4.0;
            
            let j = 1.0 + 0.5 * rand4(seed ^ IHash(anim_frame * 574595 ^ q));
            
            let d0 = lineDist(a, c, uv) * j.x;
            let d1 = lineDist(c, b, uv) * j.y;
            let d2 = lineDist(b, d, uv) * j.z;
            let d3 = lineDist(b, e, uv) * j.w;
            
            if(d0 < min(d1, min(d2, d3))) {
                b = c;
                seed = IHash(seed ^ 796489);
            } else if(d1 < min(d2, d3)) {
                a = c;
                seed = IHash(seed ^ 879235);
            } else if(d2 < d3) {
                a = b;
                b = d;
                seed = IHash(seed ^ 574595);
            } else {
                a = b;
                b = e;
                seed = IHash(seed ^ 630658);
            }
        }
        
        ds = min(ds, lineDist(a, b, uv));
    }
    
    let intensity = max(0.0, 1.0 - ds * dimensions.y / params.color_shift) * params.lightning_intensity;
    var current = vec3<f32>(0.0);
    
    if(intensity > 0.001) {
        let wave = Hash(i32(time_data.time * 1000.0));
        current = process_color(params.base_color, wave, params.spectrum_mix) * intensity;
    }
    
    let result = vec4<f32>(current, 1.0);
    textureStore(output, pixel_pos, result);
}

// Feedback accumulation pass (combines lightning + previous feedback with decay)
@compute @workgroup_size(16, 16, 1)
fn feedback(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pixel_pos = vec2<i32>(id.xy);
    let uv = vec2<f32>(id.xy) / vec2<f32>(dims);
    
    let current_lightning = textureLoad(input_texture0, pixel_pos, 0);
    
    let previous_frame = textureLoad(input_texture1, pixel_pos, 0);
    
    let result = current_lightning + previous_frame * params.feedback_decay;
    
    textureStore(output, pixel_pos, result);
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pixel_pos = vec2<i32>(id.xy);
    
    let raw_result = textureLoad(input_texture0, pixel_pos, 0);
    
    let exposed = pow(raw_result.rgb * (1.0 + 0.0), vec3<f32>(1.0/2.2));
    let result = vec4<f32>(exposed, 1.0);
    
    textureStore(output, pixel_pos, result);
}