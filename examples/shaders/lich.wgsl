// Lichtenberg noise math inspired by: Lichtenberg figure by rory618 2018, https://www.shadertoy.com/view/3sl3WH

// Group 0: Time uniform
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Group 1: Primary I/O & Parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: LichParams;

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
    glow_intensity: f32,
    specular_strength: f32,
    contrast: f32,
    gamma: f32,
    saturation: f32,
    color_shift: f32,
    spectrum_mix: f32,
    light_intensity: f32,
    _pad: f32,
};

// --- CIE 1931 COLOR SPECTRUM CHROMATICITY LOOKUP MAP ---
const spectrum = array<vec3<f32>, 45>(
    vec3<f32>(0.002362, 0.000253, 0.010482), vec3<f32>(0.019110, 0.002004, 0.086011),
    vec3<f32>(0.084736, 0.008756, 0.389366), vec3<f32>(0.204492, 0.021391, 0.972542),
    vec3<f32>(0.314679, 0.038676, 1.553480), vec3<f32>(0.383734, 0.062077, 1.967280),
    vec3<f32>(0.370702, 0.089456, 1.994800), vec3<f32>(0.302273, 0.128201, 1.745370),
    vec3<f32>(0.195618, 0.185190, 1.317560), vec3<f32>(0.080507, 0.253589, 0.772125),
    vec3<f32>(0.016172, 0.339133, 0.415254), vec3<f32>(0.003816, 0.460777, 0.218502),
    vec3<f32>(0.037465, 0.606741, 0.112044), vec3<f32>(0.117749, 0.761757, 0.060709),
    vec3<f32>(0.236491, 0.875211, 0.030451), vec3<f32>(0.376772, 0.961988, 0.013676),
    vec3<f32>(0.529826, 0.991761, 0.003988), vec3<f32>(0.705224, 0.997340, 0.000000),
    vec3<f32>(0.878655, 0.955552, 0.000000), vec3<f32>(1.014160, 0.868934, 0.000000),
    vec3<f32>(1.118520, 0.777405, 0.000000), vec3<f32>(1.123990, 0.658341, 0.000000),
    vec3<f32>(1.030480, 0.527963, 0.000000), vec3<f32>(0.856297, 0.398057, 0.000000),
    vec3<f32>(0.647467, 0.283493, 0.000000), vec3<f32>(0.431567, 0.179828, 0.000000),
    vec3<f32>(0.268329, 0.107633, 0.000000), vec3<f32>(0.152568, 0.060281, 0.000000),
    vec3<f32>(0.081261, 0.031800, 0.000000), vec3<f32>(0.040851, 0.015905, 0.000000),
    vec3<f32>(0.019941, 0.007749, 0.000000), vec3<f32>(0.009577, 0.003718, 0.000000),
    vec3<f32>(0.004553, 0.001768, 0.000000), vec3<f32>(0.002175, 0.000846, 0.000000),
    vec3<f32>(0.001045, 0.000407, 0.000000), vec3<f32>(0.000508, 0.000199, 0.000000),
    vec3<f32>(0.000251, 0.000098, 0.000000), vec3<f32>(0.000126, 0.000050, 0.000000),
    vec3<f32>(0.000065, 0.000025, 0.000000), vec3<f32>(0.000033, 0.000013, 0.000000),
    vec3<f32>(0.000018, 0.000007, 0.000000), vec3<f32>(0.000009, 0.000004, 0.000000),
    vec3<f32>(0.000005, 0.000002, 0.000000), vec3<f32>(0.000003, 0.000001, 0.000000),
    vec3<f32>(0.000002, 0.000001, 0.000000)
);

const xyz_to_rgb = mat3x3<f32>(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252
);

fn wl_to_xyz(wl: f32) -> vec3<f32> {
    let x = (wl - 390.0) * 0.1;
    let index = u32(clamp(x, 0.0, 43.0));
    return mix(spectrum[index], spectrum[index + 1u], fract(x));
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
    let wl = 390.0 + 240.0 * wave;
    let col_xyz = wl_to_xyz(wl);
    let spectral = max(vec3<f32>(0.0), xyz_to_rgb * col_xyz);
    return mix(base_color, spectral, spectrum_mix);
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Pass 1: Lightning generation pass
@compute @workgroup_size(16, 16, 1)
fn lightning(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pixel_pos = vec2<i32>(id.xy);
    let dimensions = vec2<f32>(dims);
    let uv = (vec2<f32>(id.xy) * 2.0 - dimensions.xy) / dimensions.y;
    var ds = 1e4;
    
    for(var q = 0; q < 1; q = q + 1) {
        let anim_frame = i32(time_data.time * 20.0);
        let seed_val = i32(params.cloud_density);
        var seed = seed_val;
        
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
    
    textureStore(output, pixel_pos, vec4<f32>(current, 1.0));
}

// Pass 2: Feedback accumulation pass (temporal preservation)
@compute @workgroup_size(16, 16, 1)
fn feedback(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pixel_pos = vec2<i32>(id.xy);
    let current_lightning = textureLoad(input_texture0, pixel_pos, 0);
    let previous_frame = textureLoad(input_texture1, pixel_pos, 0);
    
    let result = current_lightning + previous_frame * params.feedback_decay;
    textureStore(output, pixel_pos, result);
}

// Pass 3: Normal Mapping, Dual lighting, and Specular Compositing
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pixel_pos = vec2<i32>(id.xy);
    let uv = vec2<f32>(id.xy) / vec2<f32>(dims);
    let px = 1.0 / vec2<f32>(dims);

    let base_data = textureLoad(input_texture0, pixel_pos, 0);
    let field_lum = dot(base_data.rgb, vec3<f32>(0.299, 0.587, 0.114));

    // physical normals from the density gradient fields
    let hN = dot(textureSampleLevel(input_texture0, input_sampler0, clamp(uv + vec2<f32>(0.0, px.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(1.0));
    let hS = dot(textureSampleLevel(input_texture0, input_sampler0, clamp(uv - vec2<f32>(0.0, px.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(1.0));
    let hE = dot(textureSampleLevel(input_texture0, input_sampler0, clamp(uv + vec2<f32>(px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(1.0));
    let hW = dot(textureSampleLevel(input_texture0, input_sampler0, clamp(uv - vec2<f32>(px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(1.0));
    let fine = vec2<f32>(hE - hW, hN - hS);

    let hN3 = dot(textureSampleLevel(input_texture0, input_sampler0, clamp(uv + vec2<f32>(0.0, px.y * 3.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(1.0));
    let hS3 = dot(textureSampleLevel(input_texture0, input_sampler0, clamp(uv - vec2<f32>(0.0, px.y * 3.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(1.0));
    let hE3 = dot(textureSampleLevel(input_texture0, input_sampler0, clamp(uv + vec2<f32>(px.x * 3.0, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(1.0));
    let hW3 = dot(textureSampleLevel(input_texture0, input_sampler0, clamp(uv - vec2<f32>(px.x * 3.0, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(1.0));
    let coarse = vec2<f32>(hE3 - hW3, hN3 - hS3) / 3.0;

    let gradient = mix(coarse, fine, smoothstep(0.0, 0.04, length(fine))) * 6.0;
    let normal_z = mix(0.15, 0.5, smoothstep(0.0, 0.1, length(gradient)));
    let normal = normalize(vec3<f32>(-gradient.x, -gradient.y, normal_z));

    // dual lighting environment 
    let key_light = normalize(vec3<f32>(1.2, -1.5, 1.8));
    let fill_light = normalize(vec3<f32>(-1.8, 1.2, 1.0));
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);

    let diffuse_key = max(dot(normal, key_light), 0.0);
    let diffuse_fill = max(dot(normal, fill_light), 0.0);
    let diffuse_shading = 0.2 + (diffuse_key * 0.65) + (diffuse_fill * 0.15);

    // GGX Specular for electric plasma channel reflections
    let half_vec = normalize(key_light + view_dir);
    let NdotH = max(dot(normal, half_vec), 0.0);
    let roughness = mix(0.45, 0.12, smoothstep(0.0, 1.5, field_lum));
    let alpha2 = roughness * roughness;
    let spec_denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    let specular_shimmer = (alpha2 / (3.14159265 * spec_denom * spec_denom + 1e-6)) * 0.12 * params.specular_strength * diffuse_key;

    var colored_conduit = base_data.rgb * diffuse_shading * params.light_intensity;
    colored_conduit += vec3<f32>(specular_shimmer);

    colored_conduit += base_data.rgb * field_lum * params.glow_intensity * 0.6;

    var final_color = aces_tonemap(colored_conduit);

    let gray = dot(final_color, vec3<f32>(0.2126, 0.7152, 0.0722));
    final_color = mix(vec3<f32>(gray), final_color, params.saturation);

    final_color = mix(final_color, smoothstep(vec3<f32>(0.0), vec3<f32>(1.0), final_color), params.contrast * 0.15);

    final_color = pow(max(final_color, vec3<f32>(0.0)), vec3<f32>(1.0 / max(params.gamma, 0.1)));

    let vignette = uv * (1.0 - uv);
    final_color *= pow(vignette.x * vignette.y * 16.0, 0.12);

    textureStore(output, pixel_pos, vec4<f32>(final_color, 1.0));
}