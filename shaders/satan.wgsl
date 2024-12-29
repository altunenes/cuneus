struct TimeUniform {
    time: f32,
};
//I combined textures (Satan and smoke)
@group(0) @binding(0) var smoke_and_pentagram: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(1) @binding(0) var<uniform> time_data: TimeUniform;
struct ResolutionUniform {
    resolution: vec2<f32>,
};

@group(3) @binding(0) var<uniform> iResolution: ResolutionUniform;

struct Params {
    min_radius: f32,
    max_radius: f32,
    size: f32,
    decay: f32,
    smoke_color: vec3<f32>,
    _padding: f32,
    color2: vec3<f32>,
};
@group(2) @binding(0)
var<uniform> params: Params;
const PI: f32 = 3.14159265358979323846;

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(0.0, k - abs(b - a)) / k;
    return min(a, b) - h * h * h * k / 6.0;
}

fn super_length(p: vec2<f32>) -> f32 {
    return sqrt(length(p * p));
}

fn box_sdf(p: vec2<f32>, d: vec2<f32>) -> f32 {
    let q = sqrt(p * p + 0.005) - d;
    return super_length(max(q, vec2<f32>(0.0))) + min(0.0, max(q.x, q.y));
}

fn linedist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let k = dot(p - a, b - a) / dot(b - a, b - a);
    return distance(p, mix(a, b, clamp(k, 0.0, 1.0)));
}

fn random(st: vec2<f32>) -> f32 {
    return fract(sin(dot(st.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let p2 = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3))
    );
    return -1.0 + 2.0 * fract(sin(p2) * 43758.5453123);
}

fn snoise(p: vec2<f32>) -> f32 {
    let K1 = 0.366025404;
    let K2 = 0.211324865;
    
    let i = floor(p + (p.x + p.y) * K1);
    let a = p - i + (i.x + i.y) * K2;
    let o = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), a.x > a.y);
    let b = a - o + K2;
    let c = a - 1.0 + 2.0 * K2;
    
    let h = max(0.5 - vec3<f32>(
        dot(a, a),
        dot(b, b),
        dot(c, c)
    ), vec3<f32>(0.0));
    
    let n = h * h * h * h * vec3<f32>(
        dot(a, hash2(i + vec2<f32>(0.0))),
        dot(b, hash2(i + o)),
        dot(c, hash2(i + vec2<f32>(1.0)))
    );
    
    return dot(n, vec3<f32>(70.0));
}

fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    
    for(var i: f32 = 0.0; i < 5.0; i += 1.0) {
        value += amplitude * snoise(p * frequency);
        frequency *= params.min_radius;
        amplitude *= 0.5;
    }
    return value;
}

fn flame(p: vec2<f32>, time: f32) -> f32 {
    let f = sin(p.y * 10.0 + time) * sin(p.x * 8.0 + time * 1.5) * 0.015;
    return f + sin(p.y * 15.0 + time * 1.2) * sin(p.x * 12.0 + time) * 0.01;
}

fn spark(p: vec2<f32>, time: f32, seed: f32) -> f32 {
    let t = fract(time * 0.5 + seed);
    let sparkPos = vec2<f32>(
        sin(time * 3.0 + seed * 6.28) * 0.3,
        cos(time * 2.0 + seed * 6.28) * 0.3
    ) + vec2<f32>(cos(seed * 6.28) * 0.5, sin(seed * 6.28) * 0.5);
    
    let spark = length(p - sparkPos);
    let glow = 0.02 / (spark + 0.001);
    return glow * smoothstep(1.0, 0.0, t) * t;
}

fn ember(p: vec2<f32>, time: f32, seed: f32) -> f32 {
    let t = fract(time * 0.2 + seed);
    let emberPos = vec2<f32>(
        sin(time + seed * 10.0) * 0.6,
        fract(time * 0.5 + seed) * 1.2 - 0.6
    );
    let ember = length(p - emberPos);
    let glow = 0.01 / (ember + 0.001);
    return glow * smoothstep(1.0, 0.8, t);
}

fn darkEnergy(p: vec2<f32>, time: f32) -> f32 {
    var e = fbm(p * 2.0 + time * 0.1) * 0.2;
    
    for(var i: f32 = 0.0; i < 3.0; i += 1.0) {
        let pos = p * (1.0 + i * 0.4);
        e += sin(pos.x * 4.0 + time) * sin(pos.y * 4.0 + time * 1.2) * 0.02;
    }
    return e;
}

fn mysticCircle(p: vec2<f32>, time: f32) -> f32 {
    let r = length(p);
    var circle = abs(r - 0.9) - 0.005;
    circle += snoise(p * 5.0 + time * 0.2) * 0.01;
    return circle;
}
//this pentagram shape is from: blackle, 2020: https://www.shadertoy.com/view/WlSczh CC 1.0
fn pin_sdf(p: vec2<f32>, time: f32) -> f32 {
    var modified_p = p + sin(p.y * 20.0) * sin(p.x * 20.0) * 0.004;
    let pulse = sin(time * 2.0) * 0.03;
    modified_p *= 1.0 + pulse;
    
    let angl = atan2(modified_p.y, modified_p.x);
    let angl2 = asin(sin(angl * 5.0 - 1.4) * 0.995) / 5.0;
    let p3 = vec2<f32>(cos(angl2), sin(angl2)) * length(modified_p);
    
    var pentagram = linedist(p3, vec2<f32>(0.67, 0.22), vec2<f32>(0.25, -0.07));
    pentagram = smin(pentagram, linedist(p3, vec2<f32>(0.23, -0.07), vec2<f32>(0.0, 0.60)), 0.08);
    pentagram += flame(p3, time);
    pentagram = min(pentagram, mysticCircle(p, time));
    pentagram += darkEnergy(p, time) * 0.5;
    return pentagram;
}

@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (2.0 * FragCoord.xy - iResolution.resolution.xy) / min(iResolution.resolution.x, iResolution.resolution.y);

    let time = time_data.time;
    
    let rotated_uv = vec2<f32>(
        uv.x * cos(2.14159) - uv.y * sin(2.14159),
        uv.x * sin(2.14159) + uv.y * cos(2.14159)
    );
    
    let d = pin_sdf(rotated_uv, time);
    
    let baseColor = vec3<f32>(0.8, 0.0, 0.0);
    let glowColor = vec3<f32>(1.0, 0.2, 0.0);
    let darkEnergyColor = vec3<f32>(0.4, 0.0, 0.8);
    
    let glow = 0.02 / (abs(d) + 0.01);
    let pulse = (sin(time * 3.0) * 0.5 + 0.5) * 0.3;
    
    var col = mix(baseColor, glowColor, glow * (1.0 + pulse));
    
    let noise = fbm(uv * 3.0 + time * 0.1);
    col = mix(col, darkEnergyColor, darkEnergy(rotated_uv, time) * 2.0 + noise * 0.3);
    col *= glow;
    
    var sparkAccum = 0.0;
    for(var i: f32 = 0.0; i < 8.0; i += 1.0) {
        sparkAccum += spark(rotated_uv, time, random(vec2<f32>(i, 0.0)));
    }
    col += vec3<f32>(1.0, 0.7, 0.3) * sparkAccum;
    
    var emberAccum = 0.0;
    for(var i: f32 = 0.0; i < 12.0; i += 1.0) {
        emberAccum += ember(rotated_uv, time, random(vec2<f32>(i, 1.0)));
    }
    col += params.color2 * emberAccum;
    
    let darkness = length(uv);
    col *= 1.0 - darkness * params.size;
    
    let smoke = fbm(uv * 2.0 - time * 0.05) * 0.15;
    col += mix(vec3<f32>(0.0), darkEnergyColor * 0.3, smoke + snoise(uv * 4.0 + time * 0.1) * 0.05);
    
    return vec4<f32>(col, glow);
}

fn noise_b(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(random(i + vec2<f32>(0.0, 0.0)), random(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(random(i + vec2<f32>(0.0, 1.0)), random(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

fn fbm_b(p: vec2<f32>, time: f32) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var freq = 1.0;
    let rotAngle = time * 0.1;
    let rot = mat2x2<f32>(
        vec2<f32>(cos(rotAngle), -sin(rotAngle)),
        vec2<f32>(sin(rotAngle), cos(rotAngle))
    );
    var modified_p = p;
    
    for(var i = 0; i < 6; i++) {
        sum += amp * noise_b(modified_p * freq);
        amp *= 0.5;
        freq *= 2.0;
        modified_p = rot * modified_p;
    }
    return sum;
}

@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let uv = FragCoord.xy / iResolution.resolution.xy;

    let p = uv * 3.0;
    
    let prevData = textureSample(smoke_and_pentagram, tex_sampler, uv);
    let pentagramGlow = prevData.a;
    
    let timeScale = 1.0 + pentagramGlow * 2.0;
    let smokeOffset = vec2<f32>(
        pentagramGlow * sin(time_data.time * 2.0),
        pentagramGlow * cos(time_data.time * 3.0)
    );
    
    var smoke = fbm_b(p + smokeOffset, time_data.time * timeScale);
    smoke += fbm_b(p * 2.0 + smokeOffset * 2.0, time_data.time * timeScale * 1.5) * 0.5;
    
    let densityVariation = mix(0.4, 1.2, pentagramGlow);
    smoke *= densityVariation;
    
    let blendFactor = params.max_radius;
    smoke = mix(prevData.r, smoke, blendFactor);
    
    var col = prevData.rgb; 
    let smokeColor = params.smoke_color;

    col = mix(col, smokeColor, smoke * 0.1);
    
    return vec4<f32>(col, smoke);
}

@fragment
fn fs_pass3(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let uv = FragCoord.xy / iResolution.resolution.xy;

    
    let data = textureSample(smoke_and_pentagram, tex_sampler, uv);
    var col = data.rgb;
    
    let vignette = params.decay- length(uv - 0.5) * 1.5;
    col *= vignette;
    
    return vec4<f32>(col, 1.0);
}