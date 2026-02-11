// Kuwahara Filter, Enes Altun, 2025, MIT License
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: KuwaharaParams;

@group(2) @binding(0) var channel0: texture_2d<f32>;
@group(2) @binding(1) var channel0_sampler: sampler;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;
@group(3) @binding(4) var input_texture2: texture_2d<f32>;
@group(3) @binding(5) var input_sampler2: sampler;

struct KuwaharaParams {
    radius: f32,
    q: f32,
    alpha: f32,
    filter_strength: f32,
    sigma_d: f32,
    sigma_r: f32,
    edge_threshold: f32,
    color_enhance: f32,
    blur_samples: f32,
    blur_lod: f32,
    blur_slod: f32,
    filter_mode: i32,
    show_tensors: i32,
    lic_length: f32,
    lic_strength: f32,
    lic_width: f32,
}

const PI: f32 = 3.14159265359;


fn gaussian_weight(i: vec2f, sigma: f32) -> f32 {
    let si = i / sigma;
    return exp(-0.5 * dot(si, si)) / (6.28 * sigma * sigma);
}

fn blur_tensor(uv: vec2f, ts: vec2f) -> vec3f {
    var result = vec3f(0.0);
    var tw = 0.0;
    let samples = i32(params.blur_samples);
    let slod = i32(params.blur_slod);
    let s = samples / slod;
    let sig = params.sigma_r * 2.5;
    let lod = params.blur_lod;
    
    for (var i = 0; i < s * s; i++) {
        let d = vec2f(f32(i % s), f32(i / s)) * f32(slod) - f32(samples) / 2.0;
        let w = gaussian_weight(d, sig);
        let suv = clamp(uv + ts * d, vec2f(0.0), vec2f(1.0));
        let td = textureSampleLevel(input_texture0, input_sampler0, suv, lod);
        
        result += td.xyz * w;
        tw += w;
    }
    
    return result / tw;
}

fn calc_region_stats(uv: vec2f, lower: vec2i, upper: vec2i, ts: vec2f) -> vec2f {
    var csum = vec3f(0.0);
    var cvar = vec3f(0.0);
    var cnt = 0;
    
    for (var j = lower.y; j <= upper.y; j++) {
        for (var i = lower.x; i <= upper.x; i++) {
            let off = vec2f(f32(i), f32(j)) * ts;
            let suv = clamp(uv + off, vec2f(0.0), vec2f(1.0));
            let sc = get_input_color(suv);
            
            csum += sc;
            cvar += sc * sc;
            cnt++;
        }
    }
    
    if (cnt > 0) {
        let mc = csum / f32(cnt);
        let rv = cvar / f32(cnt) - (mc * mc);
        let tv = rv.r + rv.g + rv.b;
        let lum = dot(mc, vec3f(0.299, 0.587, 0.114));
        let cv = tv * 0.7 + dot(rv, vec3f(0.299, 0.587, 0.114)) * 0.3;
        
        return vec2f(lum, cv);
    }
    return vec2f(0.0, 999999.0);
}

// ACES tone mapping for anisotropic areas
fn aces_aniso(color: vec3f, strength: f32) -> vec3f {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    let aces = (color * (a * color + b)) / (color * (c * color + d) + e);
    return mix(color, clamp(aces, vec3f(0.0), vec3f(1.0)), strength * 0.3);
}

fn get_input_color(uv: vec2f) -> vec3f {
    let dims = textureDimensions(channel0);
    if (dims.x > 1 && dims.y > 1) {
        return textureSampleLevel(channel0, channel0_sampler, uv, 0.0).rgb;
    }
    let center = vec2f(0.5);
    let dist = distance(uv, center);
    let circle = smoothstep(0.2, 0.21, dist);
    return mix(vec3f(0.8, 0.4, 0.2), vec3f(0.1, 0.1, 0.2), circle);
}

// structure tensor pass
// Sobel approach inspired by sofiene71: https://www.shadertoy.com/view/td3BzX
@compute @workgroup_size(16, 16, 1)
fn structure_tensor(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let ts = 1.0 / vec2f(dims);
    let d = ts * params.sigma_d;
    
    // sobel kernels
    let sx = (
        -1.0 * get_input_color(clamp(uv + vec2f(-d.x, -d.y), vec2f(0.0), vec2f(1.0))) +
        -2.0 * get_input_color(clamp(uv + vec2f(-d.x,  0.0), vec2f(0.0), vec2f(1.0))) + 
        -1.0 * get_input_color(clamp(uv + vec2f(-d.x,  d.y), vec2f(0.0), vec2f(1.0))) +
        1.0 * get_input_color(clamp(uv + vec2f( d.x, -d.y), vec2f(0.0), vec2f(1.0))) +
        2.0 * get_input_color(clamp(uv + vec2f( d.x,  0.0), vec2f(0.0), vec2f(1.0))) + 
        1.0 * get_input_color(clamp(uv + vec2f( d.x,  d.y), vec2f(0.0), vec2f(1.0)))
    ) / (4.0);

    let sy = (
        -1.0 * get_input_color(clamp(uv + vec2f(-d.x, -d.y), vec2f(0.0), vec2f(1.0))) + 
        -2.0 * get_input_color(clamp(uv + vec2f( 0.0, -d.y), vec2f(0.0), vec2f(1.0))) + 
        -1.0 * get_input_color(clamp(uv + vec2f( d.x, -d.y), vec2f(0.0), vec2f(1.0))) +
        1.0 * get_input_color(clamp(uv + vec2f(-d.x,  d.y), vec2f(0.0), vec2f(1.0))) +
        2.0 * get_input_color(clamp(uv + vec2f( 0.0,  d.y), vec2f(0.0), vec2f(1.0))) + 
        1.0 * get_input_color(clamp(uv + vec2f( d.x,  d.y), vec2f(0.0), vec2f(1.0)))
    ) / 4.0;
    
    // rgb gradients
    let gr = length(vec2f(sx.r, sy.r));
    let gg = length(vec2f(sx.g, sy.g));
    let gb = length(vec2f(sx.b, sy.b));
    
    let cw = vec3f(0.299, 0.587, 0.114);
    let wg = gr * cw.r + gg * cw.g + gb * cw.b;
    
    let gx = dot(sx, cw) + (gr + gg + gb) * 0.1;
    let gy = dot(sy, cw) + (gr + gg + gb) * 0.1;
    
    // tensor components
    let Jxx = gx * gx + wg * 0.05;
    let Jyy = gy * gy + wg * 0.05;
    let Jxy = gx * gy;
    
    textureStore(output, id.xy, vec4f(Jxx, Jyy, Jxy, wg));
}

// tensor field pass
@compute @workgroup_size(16, 16, 1)
fn tensor_field(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let ts = 1.0 / vec2f(dims);

    let st = blur_tensor(uv, ts);
    
    let Jxx = st.x;
    let Jyy = st.y;
    let Jxy = st.z;

    let trace = Jxx + Jyy;
    let det = Jxx * Jyy - Jxy * Jxy;
    let disc = trace * trace - 4.0 * det;
    let sqrt_disc = sqrt(max(disc, 0.0));
    let l1 = 0.5 * (trace + sqrt_disc);
    let l2 = 0.5 * (trace - sqrt_disc);

    // eigenvector
    var v = vec2f(l1 - Jxx, -Jxy);
    var ori: vec2f;
    if (length(v) > 0.0) { 
        ori = normalize(v);
    } else {
        ori = vec2f(0.0, 1.0);
    }

    let phi = atan2(ori.y, ori.x);
    
    // anisotropy
    var anis = 0.0;
    if (l1 + l2 > 0.0) {
        anis = (l1 - l2) / (l1 + l2);
    }
    
    textureStore(output, id.xy, vec4f(ori.x, ori.y, phi, anis));
}

// kuwahara filter pass
@compute @workgroup_size(16, 16, 1) 
fn kuwahara_filter(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let ts = 1.0 / vec2f(dims);
    
    let orig = get_input_color(uv);
    var result = vec4f(orig, 1.0);
    
    if (params.filter_mode == 0) {
        // classic mode with adaptive radius
        let r = i32(min(params.radius, 8.0));
        
        var qmean: array<vec4f, 4>;
        var qvar: array<f32, 4>;
        
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                let off = vec2f(f32(dx), f32(dy)) * ts;
                let suv = clamp(uv + off, vec2f(0.0), vec2f(1.0));
                let sc = get_input_color(suv);
                
                var q = 0;
                if (dx >= 0 && dy < 0) { q = 1; }   
                else if (dx < 0 && dy >= 0) { q = 2; }   
                else if (dx >= 0 && dy >= 0) { q = 3; }   
                
                qmean[q] += vec4f(sc, 1.0);
                let ri = length(sc);
                qvar[q] += ri * ri;
            }
        }
        
        var minvar = 999999.0;
        var selq = 0;
        
        for (var q = 0; q < 4; q++) {
            if (qmean[q].w > 0.0) {
                let mc = qmean[q].rgb / qmean[q].w;
                let mi = length(mc);
                let variance = (qvar[q] / qmean[q].w) - (mi * mi);
                let avar = variance / (params.q * params.q);
                
                if (avar < minvar) {
                    minvar = avar;
                    selq = q;
                }
            }
        }
        
        if (qmean[selq].w > 0.0) {
            let sc = qmean[selq].rgb / qmean[selq].w;

            let soft_strength = min(params.filter_strength, 2.0) / 2.0;
            result = vec4f(mix(orig, sc, soft_strength), 1.0);
        }
    } else {
        // anisotropic mode
        let td = textureSampleLevel(input_texture1, input_sampler1, uv, 0.0);
        let ori = td.xy;
        let anis = td.w;
        
        let alpha = params.alpha;
        let radius = params.radius;
        
        let eff_anis = select(0.0, anis, anis > params.edge_threshold);
        
        let a = radius * (1.0 + eff_anis * alpha * 0.8);
        let b = radius * max(0.3, 1.0 - eff_anis * alpha * 0.6);
        
        var qmeans: array<vec3f, 4>;
        var qvars: array<f32, 4>;
        var qcnts: array<f32, 4>;
        
        for (var k = 0; k < 4; k++) {
            qmeans[k] = vec3f(0.0);
            qvars[k] = 0.0;
            qcnts[k] = 0.0;
        }
        
        let maxr = i32(min(radius + 1.0, 8.0));
        for (var j = -maxr; j <= maxr; j++) {
            for (var i = -maxr; i <= maxr; i++) {
                let off = vec2f(f32(i), f32(j));
                
                let ex = off.x * ori.x + off.y * ori.y;
                let ey = -off.x * ori.y + off.y * ori.x;
                let ed = (ex * ex) / (a * a) + (ey * ey) / (b * b);
                
                if (ed <= 1.0) {
                    let suv = clamp(uv + off * ts, vec2f(0.0), vec2f(1.0));
                    let sc = get_input_color(suv);
                    let ri = length(sc);
                    
                    if (i <= 0 && j <= 0) { 
                        qmeans[0] += sc;
                        qvars[0] += ri * ri;
                        qcnts[0] += 1.0;
                    }
                    if (i >= 0 && j <= 0) {  
                        qmeans[1] += sc;
                        qvars[1] += ri * ri;
                        qcnts[1] += 1.0;
                    }
                    if (i <= 0 && j >= 0) { 
                        qmeans[2] += sc;
                        qvars[2] += ri * ri;
                        qcnts[2] += 1.0;
                    }
                    if (i >= 0 && j >= 0) { 
                        qmeans[3] += sc;
                        qvars[3] += ri * ri;  
                        qcnts[3] += 1.0;
                    }
                }
            }
        }
        
        var minvar = 999999.0;
        var best = orig;
        
        for (var q = 0; q < 4; q++) {
            if (qcnts[q] > 0.0) {
                let mc = qmeans[q] / qcnts[q];
                let mi = length(mc);
                let variance = (qvars[q] / qcnts[q]) - (mi * mi);
                let avar = variance / (params.q * params.q);
                
                if (avar < minvar) {
                    minvar = avar;
                    best = mc;
                }
            }
        }
        
        let soft_strength = min(params.filter_strength, 2.0) / 2.0;
        result = vec4f(mix(orig, best, soft_strength), 1.0);
    }
    

    var fc = result.rgb;
    
    if (abs(params.color_enhance - 1.0) > 0.01) {
        let enh = params.color_enhance;
        
        // JUST FOR FEELING
        if (params.filter_mode == 1) {
            let td = textureSampleLevel(input_texture1, input_sampler1, uv, 0.0);
            let anis = td.w;
            fc = aces_aniso(fc, anis);
        }
        
        let lum = dot(fc, vec3f(0.299, 0.587, 0.114));
        let sat_factor = mix(1.0, enh * 1.2, 0.5);
        fc = mix(vec3f(lum), fc, sat_factor);
        
        fc = clamp(fc, vec3f(0.0), vec3f(1.0));
    }
    
    result = vec4f(fc, result.a);
    
    textureStore(output, id.xy, result);
}

// hash for subtle brush noise
fn hash_lic(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Line Integral Convolution along edges guided by the smoothed structure tensor.
// input_texture0 = tensor_field (ori.x, ori.y, phi, anisotropy)
// input_texture1 = kuwahara_filter output (filtered color)
// Traces curved streamlines through the tensor field to create brush strokes.
@compute @workgroup_size(16, 16, 1)
fn lic_edges(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let ts = 1.0 / vec2f(dims);

    let kuwahara_color = textureSampleLevel(input_texture1, input_sampler1, uv, 0.0).rgb;

    if (params.lic_strength <= 0.001) {
        textureStore(output, id.xy, vec4f(kuwahara_color, 1.0));
        return;
    }

    // Read smoothed tensor field: orientation and anisotropy
    let tensor = textureSampleLevel(input_texture0, input_sampler0, uv, 0.0);
    let ori = tensor.xy;
    let anis = tensor.w;

    // Blend factor: stronger LIC on edges (high anisotropy), preserve flat regions
    let edge_blend = smoothstep(params.edge_threshold * 0.3, params.edge_threshold + 0.3, anis);
    if (edge_blend < 0.01) {
        textureStore(output, id.xy, vec4f(kuwahara_color, 1.0));
        return;
    }

    let steps = i32(params.lic_length);
    let step_size = params.lic_width;
    let sigma = f32(steps) * 0.4;

    // Center sample
    var acc = kuwahara_color;
    var weight_sum = 1.0;

    // Per-pixel noise for natural brush variation
    let noise_val = (hash_lic(vec2f(id.xy) * 0.37) - 0.5) * 0.4;

    // Forward trace: follow edge tangent through the tensor field
    var pos_uv = uv;
    for (var i = 1; i <= steps; i++) {
        // Read tensor at current streamline position
        let local_tensor = textureSampleLevel(input_texture0, input_sampler0, pos_uv, 0.0);
        let local_ori = local_tensor.xy;
        // Tangent = perpendicular to gradient direction (along the edge)
        let tangent = vec2f(-local_ori.y, local_ori.x);

        pos_uv += tangent * ts * step_size + vec2f(noise_val) * ts * 0.3;
        pos_uv = clamp(pos_uv, vec2f(0.0), vec2f(1.0));

        let sample_color = textureSampleLevel(input_texture1, input_sampler1, pos_uv, 0.0).rgb;

        let fi = f32(i);
        let w = exp(-0.5 * fi * fi / (sigma * sigma));
        acc += sample_color * w;
        weight_sum += w;
    }

    // Backward trace (opposite direction along tangent)
    pos_uv = uv;
    for (var i = 1; i <= steps; i++) {
        let local_tensor = textureSampleLevel(input_texture0, input_sampler0, pos_uv, 0.0);
        let local_ori = local_tensor.xy;
        let tangent = vec2f(-local_ori.y, local_ori.x);

        pos_uv -= tangent * ts * step_size + vec2f(noise_val) * ts * 0.3;
        pos_uv = clamp(pos_uv, vec2f(0.0), vec2f(1.0));

        let sample_color = textureSampleLevel(input_texture1, input_sampler1, pos_uv, 0.0).rgb;

        let fi = f32(i);
        let w = exp(-0.5 * fi * fi / (sigma * sigma));
        acc += sample_color * w;
        weight_sum += w;
    }

    let lic_color = acc / weight_sum;

    let blend = edge_blend * params.lic_strength;
    let result = mix(kuwahara_color, lic_color, blend);

    textureStore(output, id.xy, vec4f(result, 1.0));
}

// main image pass
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let result = textureSampleLevel(input_texture2, input_sampler2, uv, 0.0);

    textureStore(output, id.xy, result);
}