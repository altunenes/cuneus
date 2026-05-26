// MIT License Enes Altun, 2025
struct TimeUniform {
    time: f32,
};

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;

struct Params {
    background_color: f32,
    hue_color: vec4<f32>,
    _pad1: f32,
    light_intensity: f32,
    rim_power: f32,
    ao_strength: f32,
    env_light_strength: f32,
    iridescence_power: f32,
    falloff_distance: f32,
    global_light: f32,
    alpha_threshold: f32,
    mix_factor_scale: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

const PI: f32 = 3.141592654;
const TAU: f32 = 2.0 * PI;

// rot matrix
fn rot(a: f32) -> mat2x2<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat2x2<f32>(c, s, -s, c);
}

fn oscillate(minn: f32, maxxi: f32, interval: f32, now: f32) -> f32 {
    return minn + (maxxi - minn) * 0.5 * (sin(TAU * now / interval) + 1.0);
}
fn pmin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
fn pmax(a: f32, b: f32, k: f32) -> f32 {
    return -pmin(-a, -b, k);
}
fn circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}
fn box_sdf(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}
// citation: Inigo Quilez
fn hs(p: vec2<f32>, c: vec2<f32>, r: f32, w: vec2<f32>) -> f32 {
    var p2 = p;
    p2.x = abs(p2.x);
    let l = length(p2);
    p2 = mat2x2<f32>(-c.x, c.y, c.y, c.x) * p2;
    
    if p2.y > 0.0 {
        p2.x = p2.x;
    } else {
        p2.x = l * sign(-c.x);
    }
    
    if p2.x > 0.0 {
        p2.y = p2.y;
    } else {
        p2.y = l;
    }
    
    p2 = vec2<f32>(p2.x, abs(p2.y - r)) - w;
    return length(max(p2, vec2<f32>(0.0))) + min(0.0, max(p2.x, p2.y));
}

fn letterc(pp: ptr<function, vec2<f32>>, off: f32) -> f32 {
    let p = *pp;
    (*pp).x -= 1.05 + off;
    let p2 = p - vec2<f32>(0.5, 0.5);
    let outer = circle(p2, 0.5);
    let inner = circle(p2, 0.3);
    let cutout = box_sdf(p2 - vec2<f32>(0.3, 0.0), vec2<f32>(0.5, 0.2));
    return max(max(outer, -inner), -cutout);
}

fn letteru(pp: ptr<function, vec2<f32>>, off: f32) -> f32 {
    let p = *pp;
    (*pp).x -= 1.0 + off;
    let p2 = p - vec2<f32>(0.5, 0.475);
    return hs(p2 - vec2<f32>(0.0, 0.125),  vec2<f32>(0.0, 1.0), 0.5, vec2<f32>(0.4, 0.1));
}

fn lettern(pp: ptr<function, vec2<f32>>, off: f32) -> f32 {
    let p = *pp;
    (*pp).x -= 1.1 + off;
    // Left vertical bar
    let leftBar = box_sdf(p - vec2<f32>(0.2, 0.5), vec2<f32>(0.1, 0.5));
    // Right vertical bar
    let rightBar = box_sdf(p - vec2<f32>(0.8, 0.5), vec2<f32>(0.1, 0.5));
    // Diagonal connecting bar
    let diagCenter = vec2<f32>(0.5, 0.5);
    let diagRot = rot(PI / 4.0);
    let diagP = p - diagCenter;
    let rotDiagP = diagRot * diagP;
    let diagBar = box_sdf(rotDiagP, vec2<f32>(0.7, 0.1));
    return min(min(leftBar, rightBar), diagBar);
}
fn lettere(pp: ptr<function, vec2<f32>>, off: f32) -> f32 {
    let p = *pp;
    (*pp).x -= 1.0 + off;
    let p2 = p - vec2<f32>(0.5, 0.5);
    // Vertical bar
    let bar = box_sdf(p2 - vec2<f32>(-0.3, 0.0), vec2<f32>(0.1, 0.5));
    // Three horizontal bars
    let topBar = box_sdf(p2 - vec2<f32>(0.0, 0.4), vec2<f32>(0.3, 0.1));
    let midBar = box_sdf(p2 - vec2<f32>(0.0, 0.0), vec2<f32>(0.3, 0.1));
    let botBar = box_sdf(p2 - vec2<f32>(0.0, -0.4), vec2<f32>(0.3, 0.1));
    return min(min(min(bar, topBar), midBar), botBar);
}

// Letter S
fn letters(pp: ptr<function, vec2<f32>>, off: f32) -> f32 {
    let rots1 = rot(-PI/6.0 - PI/2.0);
    let rots2 = rot(PI);
    let p = *pp;
    (*pp).x -= 0.875 + off;
    var p2 = p - vec2<f32>(0.435, 0.5);
    p2 = rots1 * p2;
    let u = hs(p2 - vec2<f32>(-0.25 * 3.0 / 4.0, -0.125 / 2.0),  vec2<f32>(0.0, 1.0), 0.375, vec2<f32>(0.2, 0.1));
    p2 = rots2 * p2;
    let l = hs(p2 - vec2<f32>(-0.25 * 3.0 / 4.0, -0.125 / 2.0),  vec2<f32>(0.0, 1.0), 0.375, vec2<f32>(0.2, 0.1));
    return min(u, l);
}
//CUNEUS
fn cuneus(p: vec2<f32>, off: f32) -> f32 {
    var p2 = p + vec2<f32>(3.0 + 3.0 * off, 0.5);
    var d = 1.0e6;
    d = min(d, letterc(&p2, off));
    d = min(d, letteru(&p2, off));
    d = min(d, lettern(&p2, off));
    d = min(d, lettere(&p2, off));
    d = min(d, letteru(&p2, off));
    d = min(d, letters(&p2, off));
    
    return d;
}

fn getLightIntensity(uv: vec2<f32>, textDist: f32, layer: f32, time: f32) -> f32 {
    var lightPositions: array<vec2<f32>, 8>;
    let baseOffset = 3.0 + 3.0 * layer;
    
    // C
    lightPositions[0] = vec2<f32>(-baseOffset + 0.5, 0.5);
    
    // U
    lightPositions[1] = vec2<f32>(-baseOffset + 1.55, 0.5);
    
    // N
    lightPositions[2] = vec2<f32>(-baseOffset + 2.65, 0.5);
    
    // E
    lightPositions[3] = vec2<f32>(-baseOffset + 3.65, 0.5);
    
    // U
    lightPositions[4] = vec2<f32>(-baseOffset + 4.65, 0.5);
    
    // S
    lightPositions[5] = vec2<f32>(-baseOffset + 5.525, 0.5);
    
    // Additional dynamic points
    lightPositions[6] = vec2<f32>(-baseOffset + oscillate(0.0, 6.0, 8.0, time), 
                    oscillate(0.0, 1.0, 5.0, time));
    
    lightPositions[7] = vec2<f32>(-baseOffset + oscillate(6.0, 0.0, 7.0, time), 
                    oscillate(1.0, 0.0, 6.0, time));
    
    let phaseShift1 = sin(layer * 13.37 + time * 0.3);
    let phaseShift2 = cos(layer * 7.54 - time * 0.4);
    let phaseShift3 = sin(layer * 9.21 + time * 0.5);
    
    var totalLight = 0.0;
    var totalWeight = 0.0;
    
    // Calculate light contribution from each key position
    for(var i = 0; i < 8; i++) {
        // Dynamic light position with some movement
        let lightPos = lightPositions[i] + vec2<f32>(
            cos(time * 0.5 + f32(i) * 0.7 + phaseShift1) * 0.2,
            sin(time * 0.7 + f32(i) * 0.5 + phaseShift2) * 0.2
        );
        
        let dist = length(uv - lightPos);
        let falloff = 1.0 / (1.0 + dist * params.falloff_distance * 
                   (1.0 + 0.3 * sin(phaseShift3 * f32(i) + time)));
        
        // Weight based on position and time
        let weight = 0.2 + 0.1 * sin(layer * f32(i) * 1.3 + time * 0.3);
        totalWeight += weight;
        totalLight += falloff * weight;
    }
    // Normalize weights
    totalLight = totalLight / totalWeight * 1.5;
    
    // Edge highlighting (rim effect)
    // normal using the distance field...
    let eps = vec2<f32>(0.001, 0.0);
    let normal = normalize(vec2<f32>(
        cuneus(uv + eps.xy, layer) - cuneus(uv - eps.xy, layer),
        cuneus(uv + eps.yx, layer) - cuneus(uv - eps.yx, layer)
    ));
    
    // Rim lighting
    let rimFactor = 1.0 - abs(dot(normalize(uv), normal));
    let rim = pow(rimFactor, params.rim_power);
    
    // lets aply rim lighting only near edges
    let edgeFactor = smoothstep(0.0, 0.05, abs(textDist));
    let adjustedRim = rim * (1.0 - edgeFactor);
    
    // Ambient occlusion effect
    let ao = 1.1 - (layer * params.ao_strength) * (1.0 + 0.2 * sin(layer * 20.0 + time));
    

    let shimmer = sin(layer * 10.0 + time) * cos(layer * 7.0 - time * 0.5) * 0.5 + 0.5;
    return (totalLight * ao * params.light_intensity + adjustedRim * 0.8) * (0.9 + 0.2 * shimmer);
}

// Environmental light: (g: get, E: env, L: light)
fn gel(uv: vec2<f32>, layer: f32, time: f32) -> f32 {
    let lightDir = normalize(vec2<f32>(cos(time * 0.5), sin(time * 0.5)));
    
    let eps = vec2<f32>(0.001, 0.0);
    let normal = normalize(vec2<f32>(
        cuneus(uv + eps.xy, layer) - cuneus(uv - eps.xy, layer),
        cuneus(uv + eps.yx, layer) - cuneus(uv - eps.yx, layer)
    ));
    
    var envLight = dot(normal, lightDir);
    envLight = envLight * 0.5 + 0.5;
    
    let depth = 1.0 - (layer / 1.0);
    let layerEffect = sin(layer * 4.0 + time) * 0.5 + 0.5;
    
    return mix(envLight, layerEffect, 0.5) * depth * params.env_light_strength;
}

fn gamma(color: vec3<f32>, gamma_value: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma_value));
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    var fragColor = vec4<f32>(vec3<f32>(params.background_color), 1.0);
    
    let dimensions = u_resolution.dimensions;
    let uv = 6.0 * vec2<f32>(
        fragCoord.x - 0.5 * dimensions.x, 
        -(fragCoord.y - 0.5 * dimensions.y)
    ) / dimensions.y;
    let t = u_time.time * 0.5;
    
    // Layer-based approach for depth and animation
    for(var i = 0.1; i < 1.0; i += 0.2) {
        let layer = i;
        
        let angle = sin(t * 0.3 + layer * 0.5) * 0.05;
        let offsetUV = uv * rot(angle);
        
        let scale = 0.1 + 0.02 * sin(t * 0.2 + layer * 0.7);
        
        //  SDF for the text at this layer
        let textDist = cuneus(offsetUV, scale * layer);
        
        // Alpha for edge transitions
        let alpha = smoothstep(-0.01, 0.01, textDist);
        
        // Skip distant areas
        if (alpha > params.alpha_threshold && abs(textDist) > 0.1) {
            continue;
        }
        
        let lightIntensity = getLightIntensity(offsetUV, textDist, layer, t);
        let envLight = gel(offsetUV, layer, t);
        
        // Color parameters
        let colorIntensity = 0.8 + 
            0.5 * sin(layer * 13.37 + t) * 
            cos(layer * 7.54 - t * 0.4) * 
            sin(angle * 3.0 + t * 0.7);
            
        let colorShift = 0.2 + 
            0.1 * cos(layer * 9.21 + t * 0.5) * 
            sin(angle * 5.0 - t * 0.3);
        
        let hue = sin(layer * 0.8 + angle * 2.0 + params.hue_color + t * 0.3) 
                * colorShift + colorIntensity;
        
        let litColor = hue * (lightIntensity * params.global_light + envLight);
        
        // Iridescence 
        let depthFactor = 0.0 - (i - 0.1) / (3.0 - 0.1);
        let iridescence = sin(dot(offsetUV, offsetUV) * 3.0 + t) * params.iridescence_power * depthFactor + 0.9;
        let litColorIrid = litColor * vec4<f32>(iridescence, iridescence * 0.95, iridescence * 1.05, 1.0);
        
        // edge transitions
        let mixFactor = 0.1 / (abs(textDist) + 0.05) * (1.0 - depthFactor * params.mix_factor_scale);
        
        // Combine with prev layers
        fragColor = mix(litColorIrid, fragColor, alpha) * 
                   mix(vec4<f32>(1.0), hue + 0.3 * (0.2 - alpha) * (offsetUV.y / (abs(textDist) + 0.1) + lightIntensity), mixFactor);
    }
    fragColor = vec4<f32>(fragColor.rgb * params.global_light, 1.0);
    fragColor = vec4<f32>(gamma(fragColor.rgb, 0.45), fragColor.a);
    return fragColor;
}