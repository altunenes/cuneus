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
    lambda: f32,
    theta: f32,
    alpha: f32,
    sigma: f32,
    gamma: f32,
    blue: f32,
    a: f32,
    b: f32,
    base_color_r: f32,
    base_color_g: f32,
    base_color_b: f32,
    accent_color_r: f32,
    accent_color_g: f32,
    accent_color_b: f32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    gamma_correction: f32,
    aces_tonemapping: f32,
    _padding: f32,
};

const PI: f32 = 3.14159265358979323846;
const LIGHT_INTENSITY: f32 = 2.2;
const RIM_POWER: f32 = 2.0;
const AO_STRENGTH: f32 = 0.05;
const ENV_LIGHT_STRENGTH: f32 = 0.4;
const IRIDESCENCE_POWER: f32 = 0.15;
const FALLOFF_DISTANCE: f32 = 2.5;
const VIGNETTE_STRENGTH: f32 = 0.25;

fn rotate(angle: f32) -> mat2x2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat2x2<f32>(
        vec2<f32>(c, -s),
        vec2<f32>(s, c)
    );
}

fn oscillate(minn: f32, maxxi: f32, interval: f32, now: f32) -> f32 {
    return minn + (maxxi - minn) * 0.5 * (sin(2.0 * PI * now / interval) + 1.0);
}

fn getpentpo(angle: f32, radius: f32) -> vec2<f32> {
    return vec2<f32>(cos(angle), sin(angle)) * radius;
}

fn sdPentagon(p: vec2<f32>, r: f32, angle: f32) -> f32 {
    var rotatedP = rotate(-angle) * p;
    let vertices = params.lambda;
    let angleStep = params.theta * PI / vertices;

    var d = length(rotatedP) - r;

    var i: f32 = 0.0;
    while(i < vertices) {
        let a1 = angleStep * i;
        let a2 = angleStep * (i + 1.0);

        let p1 = getpentpo(a1, r);
        let p2 = getpentpo(a2, r);

        let edge = p2 - p1;
        let normal = normalize(vec2<f32>(edge.y, -edge.x));
        let dist = dot(rotatedP - p1, normal);

        d = max(d, dist);
        i = i + 1.0;
    }

    return max(d, 0.1);
}

//get light
fn gl(uv: vec2<f32>, pentagon: f32, layer: f32, time: f32, angle: f32) -> f32 {
    var vertices: array<vec2<f32>, 5>;
    let r = layer * 0.8;

    for(var i: i32 = 0; i < 5; i = i + 1) {
        let a = (f32(i) / 5.0) * 2.0 * PI;
        vertices[i] = getpentpo(a, r);
    }

    let rot = rotate(angle);
    for(var i: i32 = 0; i < 5; i = i + 1) {
        vertices[i] = rot * vertices[i] + uv;
    }

    let phaseShift1 = sin(layer * 13.37 + time * 0.3);
    let phaseShift2 = cos(layer * 7.54 - time * 0.4);
    let phaseShift3 = sin(layer * 9.21 + time * 0.5);
    let phaseShift4 = cos(layer * 11.13 + time * 0.6);
    let phaseShift5 = sin(layer * 8.45 + time * 0.7);

    var lp: array<vec2<f32>, 5>;
    lp[0] = vertices[0] + vec2<f32>(cos(time * 0.5 + phaseShift1), sin(time * 0.7 + phaseShift2)) * 0.3;
    lp[1] = vertices[1] + vec2<f32>(sin(time * 0.3 + phaseShift2), cos(time * 0.4 + phaseShift3)) * 0.3;
    lp[2] = vertices[2] + vec2<f32>(cos(time * 0.6 + phaseShift3), sin(time * 0.5 + phaseShift4)) * 0.3;
    lp[3] = vertices[3] + vec2<f32>(sin(time * 0.4 + phaseShift4), cos(time * 0.6 + phaseShift5)) * 0.3;
    lp[4] = vertices[4] + vec2<f32>(cos(time * 0.5 + phaseShift5), sin(time * 0.3 + phaseShift1)) * 0.3;

    var distances: array<f32, 5>;
    var falloffs: array<f32, 5>;
    var weights: array<f32, 5>;
    var totalWeight: f32 = 0.0;

    for(var i: i32 = 0; i < 5; i = i + 1) {
        distances[i] = length(uv - lp[i]) *
            (1.0 + 0.2 * sin(layer * (15.0 + f32(i)) + time * (0.7 + f32(i) * 0.1)));

        falloffs[i] = 2.0 / (4.0 + distances[i] * FALLOFF_DISTANCE);
        weights[i] = 0.2 + 0.1 * sin(layer * (11.0 + f32(i) * 2.0) + time * (0.5 + f32(i) * 0.1));
        totalWeight = totalWeight + weights[i];
    }

    for(var i: i32 = 0; i < 5; i = i + 1) {
        weights[i] = weights[i] / totalWeight;
    }

    let ao = 0.4 - (layer * AO_STRENGTH) * (1.0 + 0.2 * sin(layer * 20.0 + time));

    let normal = normalize(vec2<f32>(cos(angle + layer), sin(angle + layer)));
    var rim = 2.1 - abs(dot(normalize(uv), normal));
    rim = pow(rim, RIM_POWER);

    var vertexLights: f32 = 0.0;
    for(var i: i32 = 0; i < 5; i = i + 1) {
        vertexLights = vertexLights + falloffs[i] * weights[i];
    }

    let shimmer = sin(layer * 10.0 + time) * cos(layer * 7.0 - time);
    vertexLights = vertexLights * (1.0 + 0.15 * shimmer);

    return vertexLights * ao * LIGHT_INTENSITY + rim * 0.4;
}
//env light get
fn geeee(uv: vec2<f32>, angle: f32, layer: f32, time: f32) -> f32 {
    let lightDir = normalize(vec2<f32>(cos(time), sin(time)));
    let normal = normalize(vec2<f32>(cos(angle), sin(angle)));
    var envLight = dot(normal, lightDir);
    envLight = envLight * 0.5 + 0.5;

    let depth = 1.0 - (layer / 1.5);
    let layerEffect = sin(layer * 4.0 + time) * 0.5 + 0.5;

    return mix(envLight, layerEffect, 0.5) * depth * ENV_LIGHT_STRENGTH;
}

fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let bg_color = vec3<f32>(params.background_r, params.background_g, params.background_b);
    let bg = oscillate(0.6, 0.6, 8.0, u_time.time);
    var fragColor = vec4<f32>(bg_color * bg, 1.0);
    let screen_size = u_resolution.dimensions;
    let t = u_time.time * 0.5;
    var angle: f32 = 0.25;
    let foldPattern = cos(t * 0.5) * PI * 0.25;
    let globalLight = oscillate(0.4, 1.5, 8.0, u_time.time);
    let asd = oscillate(params.alpha, params.sigma, 25.0, u_time.time);

    var i = params.gamma;
    while(i > 0.003) {
        let layer = i * 1.0;

        let fold = sin(t + layer * 0.2) * cos(t * 0.5 + layer * 0.1);
        let wave = cos(t * 0.7 + layer * 0.15) * sin(t * 0.3 + i);

        let temp_angle = foldPattern;
        angle = angle - sin(angle - sin(temp_angle)) * (0.5 + 1.5 * sin(layer));

        let alternatingFold = sign(sin(layer * params.b)) * sin(t + i * 2.0);

        var uv = 4.7 * (FragCoord.xy - screen_size) / screen_size.y;
        uv.y = uv.y + 2.3;
        uv.x = uv.x + 3.0;
        uv = rotate(i + (angle + alternatingFold) + foldPattern) * uv;

        let pentagon = sdPentagon(uv, i, angle + t * (1.0 + 0.2 * sin(layer)));
        let alpha = smoothstep(0.0, 0.2, (pentagon - 0.1) * screen_size.y * 0.15);

        let lightIntensity = gl(uv, pentagon, i, t, angle);
        let envLight = geeee(uv, angle, i, t);

        let des = oscillate(0.5, 1.0, 5.0, u_time.time);
        let des2 = oscillate(0.1, 1.0, 5.0, u_time.time);

        let colorIntensity = 0.8 +
            0.2 * sin(layer * 24.37 + t) *
            cos(layer * 12.54 - t * 0.4) *
            sin(angle * 3.0 + t * 0.7);

        let colorShift = 0.2 +
            0.1 * cos(layer * 9.21 + t * 0.5) *
            sin(angle * 5.0 - t * 0.3);

        let base_color = vec3<f32>(params.base_color_r, params.base_color_g, params.base_color_b);
        let accent_color = vec3<f32>(params.accent_color_r, params.accent_color_g, params.accent_color_b);
        let original_hue = sin(i / des2 + angle / des + vec4<f32>(params.blue, 2.0, 3.0, 1.0) + fold * 0.5) * colorShift + colorIntensity;
        let color_influence = (base_color + accent_color) * 0.5;
        let enhanced_hue = original_hue * vec4<f32>(color_influence, 1.0);
        let hue = enhanced_hue;

        let litColor = hue * (lightIntensity * globalLight + envLight);

        let depthFactor = params.a - (i - 0.003) / (1.5 - 0.003);
        let iridescence = sin(dot(uv, uv) * 4.0 + t) * IRIDESCENCE_POWER * depthFactor + 0.95;
        let litColorIrid = litColor * vec4<f32>(iridescence, iridescence * 0.98, iridescence * 1.02, 1.0);

        let des4 = oscillate(0.4, 0.2, 10.0, u_time.time);
        let des12 = oscillate(0.1, 0.1, 10.0, u_time.time);

        let mixFactor = des4 / (pentagon + 0.01) * (0.6 - depthFactor * 0.25);
        fragColor = mix(litColorIrid, fragColor, alpha) *
                   mix(vec4<f32>(1.0), hue + 1.5 * alpha * des12 * (uv.x / pentagon + lightIntensity), mixFactor);

        i = i - asd;
    }

    let des2_final = oscillate(0.7, 0.7, 5.0, u_time.time);
    fragColor = vec4<f32>(fragColor.rgb * (des2_final + globalLight * 0.1), 1.0);

    let vignetteUV = (FragCoord.xy - 0.5 * vec2<f32>(1920.0, 1080.0)) / 1080.0;
    let vignette = 1.0 - dot(vignetteUV, vignetteUV) * VIGNETTE_STRENGTH;

    var final_color = fragColor.rgb;
    if (params.aces_tonemapping > 0.0) {
        final_color = mix(final_color, aces_tonemap(final_color), params.aces_tonemapping);
    }
    let cor = gamma(final_color, params.gamma_correction);
    return vec4<f32>(cor, fragColor.a);
}