// Voronoi w/ stud-style 3D treatment — Enes Altun 2025, CC BY-NC-SA 3.0
// L/S helper functions adapted from FabriceNeyret 2025: https://www.shadertoy.com/view/3flGD7

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: Params;
@group(1) @binding(2) var input_texture: texture_2d<f32>;
@group(1) @binding(3) var input_sampler: sampler;

struct Params {
    scl: f32,        // cell scale
    off: f32,        // offset (animation/jitter)
    cidx: f32,       // cell index for color pick
    ew: f32,         // edge width
    hl: f32,         // edge highlight
    grn: f32,        // grain
    gam: f32,        // gamma
    sh_s: f32,       // shadow str
    sh_d: f32,       // shadow dist
    ao_s: f32,       // ao str
    sp_p: f32,       // spec pow
    sp_s: f32,       // spec str
    enh: f32,        // edge enh
    sh: f32,         // stud h
    bh: f32,         // base h
    rim: f32,        // rim str
    rsm: f32,        // res scale mult
    shm: f32,        // stud h mult
    lx: f32,         // light dir x
    ly: f32,         // light dir y
    lr: f32, lg: f32, lb: f32, // light rgb
    dsc: f32,        // depth scale
    ebl: f32,        // edge blend
    studr: f32,      // stud radius (inner bump)
    _p1: f32, _p2: f32, _p3: f32,
};

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;

fn aces(x: v3) -> v3 {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), v3(0.), v3(1.));
}

fn rnd(s: v2) -> f32 {
    return fract(sin(dot(s, v2(12.9898, 78.233))) * 43758.5453);
}

fn lumi(c: v3) -> f32 { return dot(c, v3(0.299, 0.587, 0.114)); }

// Closest point on segment distance solution (FabriceNeyret, 2025)
fn L(p: v2, a: v2, b: v2) -> f32 {
    let pl = p - a;
    let bl = b - a;
    return length(pl - bl * clamp(dot(pl, bl) / dot(bl, bl), 0., 1.));
}

// Site jitter solution (FabriceNeyret, 2025)
fn S(P: v2) -> v2 {
    let R = v2(1., 87.);
    return P + 0.5 * fract(1e4 * sin((P) * mat2x2<f32>(R.x, -R.x, R.y, -R.y)))
         + 0.25 + 0.25 * cos(u_time.time + 6.3 * fract(1e4 * sin(dot(P, R - 37.0))) + v2(0., 11.));
}

// Voronoi cell info (4 nearest sites + edge dist)
struct VInfo {
    A0: v2, A1: v2, A2: v2, A3: v2,    // 4 nearest sites
    d_edge: f32,                       // signed dist to nearest edge
    d_site: f32,                       // dist to closest site
    cidx_uv: v2,                       // uv to sample for cell color
};

fn voronoi(U: v2) -> VInfo {
    var l = v4(9.);
    var A0 = v2(0.); var A1 = v2(0.); var A2 = v2(0.); var A3 = v2(0.);

    for (var k = 0; k < 9; k++) {
        let P = S(floor(U) + v2(f32(k % 3), f32(k / 3)) + p.off);
        let d = length(P - U);
        if (d < l.x) {
            l = v4(d, l.x, l.y, l.z);
            A3 = A2; A2 = A1; A1 = A0; A0 = P;
        } else if (d < l.y) {
            l = v4(l.x, d, l.y, l.z);
            A3 = A2; A2 = A1; A1 = P;
        } else if (d < l.z) {
            l = v4(l.x, l.y, d, l.z);
            A3 = A2; A2 = P;
        } else if (d < l.w) {
            l.w = d;
            A3 = P;
        }
    }

    // Distance to nearest cell edge (perpendicular bisector trick)
    var P = A1 - A0;
    var d = length(P)/2. - dot(U - A0, P) / length(P);
    P = A2 - A0;
    d = min(d, length(P)/2. - dot(U - A0, P) / length(P));
    P = A3 - A0;
    d = min(d, length(P)/2. - dot(U - A0, P) / length(P));

    var info: VInfo;
    info.A0 = A0; info.A1 = A1; info.A2 = A2; info.A3 = A3;
    info.d_edge = d;
    info.d_site = l.x;
    return info;
}

// Height map: cell base puff + stud at site
// U is in jittered cell space (same units as voronoi() input)
fn hmap(U: v2, ls: f32) -> f32 {
    let info = voronoi(U);

    // Base height: rises sharply at cell edge, plateaus inside
    let edge_chamf = clamp(info.d_edge / (0.06 + ls * 0.2), 0., 1.);
    let h_base = p.bh * edge_chamf;

    // Stud (inner bump) centered on the site
    let dr = length(U - info.A0);
    let stud_chamf = clamp((p.studr - dr) / (0.04 + ls * 0.15), 0., 1.);
    let h_stud = p.sh * p.shm * stud_chamf;

    return h_base + h_stud * edge_chamf;
}

fn nmap(U: v2, ls: f32) -> v3 {
    let em = 0.025 + ls * 0.05;
    let e = v2(em, 0.);
    let dx = hmap(U + e.xy, ls) - hmap(U - e.xy, ls);
    let dy = hmap(U + e.yx, ls) - hmap(U - e.yx, ls);
    return normalize(v3(-dx, -dy, 0.05 + ls * 0.5));
}

// Sample input image at a given cell site (in jittered cell-space).
// We need to map cell-space site -> uv [0,1].
fn site_uv(site: v2, R: v2) -> v2 {
    return (site * R.y / p.scl) / R;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let R = v2(textureDimensions(output));
    let coords = vec2<u32>(g.xy);
    if (coords.x >= u32(R.x) || coords.y >= u32(R.y)) { return; }

    let FragCoord = v2(f32(coords.x), R.y - f32(coords.y));
    let uv = FragCoord / R;

    let U = p.scl * FragCoord.xy / R.y;

    let info = voronoi(U);

    // approx pixels per cell
    let ppc = R.y / p.scl;
    let ls = smoothstep(60., 14., ppc);

    // Sample color at the cell's site 
    let cuv = site_uv(info.A0, R);
    let tex_dims = textureDimensions(input_texture);
    var bc = v3(0.5, 0.5, 1.0);
    if (tex_dims.x > 1u && tex_dims.y > 1u) {
        let tc = vec2<i32>(i32(cuv.x * f32(tex_dims.x)), i32((1. - cuv.y) * f32(tex_dims.y)));
        let cc = clamp(tc, vec2<i32>(0), vec2<i32>(tex_dims) - vec2<i32>(1));
        bc = textureLoad(input_texture, cc, 0).rgb;
    }


    var picked_uv = cuv;
    if (p.cidx >= 0.5 && p.cidx < 1.5) { picked_uv = site_uv(info.A1, R); }
    else if (p.cidx >= 1.5 && p.cidx < 2.5) { picked_uv = site_uv(info.A2, R); }
    else if (p.cidx >= 2.5) { picked_uv = site_uv(info.A3, R); }
    if (tex_dims.x > 1u && tex_dims.y > 1u) {
        let tc = vec2<i32>(i32(picked_uv.x * f32(tex_dims.x)), i32((1. - picked_uv.y) * f32(tex_dims.y)));
        let cc = clamp(tc, vec2<i32>(0), vec2<i32>(tex_dims) - vec2<i32>(1));
        bc = textureLoad(input_texture, cc, 0).rgb;
    }

    // Height & normal for THIS pixel
    let h = hmap(U, ls);
    let n = nmap(U, ls);

    // Mortar/edge gap shading
    if (info.d_edge < 0.005 && ls < 0.85) {
        let gc = mix(v3(0.04), bc * 0.4, ls);
        textureStore(output, vec2<i32>(coords.xy), v4(gc, 1.));
        return;
    }

    // Edge proximity
    let ee = (1.1 / max(info.d_edge, 0.01)) * (1. - ls);
    let ef = smoothstep(0., 2., ee);

    // AO
    var ao = 1.;
    let l2 = v2(p.lx, p.ly);

    if (ls < 0.85) {
        let edge_ao = smoothstep(0.0, 0.15, info.d_edge);
        ao *= mix(0.55, 1.0, edge_ao);
        let dr = length(U - info.A0);
        if (dr < p.studr + 0.05 && dr > p.studr) {
            ao *= mix(0.7, 1.0, smoothstep(p.studr, p.studr + 0.05, dr));
        }
    }

    // Raymarched soft shadow along light dir
    if (ls < 0.85) {
        var sh = 0.;
        let sam = 12;
        for (var i = 0; i < sam; i++) {
            let t = f32(i) / f32(sam - 1);
            let sp = U - l2 * (0.05 + t * p.sh_d);
            let ho = hmap(sp, ls);
            sh += max(0., ho - h) * (1. - t) * p.sh_s;
        }
        ao *= 1. - smoothstep(0., 1., sh);
    }
    ao = pow(clamp(ao, 0., 1.), p.ao_s);

    // Lighting
    let l = normalize(v3(l2, 1.));
    let lc = v3(p.lr, p.lg, p.lb);
    let v  = normalize(v3(0., 0., 1.));

    // Multi-sample diffuse accumulation (depth scaling)
    var acc = v3(0.);
    var tw = 0.;
    let dsam = 4;
    for (var i = 0; i < dsam; i++) {
        let dl = f32(i) / f32(dsam - 1);
        let sc = p.dsc + dl * (1. - p.dsc);
        let su = (U - info.A0) * sc + info.A0;
        let sh = hmap(su, ls);
        if (sh > 0.) {
            let sn = nmap(su, ls);
            let ld = max(0., dot(sn, l));
            let la = sn.z * 0.5 + 0.5;
            let w  = (1. - dl * 0.3) * (0.5 + 0.5 * ld);
            acc += v3(la, ld, 1.) * w;
            tw += w;
        }
    }
    acc /= max(tw, 0.1);

    let sky = v3(0.2, 0.25, 0.3);
    let gnd = bc * 0.1;
    let amb = mix(gnd, sky, n.z * 0.5 + 0.5) * ao * acc.x;
    let diff = max(0., dot(n, l)) * acc.y * lc;

    let h_vec = normalize(l + v);
    let spec = pow(max(0., dot(n, h_vec)), p.sp_p) * lc * p.sp_s;

    // Rim using a perturbed normal (plastic edge)
    let rn = normalize(v3(U - info.A0, h * 2.));
    let rim = pow(1. - abs(dot(rn, v)), 2.) * diff * p.rim;

    var col = bc * (amb + diff) + spec;

    // Soft directional shade
    let local = U - info.A0;
    let ssd = local.y / max(length(local), 0.01);
    let ss = 0.5 + 0.5 * ssd * lumi(diff);
    col = mix(col, col + bc * ss * ee * p.enh, ef * p.ebl);
    col += rim * lc;

    // Edge highlight
    let eF = smoothstep(-p.ew, p.ew, info.d_edge);
    let eH = smoothstep(0.1, 0.0, abs(info.d_edge)) * p.hl;
    col += bc * eH * 0.15;
    col = mix(v3(0.0), col, eF);

    // Corner dot accent (FabriceNeyret)
    let a = dot(info.A0, info.A0);
    let b = dot(info.A1, info.A1);
    let c = dot(info.A2, info.A2);
    let mat = mat2x2<f32>(
        info.A2.x - info.A1.x, info.A2.y - info.A1.y,
        info.A2.x - info.A0.x, info.A2.y - info.A0.y
    );
    let CP = v2(c - b, c - a) / 2. * mat;
    let cP = smoothstep(15./R.y, 0., length(CP - U) - 0.02);
    col = mix(col, bc * 0.2, cP * 0.7);

    // Grain (LOD-aware)
    col += (rnd(uv) - 0.5) * p.grn * (1. - ls);

    // tone + gamma
    col = pow(aces(col), v3(1. / max(p.gam, 0.05)));

    textureStore(output, vec2<i32>(i32(coords.x), i32(coords.y)), v4(clamp(col, v3(0.), v3(1.)), 1.));
}
