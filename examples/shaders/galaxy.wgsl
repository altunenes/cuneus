// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Unported License.
// Spiral galaxy, ray-marched like the black hole's disk: density-wave arms, multifractal clouds,
// core light through a shadow map, resolved stars, young clusters, dust.
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

struct Params {
    arms: f32, twist: f32, bar: f32, core: f32,
    young: f32, nebula: f32, dust: f32, thick: f32,
    anim: f32, orbit: f32, view: f32, zm: f32,
    stars: f32, yaw: f32, blm: f32, vig: f32,
    br: f32, expo: f32, gam: f32, sat: f32,
    hue: f32, spc: f32, relief: f32, field: f32,
    suns: f32, halo: f32, ion: f32, taw: f32,
    shp: f32, tlt: f32, logd: f32, clus: f32,
    size: f32, seed: f32, nscale: f32, glow: f32,
    sky_den: f32, sky_blue: f32, sky_cld: f32, sky_hue: f32,
    pyaw: f32, pview: f32, pzm: f32, ptime: f32,
    sglow: f32, bthr: f32, fil: f32, fsc: f32,
    dhard: f32, fwd: f32, lump: f32, neb2: f32,
    nhue: f32, hii: f32, glob: f32, msc: f32,
    dist: f32, pdist: f32, oval: f32, jwst: f32,
    dalb: f32, shd: f32, cine: f32, ctemp: f32,
    lines: f32, hst: f32, redd: f32, ap: f32,
    focus: f32, bokeh: f32, z: f32, dop: f32,
    exc: f32, hb: f32, nii: f32, lviv: f32,
    gflow: f32, cevo: f32, swirl: f32, twk: f32,
    brad: f32, cr: f32, life: f32, sn: f32,
    bub: f32, hpres: f32, frng: f32, brst: f32,
    res: f32, snr: f32, wind: f32, wlen: f32,
    wopen: f32, pan_x: f32, pan_y: f32, pan_z: f32,
};
@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: Params;
@group(3) @binding(0) var t0: texture_2d<f32>; @group(3) @binding(1) var s0: sampler;
@group(3) @binding(2) var t1: texture_2d<f32>; @group(3) @binding(3) var s1: sampler;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>;
alias m2 = mat2x2<f32>; alias m3 = mat3x3<f32>; alias u3 = vec3<u32>;
const pi = 3.14159265; const tau = 6.28318531;
const LUMA = v3(0.2126, 0.7152, 0.0722);

fn hu(a0: u32) -> u32 { var a=a0; a^=a>>16u; a*=0x7feb352du; a^=a>>15u; a*=0x846ca68bu; a^=a>>16u; return a; }

// YCoCg (gelami/mrange, CC0: https://www.shadertoy.com/view/fXSGR1)
fn r2y(c: v3) -> v3 { return m3(0.25,0.5,-0.25, 0.5,0.,0.5, 0.25,-0.5,-0.25)*c; }
fn y2r(c: v3) -> v3 { return m3(1.,1.,1., 1.,0.,-1., -1.,1.,-1.)*c; }
fn aces(c: v3) -> v3 {
    let n1=m3(0.59719,0.07600,0.02840, 0.35458,0.90834,0.13383, 0.04823,0.01566,0.83777);
    let n2=m3(1.60475,-0.10208,-0.00327, -0.53108,1.10813,-0.07276, -0.07367,-0.00605,1.07602);
    var v=n1*c; let a=v*(v+0.0245786)-0.000090537; let b=v*(0.983729*v+0.4329510)+0.238081;
    return n2*(a/b);
}

// noise, as in the black hole
fn hash13(q: v3) -> f32 { var p3 = fract(q * v3(0.1031, 0.1107, 0.0973)); p3 += dot(p3, p3.yzx + 19.19); return fract((p3.x + p3.y) * p3.z); }
fn vnoise(x: v3) -> f32 {
    let i = floor(x); let f = fract(x); let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i), hash13(i + v3(1.,0.,0.)), u.x), mix(hash13(i + v3(0.,1.,0.)), hash13(i + v3(1.,1.,0.)), u.x), u.y),
               mix(mix(hash13(i + v3(0.,0.,1.)), hash13(i + v3(1.,0.,1.)), u.x), mix(hash13(i + v3(0.,1.,1.)), hash13(i + v3(1.,1.,1.)), u.x), u.y), u.z);
}
fn fbm(x: v3) -> f32 {
    var v = 0.0; var amp = 0.5; var pos = x;
    for (var i = 0; i < 4; i++) { v += amp * vnoise(pos); pos = pos * 2.02 + v3(11.3, 7.7, 3.1); amp *= 0.5; }
    return v;
}
// multifractal noise -> wispy filaments
// fpn: pixel footprint in noise space, sub-pixel octaves fade out
var<private> FP: f32 = 0.;
fn cloud_noise(p_in: v3, octaves: i32, contrast: f32, fpn: f32) -> f32 {
    var acc = 1.0; var freq = 1.0;
    for (var i = 0; i < octaves; i++) {
        let fade = clamp(1. / (freq * fpn + 1e-6) - 1., 0., 1.);
        if (fade <= 0.) { break; }
        acc *= 1.0 + 0.1 * (vnoise(p_in * freq) * 2.0 - 1.0) * fade; freq *= 3.0;
    }
    return log(1.0 + pow(max(0.0, acc), contrast));
}
fn curl2(q: v3) -> v2 {
    let e = 0.5;
    return v2(vnoise(q + v3(0., e, 0.)) - vnoise(q - v3(0., e, 0.)), -(vnoise(q + v3(e, 0., 0.)) - vnoise(q - v3(e, 0., 0.)))) / (2.0 * e);
}

// blackbody colour (Tanner Helland's fit)
fn bb(T: f32) -> v3 {
    let t = clamp(T, 1000., 40000.) / 100.;
    var c: v3;
    if (t <= 66.) {
        c.x = 1.;
        c.y = clamp((99.4708 * log(t) - 161.1196) / 255., 0., 1.);
        c.z = select(clamp((138.5177 * log(t - 10.) - 305.0448) / 255., 0., 1.), 0., t <= 19.);
    } else {
        c.x = clamp(329.6987 * pow(t - 60., -0.1332) / 255., 0., 1.);
        c.y = clamp(288.1222 * pow(t - 60., -0.0755) / 255., 0., 1.);
        c.z = 1.;
    }
    let l = pow(c, v3(2.2));
    return l / max(dot(l, LUMA), 0.05);
}
// chroma boost
fn vivid(c: v3, s: f32) -> v3 { let l = dot(c, LUMA); return max(v3(l) + (c - v3(l)) * s, v3(0.)); }

// wavelength -> linear sRGB (CIE 1931, Wyman et al. fit)
fn lobe(x: f32, m: f32, a: f32, b: f32) -> f32 { let t = (x - m) / select(b, a, x < m); return exp(-0.5 * t * t); }
fn wl_rgb(l: f32) -> v3 {
    let X = 1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2);
    let Y = 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1);
    let Z = 1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8);
    return v3(3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0.0557 * X - 0.2040 * Y + 1.0570 * Z);
}
// out of gamut: add white
fn gamut(c: v3) -> v3 { let k = c - min(min(c.r, min(c.g, c.b)), 0.); return k / max(dot(k, LUMA), 1e-5); }
// nebula lines (Halpha, [NII], [SII], [OIII]) shifted by redshift + rotation Doppler
var<private> LHt: array<v3, 9>; var<private> LOt: array<v3, 9>;
// line-of-sight speed, pattern-frame ray, star drift
var<private> VL: f32 = 0.; var<private> RD: v3; var<private> SB: f32 = 0.;
fn pos3(c: v3) -> v3 { return c - min(min(c.r, min(c.g, c.b)), 0.); }
fn hmix(f: f32) -> v3 { return wl_rgb(656.3 * f) * 2.86 + wl_rgb(486.1 * f) * p.hb + (wl_rgb(658.4 * f) * 0.8 + wl_rgb(671.6 * f) * 0.4 + wl_rgb(673.1 * f) * 0.3) * p.nii; }
fn omix(f: f32) -> v3 { return wl_rgb(500.7 * f) + wl_rgb(495.9 * f) * 0.33 + wl_rgb(486.1 * f) * 0.25 * p.hb; }
fn init_lines() {
    let lh = max(dot(pos3(hmix(1.)), LUMA), 1e-5); let lo = max(dot(pos3(omix(1.)), LUMA), 1e-5);
    for (var k = 0; k < 9; k++) {
        let f = (1. + p.z) * (1. + 0.12 * p.dop * (f32(k) / 4. - 1.));
        LHt[k] = pos3(hmix(f)) / lh; LOt[k] = pos3(omix(f)) / lo;
    }
}
// clockwise disk, flat rotation curve
fn vlos(w: v3) -> f32 { let r = length(w.xy); return dot(v3(w.y, -w.x, 0.) / max(r, 1e-4), RD) * tanh(r / 0.08); }
// redshift x Doppler
fn zfac() -> f32 { return (1. + p.z) * (1. + 0.12 * p.dop * VL); }
// shifted continuum colour
fn zcont(f: f32) -> v3 { return bb(6500. / f) / bb(6500.); }
// emission colour, hydrogen (0) to oxygen (1); Hubble palette option
fn lc(old: v3, ox: f32) -> v3 {
    let l = dot(old, LUMA); let x = clamp(ox + p.exc, 0., 1.);
    let v = clamp(VL * 0.5 + 0.5, 0., 1.) * 8.; let i = u32(min(floor(v), 7.)); let fr = v - f32(i);
    let ph = vivid(mix(mix(LHt[i], LHt[i + 1u], fr), mix(LOt[i], LOt[i + 1u], fr), x), p.lviv) * l;
    let hb = gamut(mix(v3(1.0, 0.72, 0.2), v3(0.15, 0.5, 1.0), x)) * l;
    return mix(mix(old, ph, p.lines), hb, p.hst);
}

// sky: pixel-sized stars, far galaxies, Milky Way band
fn starfield(dir: v3, pa: f32) -> v3 {
    var col = v3(0.0); let d0 = normalize(dir);
    let sg = pa * 0.75;
    for (var k = 0; k < 3; k++) {
        let scale = array<f32, 3>(90., 180., 330.)[k];
        let d = select(select(d0, d0.yzx, k == 1), d0.zxy, k == 2);
        let cell = floor(d * scale);
        let h = hash13(cell + f32(k) * 17.);
        if (h > 1. - array<f32, 3>(0.004, 0.0025, 0.0015)[k] * p.sky_den) {
            let sd = normalize(cell + 0.3 + 0.4 * v3(hash13(cell + 1.7), hash13(cell + 3.1), hash13(cell + 5.9)));
            let a = length(d - sd);
            let T = star_T(hash13(cell + 8.3), hash13(cell + 9.7), 0.15);
            col += vivid(bb(T), 1.2) * exp(-a * a / (sg * sg)) * 0.25 * pow(1. - fract(h * 113.) * 0.9, -0.7);
        }
    }
    // far galaxies
    let gc = floor(d0 * 36.);
    let gh = hash13(gc + 91.);
    if (gh > 1. - 0.008 * p.sky_den) {
        let sd = normalize(gc + 0.4 + 0.2 * v3(hash13(gc + 2.2), hash13(gc + 4.4), hash13(gc + 6.6)));
        let e1 = normalize(cross(sd, v3(0.31, 0.12, 0.94))); let e2 = cross(sd, e1);
        let dv = d0 - sd;
        let an = hash13(gc + 7.7) * pi; let ca = cos(an); let sa = sin(an);
        let x = dot(dv, e1) * ca + dot(dv, e2) * sa; let y = -dot(dv, e1) * sa + dot(dv, e2) * ca;
        let sz = mix(0.0012, 0.004, hash13(gc + 8.8)); let q = mix(0.25, 0.9, hash13(gc + 9.9));
        let r2 = (x * x + y * y / (q * q)) / (sz * sz + sg * sg);
        col += mix(v3(1.0, 0.8, 0.55), v3(0.7, 0.8, 1.0), hash13(gc + 3.3)) * (exp(-r2 * 3.) * 0.12 + exp(-r2 * 25.) * 0.25);
    }
    return col * p.field;
}
// sky gas: sparse coloured clouds cut by dust
fn sky_tint(h: f32) -> v3 {
    let k = fract(h) * 4.;
    let c = array<v3, 5>(v3(0.25, 0.4, 1.0), v3(0.65, 0.3, 1.0), v3(1.0, 0.3, 0.55), v3(0.2, 0.8, 0.8), v3(0.25, 0.4, 1.0));
    let i = u32(floor(k));
    return mix(c[i], c[i + 1u], smoothstep(0., 1., fract(k)));
}
fn skyband(dir: v3) -> v3 {
    let d = normalize(dir);
    var col = v3(0.0012, 0.0016, 0.003) * p.sky_blue;
    if (p.sky_cld <= 0.) { return col; }
    let warp = fbm(d * 1.4 + 4.0) - 0.5;
    let m = smoothstep(0.5, 0.72, fbm(d * v3(2.4, 2.4, 0.7) + warp * 1.2 + 13.));
    if (m < 0.001) { return col; }
    let n = fbm(d * 3.0 + warp * 2.0 + 21.0);
    let wisp = fbm(d * 7.0 + warp * 3.0 + 7.0);
    let tint = sky_tint(fbm(d * 0.9 + 40.) * 1.6 + p.sky_hue * 0.5);
    let lane = 1. - 0.8 * smoothstep(0.55, 0.72, fbm(d * 11. + warp * 4. + 3.));
    col += tint * m * (n * n * 0.05 + pow(wisp * n, 2.) * 0.12) * lane * p.sky_cld * mix(1., 3., p.sky_blue * 0.25);
    return col;
}

// log spiral arm pattern
fn pattern(a: f32, t: f32) -> f32 { return p.twist * 3.0 * log(1. + a * 4.) + 0.12 * t; }
fn m_arms() -> f32 { return max(round(p.arms), 1.); }
fn rotz(v: v3, a: f32) -> v3 { let c = cos(a); let s = sin(a); return v3(c * v.x - s * v.y, s * v.x + c * v.y, v.z); }
fn hn(s: ptr<function, u32>) -> f32 { *s = hu(*s); return f32(*s >> 8u) / 16777216.; }
// Schlick phase (HG approx.)
fn hg(c: f32, g: f32) -> f32 { let k = 1.55 * g - 0.55 * g * g * g; let d = 1. - k * c; return (1. - k * k) / (12.566 * d * d); }

// orbit camera (mouse + auto orbit) + cinematic drift
fn cam_state(t: f32, yaw: f32, view: f32, dist: f32) -> v3 {
    let c = t * p.cine;
    let az = 0.6 + p.orbit * t * 0.05 + yaw + c * 0.1 + 0.3 * sin(c * 0.13);
    let vw = clamp(view + 0.45 * sin(c * 0.07), 0.02, 1.52);
    return v3(az, vw, dist * (1. - 0.4 * (0.5 - 0.5 * cos(c * 0.05))));
}
fn cam_rot(v: v3) -> v3 { let st = cam_state(u_t.time, p.yaw, p.view, p.dist); return cam_pose(v, st.x, st.y); }
// camera X/Y/Z offset
fn pan() -> v3 { return v3(p.pan_x, p.pan_y, p.pan_z); }
fn cam_pos() -> v3 { let st = cam_state(u_t.time, p.yaw, p.view, p.dist); return cam_pose(pan() - v3(0., 0., st.z), st.x, st.y); }
// focus distance
fn cam_focus_dist() -> f32 { return cam_state(u_t.time, p.yaw, p.view, p.dist).z - p.pan_z; }
fn cam_pose(v: v3, az: f32, view: f32) -> v3 {
    let ca = cos(az); let sa = sin(az); let ci = cos(view); let si = sin(view);
    let y = v.y * ci + v.z * si; let z = -v.y * si + v.z * ci;
    return v3(v.x * ca + y * sa, -v.x * sa + y * ca, z);
}

// density waves: tau = time since the arm's shock (Myr)
fn omega(r: f32) -> f32 { return 0.015 * tanh(r / 0.08) / max(r, 1e-3); }
fn omp() -> f32 { return omega(p.cr); }
// galaxy clock (Myr)
fn tmyr() -> f32 { return gt() * 0.12 / omp(); }
// Rotation speed drives everything
fn gt() -> f32 { return u_t.time * 0.25 * p.anim; }
struct Arm { gas: f32, lane: f32, tau: f32, amp: f32 };
fn arm(w: v3, wob: f32) -> Arm {
    var a: Arm;
    let rho = length(w.xy); let m = m_arms();
    let psi = m * (atan2(w.y, w.x) - p.twist * 3. * log(1. + rho * 4.) + wob);
    let dO = omega(rho) - omp();
    // phase since the shock
    let xi = fract(select(psi, -psi, dO > 0.) / tau) * tau;
    let xs = select(xi, xi - tau, xi > pi);
    a.tau = xi / (m * max(abs(dO), 0.004)) / max(p.life, 0.05);
    // gas piles up behind the shock
    a.gas = 0.15 + 0.85 * smoothstep(-0.4, 0.02, xs) * exp(-max(xs - 0.08, 0.) / 1.4);
    a.lane = exp(-(xs + 0.04) * (xs + 0.04) / 0.02) + select(0., 0.3 * exp(xs / 0.7), xs < 0.);
    // Lindblad resonances
    let k = 1.414 / m;
    let ilr = p.cr * max(1. - k, 0.); let olr = p.cr * (1. + k);
    a.amp = (0.45 + 0.55 * smoothstep(ilr * 0.7, ilr * 1.3 + 0.03, rho) * (1. - smoothstep(olr * 0.8, olr * 1.05, rho)))
          * mix(0.75, 1., smoothstep(0., 0.35, abs(dO) / omp()));
    return a;
}
// young stars + HII after the shock
fn young(t: f32) -> f32 { return smoothstep(0.5, 3., t) * exp(-t / 30.); }
fn hii_t(t: f32) -> f32 { return smoothstep(0.5, 2.5, t) * exp(-t / 7.); }
// star formation bursts
fn burst(w: v3, lag: f32) -> f32 {
    if (p.brst <= 0.) { return 1.; }
    let n = vnoise(v3(w.xy * 7. + p.seed * 3.1, (tmyr() - lag) / 25. + p.seed));
    return mix(1., 0.55 + 1.6 * smoothstep(0.5, 0.75, n), p.brst);
}

// advected textures (Neyret), two phases crossfaded
const FLOWP: f32 = 24.;
struct Fl { r0: f32, r1: f32, o0: f32, o1: f32, b: f32 };
// fresh patch per reset
fn cyc(c: f32) -> f32 { return fract(c * 0.6180339) * tau; }
fn flow_at(rho: f32) -> Fl {
    var l: Fl;
    let T = tmyr() / FLOWP; let f0 = fract(T); let f1 = fract(T + 0.5);
    let k = (omega(rho) - omp()) * FLOWP * p.gflow;
    l.r0 = k * (f0 - 0.5); l.r1 = k * (f1 - 0.5);
    l.o0 = cyc(floor(T) * 2.); l.o1 = cyc(floor(T + 0.5) * 2. + 1.);
    l.b = 1. - abs(2. * f0 - 1.);
    return l;
}

// cloud field
struct F { dens: f32, dust: f32, A: f32, cloud: f32, fine: f32, q: v3, ed: v2, fl: f32,
           yb: f32, hii: f32, lane: f32, tau: f32, rot: f32, rot2: f32, ed2: v2, b2: f32 };
// cloud coordinates, sheared k along the arms
fn shear_q(w: v3, k: f32) -> v3 {
    let rho = length(w.xy);
    // evolution
    let a = atan2(w.y, w.x) - k * p.twist * 3.0 * log(1. + rho * 4.);
    return v3(cos(a) * rho, sin(a) * rho, w.z) * 3.2 * p.nscale + v3(p.seed * 1.7, p.seed * 0.9, p.cevo * gt() * 0.16);
}

// baked gas noise (bake_ga/bake_gb): coarse octaves over the unwound disk, z repeats
@group(3) @binding(16) var vga: texture_3d<f32>;
@group(3) @binding(17) var vga_s: sampler;
@group(3) @binding(18) var vga_out: texture_storage_3d<rgba16float, write>;
@group(3) @binding(19) var vgb: texture_3d<f32>;
@group(3) @binding(20) var vgb_s: sampler;
@group(3) @binding(21) var vgb_out: texture_storage_3d<rgba16float, write>;
const GR: f32 = 1.4;
fn gz() -> f32 { return 1.2 * max(p.thick, 0.6); }
// unwound noise space
fn nq(x: v3) -> v3 { return x * 3.2 * p.nscale + v3(p.seed * 1.7, p.seed * 0.9, 0.); }
fn unshear(w: v3, k: f32) -> v3 { return rotz(w, -k * p.twist * 3.0 * log(1. + length(w.xy) * 4.)); }
// evolution as a z scroll
fn evo_z() -> f32 { return p.cevo * gt() * 0.16 / (3.2 * p.nscale); }
fn guvw(x: v3, d: v3) -> v3 { return v3(clamp(x.xy / (2. * GR) + 0.5, 0.5 / d.xy, 1. - 0.5 / d.xy), x.z / gz() + 0.5); }
fn ga(x: v3) -> v4 { return textureSampleLevel(vga, vga_s, guvw(x, v3(textureDimensions(vga))), 0.); }
fn gb(x: v3) -> v4 { return textureSampleLevel(vgb, vgb_s, guvw(x, v3(textureDimensions(vgb))), 0.); }
// multifractal split: product of octaves [i0, i1) + curve
fn mf_acc(q: v3, i0: i32, i1: i32, fpn: f32) -> f32 {
    var acc = 1.; var freq = pow(3., f32(i0));
    for (var i = i0; i < i1; i++) {
        let fade = clamp(1. / (freq * fpn + 1e-6) - 1., 0., 1.);
        if (fade <= 0.) { break; }
        acc *= 1. + 0.1 * (vnoise(q * freq) * 2. - 1.) * fade; freq *= 3.;
    }
    return acc;
}
fn mf(acc: f32, contrast: f32) -> f32 { return log(1. + pow(max(0., acc), contrast)); }
fn gbake_pos(id: vec3<u32>, d: vec3<u32>) -> v3 { let f = (v3(id) + 0.5) / v3(d); return v3((f.xy - 0.5) * 2. * GR, (f.z - 0.5) * gz()); }
fn bake_ga_at(x: v3) -> v4 { let q = nq(x); return v4(curl2(q * 1.4 + 9.) * 0.35, vnoise(q * 0.9 + 3.1), 0.); }
fn bake_gb_at(x: v3) -> v4 {
    let q = nq(x);
    return v4(mf_acc(q, 0, 2, 0.), mf_acc(q * 2.8 + 21., 0, 1, 0.), mf_acc(q * v3(0.6, 0.6, 1.2) + 30., 0, 2, 0.), mf_acc(q * 1.1 + 60., 0, 2, 0.));
}
// z wraps seamlessly
@compute @workgroup_size(4, 4, 4)
fn bake_ga(@builtin(global_invocation_id) id: u3) {
    let d = textureDimensions(vga_out); if (any(id >= d)) { return; }
    let x = gbake_pos(id, d); var v = bake_ga_at(x);
    let sw = smoothstep(0.35 * gz(), 0.5 * gz(), x.z);
    if (sw > 0.) { v = mix(v, bake_ga_at(x - v3(0., 0., gz())), sw); }
    textureStore(vga_out, id, v);
}
@compute @workgroup_size(4, 4, 4)
fn bake_gb(@builtin(global_invocation_id) id: u3) {
    let d = textureDimensions(vgb_out); if (any(id >= d)) { return; }
    let x = gbake_pos(id, d); var v = bake_gb_at(x);
    let sw = smoothstep(0.35 * gz(), 0.5 * gz(), x.z);
    if (sw > 0.) { v = mix(v, bake_gb_at(x - v3(0., 0., gz())), sw); }
    textureStore(vgb_out, id, v);
}

// bar ovality (NGC 1300)
fn oval_w(w: v3) -> v3 { return v3(w.x / (1. + p.oval * exp(-length(w.xy) / 0.35)), w.y, w.z); }

// centre: bar dust lanes, star-forming ring, nuclear spiral
fn nucleus(w: v3, rho: f32) -> v3 {
    if (rho > 0.45) { return v3(0.); }
    let zh = 0.014 * p.thick; let zn = exp(-w.z * w.z / (zh * zh));
    if (zn < 0.01) { return v3(0.); }
    let n1 = vnoise(w * v3(40., 40., 60.) + p.seed); let n2 = vnoise(w * v3(110., 110., 150.) + 7.);
    let ragged = smoothstep(0.25, 0.8, n1 * 0.65 + n2 * 0.35);
    let Lb = 0.07 * p.core * (1. + p.bar * 1.4) * 3.2;
    // bar dust lanes
    let ph = 2. * (atan2(w.y, w.x) + 0.5) - 2.6 * log(rho / Lb + 1e-3);
    let lane = pow(0.5 + 0.5 * cos(ph), 10.) * smoothstep(Lb * 1.1, Lb * 0.5, rho) * smoothstep(0.03, 0.07, rho)
             * smoothstep(0.35, 0.7, n1) * min(p.bar * 2., 1.2);
    let rr = (rho - 0.075 * p.core) / (0.012 * p.core);
    let ring = exp(-rr * rr);
    let sp = 0.5 + 0.5 * cos(2. * (atan2(w.y, w.x) - 2.4 * log(rho + 0.004)));
    let nsp = pow(sp, 3.) * smoothstep(0.075 * p.core, 0.02, rho) * smoothstep(0.004, 0.015, rho);
    return v3(ring * ragged * 1.3, (lane * 1.6 + ring * 0.5 + nsp * 1.2) * ragged * p.dust, ring * smoothstep(0.5, 0.75, n2) * 1.6) * zn;
}

// w: density wave, wn: noise lookup
fn field1(w: v3, wn: v3) -> F {
    var f: F; f.dens = 0.; f.dust = 0.; f.A = 0.; f.cloud = 0.; f.fine = 0.; f.q = v3(0.); f.ed = v2(0.); f.fl = 0.;
    f.yb = 0.; f.hii = 0.; f.lane = 0.; f.tau = 1e3; f.rot = 0.; f.rot2 = 0.; f.ed2 = v2(0.); f.b2 = 0.;
    let rho = length(w.xy);
    if (rho > 1.35) { return f; }
    let nu = nucleus(w, rho);
    f.dens = nu.x; f.dust = nu.y; f.fine = nu.z; f.A = nu.x;
    let env = exp(-rho / 0.34) * (1. - smoothstep(0.85, 1.3, rho)) * smoothstep(0.02, 0.1, rho);
    // arm frame + swirled eddies
    let xc = unshear(wn, 1. - p.lump) + v3(0., 0., evo_z());
    let va = ga(xc + v3(0., 0., p.swirl * gt() * 0.6 / (1.4 * 3.2 * p.nscale)));
    // relief
    let hb = 0.045 * p.thick * (0.3 + rho) * max(1. + p.relief * (va.z - 0.5) * 1.6, 0.25);
    let yr = w.z / hb;
    let ch = cosh(min(abs(yr) * 1.5, 10.)); let vfall = 1. / (ch * ch);
    if (vfall * env < 0.002) { return f; }
    let ed = va.xy;
    let ar = arm(w, ed.x * 0.5);
    f.A = max(f.A, ar.gas * ar.amp);
    let bq = burst(w, 0.);
    f.yb = ar.amp * young(ar.tau) * bq; f.hii = ar.amp * hii_t(ar.tau) * bq; f.lane = ar.lane * ar.amp; f.tau = ar.tau;
    f.q = nq(xc) + v3(ed, 0.); f.ed = ed;
    let fpn = FP * 3.2 * p.nscale;
    let ede = v3(ed / (3.2 * p.nscale), 0.);
    let vb = gb(xc + ede);
    f.cloud = mf(vb.x * mf_acc(f.q, 2, 5, fpn), 46.);
    f.fine = max(f.fine, mf(vb.y * mf_acc(f.q * 2.8 + 21., 1, 4, fpn * 2.8), 30.));
    let cn = 1. / sqrt(max(p.thick, 0.2));
    // gas-poor bulge
    let hole = smoothstep(0.04 * p.core, 0.2 * p.core, length(w * v3(1., 1., 1.5)));
    // supernova bubbles
    let zone = ar.amp * smoothstep(2., 6., ar.tau) * exp(-ar.tau / 60.) * mix(1., burst(w, 10.), 0.6);
    let bub = max(1. - p.bub * 0.85 * zone * smoothstep(0.5, 0.72, vnoise(f.q * 1.3 + 70.)), 0.);
    f.dens += smoothstep(0.33, 1.05, f.cloud * (0.5 + 0.85 * f.fine)) * (0.2 + 1.4 * f.A) * env * vfall * cn * mix(0.06, 1., hole) * bub;
    // broad dust + shock lane
    let xd = unshear(wn, 0.5 * (1. - p.lump)) + v3(0., 0., evo_z()) + ede;
    let bd = pow(smoothstep(0.3, 0.95, mf(gb(xd).z * mf_acc(nq(xd) * v3(0.6, 0.6, 1.2) + 30., 2, 3, fpn * 0.6), 22.)), p.dhard);
    // thin dust layer + midplane sheet
    let yd = yr / 0.6; let chd = cosh(min(abs(yd) * 1.5, 10.)); let vfd = 1. / (chd * chd);
    let mol = smoothstep(0.75, 1.3, f.cloud * (0.5 + 0.85 * f.fine)) * f.A;
    f.dust += (bd * (0.12 + 0.4 * f.A) + f.lane * (0.4 + 0.6 * bd) * 1.6 + mol * 0.6 + 0.12 * (0.5 + 0.5 * bd))
            * env * vfd * p.dust * cn * mix(0.6, 1., hole) * mix(1., bub, 0.5);
    // dust filaments
    let vf2 = exp(-yr * yr * 1.2);
    if (p.fil > 0. && env * vf2 > 0.02) {
        f.fl = filaments(wn, FP) * env * vf2 * cn * mix(0.4, 1., hole) * (0.4 + 1.2 * ar.amp * max(ar.lane, ar.gas));
        f.dust += f.fl;
    }
    return f;
}
fn fmix(a: F, b: F, t: f32) -> F {
    var f = a;
    f.dens = mix(a.dens, b.dens, t); f.dust = mix(a.dust, b.dust, t); f.A = mix(a.A, b.A, t);
    f.cloud = mix(a.cloud, b.cloud, t); f.fine = mix(a.fine, b.fine, t); f.fl = mix(a.fl, b.fl, t);
    f.yb = mix(a.yb, b.yb, t); f.hii = mix(a.hii, b.hii, t); f.lane = mix(a.lane, b.lane, t); f.tau = mix(a.tau, b.tau, t);
    // shadow taps read both phases
    f.rot2 = b.rot; f.ed2 = b.ed; f.b2 = t;
    return f;
}
fn field(w: v3) -> F {
    if (p.gflow <= 0.) { return field1(w, w); }
    let l = flow_at(length(w.xy));
    var fa = field1(w, rotz(w, l.r0 + l.o0)); fa.rot = l.r0 + l.o0;
    if (l.b > 0.99) { return fa; }
    var fb = field1(w, rotz(w, l.r1 + l.o1)); fb.rot = l.r1 + l.o1;
    if (l.b < 0.01) { return fb; }
    return fmix(fb, fa, l.b);
}
fn filaments(w: v3, fp: f32) -> f32 {
    let k = 3.2 * p.nscale * 10. * p.fsc;
    var q = w * k + v3(p.seed * 2.1, p.seed * 0.4, 11. + p.cevo * gt() * 0.48);
    let fpn = fp * k;
    q += (v3(vnoise(q * 0.22 + 1.), vnoise(q * 0.22 + 7.), vnoise(q * 0.22 + 13.)) - 0.5) * 5.;
    let brk = smoothstep(0.38, 0.68, vnoise(q * 0.07 + 21.));
    if (brk <= 0.) { return 0.; }
    var r = 0.; var amp = 0.6; var fr = 1.; var wgt = 1.;
    for (var i = 0; i < 3; i++) {
        let fade = clamp(1. / (fr * fpn + 1e-6) - 1., 0., 1.);
        if (fade <= 0.) { break; }
        let n = 1. - abs(vnoise(q * fr) * 2. - 1.);
        let rn = n * n * wgt;
        r += rn * amp * fade; wgt = clamp(rn * 1.6, 0., 1.); fr *= 2.13; amp *= 0.5;
    }
    return pow(clamp(r, 0., 1.), 5.) * brk * 9. * p.fil * p.dust;
}

// cheap medium octaves for short shadow rays
fn cloud_tap1(w: v3, rot: f32, ed: v2, fpn: f32) -> f32 {
    let x = unshear(rotz(w, rot), 1. - p.lump) + v3(0., 0., evo_z()) + v3(ed / (3.2 * p.nscale), 0.);
    return mf(gb(x).x * mf_acc(nq(x), 2, 3, fpn), 46.);
}
fn cloud_tap(w: v3, f: F) -> f32 {
    let fpn = FP * 3.2 * p.nscale;
    var v = cloud_tap1(w, f.rot, f.ed, fpn);
    if (f.b2 > 0.) { v = mix(v, cloud_tap1(w, f.rot2, f.ed2, fpn), f.b2); }
    return smoothstep(0.33, 1.05, v * (0.5 + 0.85 * f.fine));
}
fn ext_of(f: F) -> f32 { return (f.dens * 9.6 + f.dust * 22.) * 1.6; }

// P0 (quarter res): core shadow map, optical depth at four radii
const SHR = v4(0.12, 0.3, 0.6, 1.2);
const SHE: f32 = 0.7;
@compute @workgroup_size(16,16,1)
fn shadow(@builtin(global_invocation_id) id: u3) {
    let d = vec2<u32>(textureDimensions(out)); if (id.x >= d.x || id.y >= d.y) { return; }
    let uv = (v2(id.xy) + 0.5) / v2(d);
    let th = uv.x * tau - pi;
    let u = uv.y * 2. - 1.; let ph = sign(u) * u * u * SHE;
    let dir = v3(cos(ph) * cos(th), cos(ph) * sin(th), sin(ph));
    let N = 64;
    let k = log(1.3 / 0.02) / f32(N);
    var tau_ = 0.; var o = v4(0.); var r0 = 0.02;
    for (var i = 0; i < N; i++) {
        let r1 = 0.02 * exp(f32(i + 1) * k);
        let rm = 0.5 * (r0 + r1);
        FP = rm * 0.02;
        tau_ += ext_of(field(oval_w(dir * rm))) * (r1 - r0);
        if (r1 <= SHR.x) { o.x = tau_; }
        if (r1 <= SHR.y) { o.y = tau_; }
        if (r1 <= SHR.z) { o.z = tau_; }
        o.w = tau_;
        r0 = r1;
    }
    textureStore(out, vec2<i32>(id.xy), o);
}
// optical depth core -> w
fn core_tau(w: v3) -> f32 {
    let rho = length(w.xy); let r = length(w);
    let ph = atan2(w.z, rho);
    let u = sign(ph) * sqrt(min(abs(ph) / SHE, 1.));
    let dm = v2(textureDimensions(t0));
    let uv = v2(atan2(w.y, w.x) / tau + 0.5, clamp(u * 0.5 + 0.5, 0.5 / dm.y, 1. - 0.5 / dm.y));
    let t = textureSampleLevel(t0, s0, uv, 0.);
    if (r < SHR.x) { return t.x * r / SHR.x; }
    if (r < SHR.y) { return mix(t.x, t.y, (r - SHR.x) / (SHR.y - SHR.x)); }
    if (r < SHR.z) { return mix(t.y, t.z, (r - SHR.y) / (SHR.z - SHR.y)); }
    return mix(t.z, t.w, clamp((r - SHR.z) / (SHR.w - SHR.z), 0., 1.));
}

// differential rotation: rings turning rigidly
const RW: f32 = 0.1;
var<private> RING: f32 = 0.;
fn ring_sb(i: f32) -> f32 { return select(-(omega((i + 0.5) * RW) - omp()) * tmyr() * p.gflow, 0., p.gflow <= 0.); }
fn ring_of(w: v3) -> f32 { return floor(length(w.xy) / RW); }

// young clusters: light, colour, ionised glow
const G2: f32 = 14.;
const LR: f32 = 1. / G2;
struct Cl { p: v4, c: v3, e: v3, s: v4 };
// 3D DDA star walk
struct Walk { cell: v3, tmx: v3, tdel: v3, stp: v3, inv: v3, sg: f32, lvl: f32 };
fn walk_at(o: v3, dr: v3, t: f32, sg_: f32, lvl: f32) -> Walk {
    var k: Walk;
    let ds = select(dr, select(v3(-1e-6), v3(1e-6), dr >= v3(0.)), abs(dr) < v3(1e-6));
    k.inv = 1. / ds; k.stp = select(v3(-1.), v3(1.), ds >= v3(0.)); k.tdel = abs(k.inv) / sg_; k.sg = sg_; k.lvl = lvl;
    k.cell = floor((o + dr * t) * sg_); k.tmx = ((k.cell + max(k.stp, v3(0.))) / sg_ - o) * k.inv;
    return k;
}
// stars passed by t1
fn walk_run(k: ptr<function, Walk>, o: v3, dr: v3, pa: f32, t: f32, t1: f32) -> v3 {
    // restart after a long jump
    if (min((*k).tmx.x, min((*k).tmx.y, (*k).tmx.z)) < t - 0.06) { *k = walk_at(o, dr, t, (*k).sg, (*k).lvl); }
    var c = v3(0.);
    for (var n = 0; n < 16; n++) {
        let tm = min((*k).tmx.x, min((*k).tmx.y, (*k).tmx.z));
        if (tm > t1) { break; }
        c += disk_star((*k).cell, (*k).sg, (*k).lvl, o, dr, pa);
        if ((*k).tmx.x <= (*k).tmx.y && (*k).tmx.x <= (*k).tmx.z) { (*k).cell.x += (*k).stp.x; (*k).tmx.x += (*k).tdel.x; }
        else if ((*k).tmx.y <= (*k).tmx.z) { (*k).cell.y += (*k).stp.y; (*k).tmx.y += (*k).tdel.y; }
        else { (*k).cell.z += (*k).stp.z; (*k).tmx.z += (*k).tdel.z; }
    }
    return c;
}

// stars of ring ri over [t, t1]
fn ring_stars(wa: ptr<function, Walk>, wb: ptr<function, Walk>, cur: ptr<function, f32>, ri: f32, o: v3, dr: v3, pa: f32, t: f32, t1: f32, ka: f32, fw: f32) -> v3 {
    SB = ring_sb(ri); RING = ri;
    let os = rotz(o, -SB); let ds = rotz(dr, -SB);
    if (*cur != ri) { *wa = walk_at(os, ds, t, 48. * exp2(ka), ka); *wb = walk_at(os, ds, t, 96. * exp2(ka), ka + 1.); *cur = ri; }
    var c = walk_run(wa, os, ds, pa, t, t1) * (1. - fw);
    if (fw > 0.01) { c += walk_run(wb, os, ds, pa, t, t1) * fw; }
    return c;
}
// optional twinkle
fn twinkle(s: ptr<function, u32>) -> f32 {
    let a = hn(s); let f = hn(s); let ph = hn(s);
    return select(1., mix(1., 0.55 + 0.45 * sin(u_t.time * mix(0.25, 0.8, f) + ph * tau), p.twk), a < 0.35);
}
// star temperature by region
fn star_T(h: f32, h2: f32, yg: f32) -> f32 {
    if (h < 0.08 + 0.45 * yg) { return mix(11000., 30000., h2); }
    if (h < 0.22 + 0.45 * yg) { return mix(7000., 10000., h2); }
    if (h < 0.48 + 0.3 * yg) { return mix(5200., 6600., h2); }
    if (h < 0.78 + 0.15 * yg) { return mix(3900., 5200., h2); }
    return mix(2700., 3800., h2);
}
// c: cell, w: ring-frame point
fn cluster(c: v2, w: v3, near: bool, sb: f32, ri: f32) -> Cl {
    var o: Cl; o.p = v4(0.); o.c = v3(0.); o.e = v3(0.); o.s = v4(0.);
    let ci = vec2<u32>(vec2<i32>(c) + 512);
    var s = hu((ci.x * 73856093u) ^ (ci.y * 19349663u) ^ (u32(p.seed) * 2654435761u) ^ 0x9e3779b9u);
    let pc = (c + 0.06 + 0.88 * v2(hn(&s), hn(&s))) / G2;
    let rho = length(pc);
    let z = (hn(&s) - 0.5) * 0.03 * p.thick * (0.5 + rho);
    let dv = w - v3(pc, z);
    if (near && dot(dv, dv) >= LR * LR) { return o; }
    if (p.gflow > 0. && floor(rho / RW) != ri) { return o; }
    let pp = rotz(v3(pc, z), sb);
    // ageing after the shock
    let ar = arm(pp, 0.);
    let yb = ar.amp * young(ar.tau);
    let bq = burst(pp, 0.);
    // only in dense clouds
    let gm = smoothstep(0.35, 0.7, vnoise(shear_q(pp, 1. - p.lump)));
    if (hn(&s) > gm * (yb + 0.03) * smoothstep(0.08, 0.2, rho) * (1. - smoothstep(0.9, 1.2, rho)) * 0.85) { return o; }
    o.p = v4(pp, 0.22 * pow(1. - hn(&s) * 0.99, -0.6) * p.clus * mix(0.35, 1., exp(-ar.tau / 25.)) * mix(1., bq, exp(-ar.tau / 20.)));
    // colour by age
    let ag = max(ar.tau + hn(&s) * 12., 1.5);
    let T = clamp(30000. * pow(ag / 3., -0.35), 6500., 30000.) * mix(0.85, 1.15, hn(&s));
    let frsg = smoothstep(7., 12., ag) * (1. - smoothstep(40., 80., ag)) * (0.3 + 0.7 * hn(&s));
    // twinkle
    let dr = sin(u_t.time * mix(0.04, 0.12, hn(&s)) + hn(&s) * tau) * p.twk;
    VL = vlos(pp);
    o.c = vivid(mix(bb(T * (1. + 0.3 * dr) / zfac()), bb(3700. / zfac()), frsg * 0.7), p.spc * 1.3);
    o.p.w *= mix(1., 0.8 + 0.2 * sin(u_t.time * mix(0.1, 0.3, hn(&s)) + hn(&s) * tau), p.twk);
    if (hn(&s) < 0.6 && T > 9000. && ar.tau < 15.) {
        let ox = clamp(smoothstep(18000., 28000., T) + 0.35 * dr, 0., 1.) * min(p.ion, 1.);
        o.e = lc(mix(v3(1.0, 0.22, 0.45), v3(0.3, 0.9, 0.85), ox), ox) * p.nebula * exp(-ar.tau / 8.) * bq;
    }
    // supernovae: flash, then a glowing shell
    if (p.sn > 0. && ar.tau < 40. && hn(&s) < 0.3) {
        let per = mix(120., 300., hn(&s)) / max(p.snr, 0.05);
        let ts = fract(gt() * 4. / per + hn(&s)) * per;
        let bl = mix(1., clamp(burst(pp, 8.) - 0.5, 0., 1.), p.brst);
        o.s = v4(exp(-ts / 1.5) * smoothstep(0., 0.5, ts), 0.003 + 0.012 * sqrt(ts), exp(-ts / 5.) * smoothstep(0., 1.2, ts), 0.) * p.sn * bl;
    }
    return o;
}

// diffuse nebula density + hue
fn neb_d(x: v3, fpn: f32) -> v2 {
    var q = nq(x) * 1.1 + 60.;
    let sw = v3(0., 0., p.swirl * gt() * 0.2);
    let wp = (v3(vnoise(q * 0.35 + 3. + sw), vnoise(q * 0.35 + 9. + sw), 0.) - 0.5) * 2.2;
    q += wp;
    let body = mf(gb(x + wp / (1.1 * 3.2 * p.nscale)).w * mf_acc(q, 2, 5, fpn), 14.);
    var rg = 0.; var amp = 0.6; var fr = 3.1;
    for (var i = 0; i < 2; i++) {
        let fade = clamp(1. / (fr * fpn + 1e-6) - 1., 0., 1.);
        let r = 1. - abs(vnoise(q * fr + 17.) * 2. - 1.);
        rg += r * r * amp * fade; fr *= 2.3; amp *= 0.5;
    }
    return v2(pow(smoothstep(0.3, 1.25, body * (0.6 + 0.9 * rg)), 1.5), vnoise(q * 0.12 + 40.));
}

// emission + extinction, lit by the core through the shadow map
struct G { em: v3, ext: v3 };
fn gsample(w0: v3, rd: v3, Ll: v3, Le: v3) -> G {
    var g: G; g.em = v3(0.); g.ext = v3(0.);
    let w = oval_w(w0);
    VL = vlos(w);
    let DT = zcont(zfac());
    var ln = v3(0.);
    let rho = length(w.xy);
    if (rho > 1.35) { return g; }
    let jw = clamp(p.jwst, 0., 1.);
    // unresolved starlight
    let a0 = arm(w, 0.);
    let bq = burst(w, 0.);
    let As = a0.gas * a0.amp; let yb0 = a0.amp * young(a0.tau) * bq;
    let zs = 0.05 * p.thick * (0.6 + rho);
    let sdisk = exp(-rho / 0.27) * (1. - smoothstep(0.9, 1.35, rho)) * exp(-abs(w.z) / zs);
    g.em = mix(v3(1.0, 0.78, 0.5), v3(0.55, 0.7, 1.0), clamp(yb0 * 0.9 * p.young * (1. - jw), 0., 1.)) * sdisk * (0.75 + 0.35 * As) * 0.8 / max(p.thick, 0.2);
    let f = field(w);
    let r3 = length(w);
    let cs = dot(rd, w / max(r3, 1e-4));
    // core light, reddened
    let Lc = core_L(r3);
    let tcore = core_tau(w) * 0.6 * p.shd * pow(v3(1., 1.33, 1.76), v3(p.redd));
    let ph0 = mix(1., hg(cs, 0.65) * 12.566, 0.6);
    // shafts
    g.em += (Lc * ph0 * exp(-tcore) * v3(0.7, 0.8, 1.0) + Ll * 0.6) * sdisk * 0.05;
    // diffuse ionised gas around young stars
    var n2 = 0.;
    if (p.neb2 > 0. && sdisk > 0.002) {
        let fpn = FP * 3.2 * p.nscale * 1.1;
        var nd = v2(0.);
        let ez = v3(0., 0., evo_z());
        if (p.gflow <= 0.) { nd = neb_d(unshear(w, 0.5 * (1. - p.lump)) + ez, fpn); }
        else {
            let l = flow_at(rho);
            let qa = unshear(rotz(w, l.r0 + l.o0), 0.5 * (1. - p.lump)) + ez;
            let qb = unshear(rotz(w, l.r1 + l.o1), 0.5 * (1. - p.lump)) + ez;
            if (l.b > 0.99) { nd = neb_d(qa, fpn); } else if (l.b < 0.01) { nd = neb_d(qb, fpn); }
            else { nd = mix(neb_d(qb, fpn), neb_d(qa, fpn), l.b); }
        }
        let d = nd.x;
        n2 = d * exp(-rho / 0.45) * (1. - smoothstep(0.9, 1.3, rho)) * exp(-abs(w.z) / (zs * 1.1)) * p.neb2;
        if (n2 > 1e-4) {
            let ion = 0.12 + 1.4 * a0.amp * exp(-a0.tau / 15.) * smoothstep(0.3, 2., a0.tau) * bq * p.young + dot(Ll, LUMA) * 2.5;
            let h = p.nhue * 0.25 + 0.3 * nd.y;
            let sd = smoothstep(0.2, 0.8, d);
            // oxygen thin, hydrogen dense
            let tint = lc(mix(sky_tint(h + 0.75), sky_tint(h + 0.5), sd), 1. - sd + p.nhue * 0.25 + (nd.y - 0.5) * 0.6);
            ln += tint * ion * 1.4 * (1. + jw) * n2;
            g.em += Ll * 0.3 * n2;
            g.ext = v3(n2 * 0.4);
        }
    }
    if (f.dens + f.dust < 0.002) { g.em = g.em * DT + ln; return g; }
    let A = f.A; let fine = f.fine;
    // gas temperature
    let T = mix(4600., 16000., clamp(f.yb * p.young * 1.3, 0., 1.)) * mix(0.85, 1.25, smoothstep(0.3, 1.2, f.cloud));
    var S = vivid(bb(T), p.spc) * (0.3 + (0.9 * A + 1.0 * f.yb) * p.young);
    // slow colour fields
    let qs = shear_q(w, 1. - p.lump);
    let hv = vnoise(qs * 0.35 + 50.);
    S *= mix(v3(0.8, 1.0, 1.15), v3(1.15, 0.85, 1.1), smoothstep(0.25, 0.75, hv));
    S = mix(S, v3(1.2, 0.85, 0.45) * max(S.b, 0.3), smoothstep(0.62, 0.85, vnoise(qs * 0.6 + 77.)) * 0.6);
    // star-forming knots
    let kn = f.hii * smoothstep(0.7, 1.3, fine) * p.nebula * (1. + jw) * 1.6;
    S += v3(1.0, 0.8, 0.85) * smoothstep(1.5, 2.0, fine) * kn * 2.;
    var Sl = lc(v3(1.0, 0.2, 0.42), 0.) * 1.8 * kn + lc(v3(0.12, 0.85, 0.78), 1.) * smoothstep(1.25, 1.7, fine) * kn * 0.9 * p.ion;
    S += v3(0.5, 0.45, 1.0) * smoothstep(0.5, 1.1, rho) * A * 0.5 * p.ion;
    // infrared
    let sk = mix(0.45, 1.6, exp(-rho / 0.25)) * mix(v3(1.), v3(1.0, 0.72, 0.45), jw);
    S *= sk; Sl *= sk;
    let df = f.dust / (f.dens + f.dust);
    // short shadow rays
    var tl = 0.;
    if (f.dens > 0.01) {
        let ld = -w / max(r3, 1e-4);
        tl = (cloud_tap(w + ld * 0.01, f) * 0.01 + cloud_tap(w + ld * 0.028, f) * 0.018) * (0.2 + 1.4 * f.A) * 110. * p.shd;
    }
    // multiple scattering (Wrenninge)
    var lit = v3(0.); var ai = 1.; var bi = 1.; var gi = 0.65; var ws = 0.;
    for (var i = 0; i < 3; i++) {
        lit += ai * mix(1., hg(cs, gi) * 12.566, 0.6) * exp(-(tcore + tl) * bi);
        ws += ai; ai *= 0.7 * p.msc; bi *= 0.3; gi *= 0.5;
    }
    lit *= Lc / ws;
    let occ = (exp(-tl) + 0.7 * p.msc * exp(-0.3 * tl)) / (1. + 0.7 * p.msc);
    // gas glow takes the shading
    let sh = f.dens * 7. * (1. - df * 0.7) * mix(1., occ, 0.85);
    g.em += S * sh; ln += Sl * sh;
    // core light scattering
    g.em += lit * (f.dens * v3(0.75, 0.85, 1.0) * 1.6 + f.dust * v3(1.0, 0.72, 0.45) * 0.4);
    // silver linings
    g.em += Lc * exp(-tcore) * hg(cs, 0.8) * 12.566 * f.dust * v3(1.0, 0.7, 0.42) * 0.04 * p.fwd;
    // disk starlight on clouds
    let amb = sdisk * 1.6 + n2 * 0.3 + 0.5 / (1. + r3 * r3 / (0.02 * p.core * p.core)) + 0.03;
    g.em += (f.dens * v3(0.7, 0.78, 0.9) * 0.6 + f.dust * v3(0.62, 0.4, 0.25) * 0.45) * amb * mix(1., occ, 0.5) * p.dalb;
    // cluster light
    g.em += Ll * (f.dens * 0.8 + f.dust * 0.35 * v3(1.0, 0.8, 0.6));
    ln += Le * f.dens * 2.5 * (1. + jw);
    // infrared dust glow
    g.em += f.dust * v3(1.0, 0.42, 0.14) * (0.15 + dot(lit, LUMA) * 0.05 + sdisk * 0.5) * jw * 2.;
    // reddening
    g.ext += f.dens * 9.6 * v3(1., 1.1, 1.25) + f.dust * 22. * pow(v3(1., 1.33, 1.76), v3(p.redd)) * (1. - 0.75 * jw);
    g.em = g.em * DT + ln;
    return g;
}

// resolved star in DDA cell c (lvl: detail level)
fn disk_star(c: v3, sg_: f32, lvl: f32, o: v3, d: v3, pa: f32) -> v3 {
    let ci = vec3<u32>(vec3<i32>(c) + 8192);
    var s = hu((ci.x * 73856093u) ^ (ci.y * 19349663u) ^ (ci.z * 83492791u) ^ (u32(p.seed) * 2654435761u) ^ (u32(lvl) * 0x9e3779b9u));
    // position in cell
    let pc = (c + 0.06 + 0.88 * v3(hn(&s), hn(&s), hn(&s))) / sg_;
    if (p.gflow > 0. && floor(length(pc.xy) / RW) != RING) { return v3(0.); }
    // differential rotation
    let pm = rotz(pc, SB);
    let pl = exp2(-lvl);
    let rho = length(pm.xy);
    let zs = 0.05 * p.thick * (0.5 + rho);
    let ar = arm(pm, 0.);
    let yb = ar.amp * young(ar.tau);
    let disk = exp(-rho / 0.5) * (1. - smoothstep(0.95, 1.3, rho)) * exp(-pc.z * pc.z / (zs * zs));
    let bd = length(pm / v3(1. + p.bar * 1.4, 1., 0.6)) / (0.11 * p.core);
    let bul = 2. * p.core / ((1. + bd * bd) * (1. + bd * bd));
    // young stars in dense gas
    var cl = 0.;
    if (yb > 0.003) {
        let xs = unshear(pc, 1. - p.lump) + v3(0., 0., evo_z());
        cl = smoothstep(0.35, 0.95, mf(gb(xs).x * mf_acc(nq(xs), 2, 3, FP * 3.2 * p.nscale), 46.) * 1.2);
    }
    let tc = dot(pc - o, d);
    if (tc <= 0.) { return v3(0.); }
    let dd = length(o + d * tc - pc);
    let sg = tc * pa * 0.75;
    // HII clumps
    if (p.hii > 0.) {
        let ch = smoothstep(0.6, 0.85, vnoise(pc * 16. + p.seed + 5.));
        let ph = ar.amp * hii_t(ar.tau) * 1.5 * ch * exp(-pc.z * pc.z / (zs * zs * 0.3)) * smoothstep(0.1, 0.25, rho) * (1. - smoothstep(0.9, 1.2, rho));
        let u = hn(&s); let P = ph * 0.9 * p.hii * pl;
        if (u < P) {
            let pf = exp(-dd * dd / (sg * sg * 1.6)) * sfade(u, P);
            VL = vlos(pm);
            return lc(mix(v3(1.0, 0.08, 0.2), v3(1.0, 0.3, 0.12), hn(&s)), 0.) * 0.12 * pow(1. - hn(&s) * 0.9, -0.6) * pf * twinkle(&s) * (1. + p.jwst) * burst(pm, 0.);
        }
    }
    // brightest fraction resolved
    let u = hn(&s); let P = (disk * (0.2 * (1. + 0.5 * ar.gas * ar.amp) + 5. * cl * yb) * 1.1 + bul * 0.12) * pl * p.res;
    if (u > P) { return v3(0.); }
    let pf = exp(-dd * dd / (sg * sg)) * sfade(u, P);
    if (pf < 0.002) { return v3(0.); }
    // blue giants behind the shock
    let yg = min(yb * 5. * cl * p.young * disk / (disk * 0.12 + 5. * cl * yb * disk + bul * 0.12 + 1e-4), 1.);
    let T = star_T(hn(&s), hn(&s), yg);
    var L = 0.05 * pow(0.75, lvl) * pow(1. - (1. - p.res * hn(&s)) * 0.995, -0.65) * twinkle(&s) * mix(0.6, 1.5, yg);
    if (yg > 0.01) { L *= mix(1., burst(pm, 0.), yg); }
    VL = vlos(pm);
    var sc = vivid(bb(T / zfac()), p.spc * 1.4) * L;
    // lifetimes, some end as supernovae
    if (T > 10000.) {
        let ex = ar.tau - 10. * pow(25000. / T, 2.5);
        if (ex > 0.) {
            if (ex > 1.5) { return v3(0.); }
            let fl = hn(&s) < 0.12 * p.snr * mix(1., clamp(burst(pm, 8.) - 0.5, 0., 1.), p.brst);
            if (p.gflow > 0. && fl && omega(rho) > omp()) { sc = v3(0.85, 0.9, 1.) * 2. * exp(-ex / 0.35) * smoothstep(0., 0.05, ex); }
            else { sc *= 1. - smoothstep(0., 1.5, ex); }
        }
    }
    return sc * pf * p.stars;
}
// fade drifting stars
fn sfade(u: f32, P: f32) -> f32 { let k = 0.3 * min(p.gflow, 1.); return select(1., clamp((P - u) / (k * P + 1e-6), 0., 1.), k > 0.); }

// foreground stars
fn suns(sr: v2, B: m3) -> v3 {
    var col = v3(0.);
    if (p.suns <= 0.) { return col; }
    for (var i = 0u; i < 96u; i++) {
        var s = hu(i * 2654435761u + 977u + u32(p.seed) * 131u);
        let ct = hn(&s) * 2. - 1.; let ph = hn(&s) * tau; let st = sqrt(1. - ct * ct);
        let sd = v3(st * cos(ph), st * sin(ph), ct);
        let q = v3(dot(sd, B[0]), dot(sd, B[1]), dot(sd, B[2]));
        if (q.z < 0.5) { continue; }
        let dp = (sr - q.xy / q.z * 1.7 * p.zm) * 540.;
        let r2 = dot(dp, dp);
        if (r2 > 400.) { continue; }
        let L = 0.3 * pow(1. - hn(&s) * 0.98, -0.8) * p.suns;
        col += vivid(bb(mix(3000., 16000., hn(&s) * hn(&s))), 1.2) * L * (exp(-r2 / 1.2) + 0.03 / (1. + r2 / 6.));
    }
    return col;
}

// bulge: multi-Gaussian expansion, integrated with erf
fn erf_(x: f32) -> f32 { let x2 = x * x; return sign(x) * sqrt(1. - exp(-x2 * (1.2732 + 0.147 * x2) / (1. + 0.147 * x2))); }
// unit Gaussian
fn gauss_seg(o: v3, d: v3, sg: f32, S: v3, t0: f32, t1: f32) -> f32 {
    let P = o / (S * sg); let D = d / (S * sg);
    let B = dot(D, D); let tc = -dot(P, D) / B;
    let h2 = max(dot(P, P) - dot(P, D) * dot(P, D) / B, 0.);
    if (h2 > 18.) { return 0.; }
    let sb = sqrt(B * 0.5);
    let I = exp(-0.5 * h2) * 0.5 * sqrt(6.2832 / B) * (erf_((t1 - tc) * sb) - erf_((t0 - tc) * sb));
    return I / (2.5066 * sg * S.z);
}
// nucleus, inner bulge, bar, envelope
fn bulge(o: v3, d: v3, t0: f32, t1: f32) -> v3 {
    let c = max(p.core, 0.05);
    var col = v3(1.0, 0.93, 0.84) * gauss_seg(o, d, 0.012 * c, v3(1., 1., 0.85), t0, t1) * 0.35;
    col += v3(1.0, 0.82, 0.6) * gauss_seg(o, d, 0.04 * c, v3(1., 1., 0.75), t0, t1) * 0.22;
    col += v3(1.0, 0.72, 0.45) * gauss_seg(o, d, 0.05 * c, v3(1. + p.bar * 2.2, 1., 0.5), t0, t1) * 0.16 * p.bar;
    col += v3(1.0, 0.6, 0.32) * gauss_seg(o, d, 0.16 * c, v3(1., 1., 0.55), t0, t1) * 0.2;
    return col * core_tint() * zcont(1. + p.z) * mix(v3(1.), v3(1.0, 0.85, 0.65), clamp(p.jwst, 0., 1.)) * p.br;
}
// globular clusters (Plummer stars), dimmed behind the disk
fn globulars(o: v3, d: v3, depth: f32, trf: v3, pa: f32) -> v3 {
    var col = v3(0.);
    if (p.glob <= 0.) { return col; }
    for (var i = 0u; i < 18u; i++) {
        var s = hu(i * 747796405u + 12345u + u32(p.seed) * 977u);
        let ct = hn(&s) * 2. - 1.; let ph = hn(&s) * tau; let st = sqrt(1. - ct * ct);
        let c = v3(st * cos(ph), st * sin(ph), ct * 0.8) * (0.3 + 1.1 * pow(hn(&s), 1.5)) * p.size;
        let tc = dot(c - o, d);
        if (tc <= 0.) { continue; }
        let sg = mix(0.004, 0.009, hn(&s));
        let dv = o + d * tc - c; let d2 = dot(dv, dv);
        let px = tc * pa;
        let ext = 5. * sg + 4. * px;
        if (d2 > ext * ext) { continue; }
        let tint = mix(v3(1.0, 0.86, 0.68), v3(0.92, 0.92, 1.0), hn(&s));
        let L = mix(0.4, 1.0, hn(&s)) * p.glob;
        // King-like ball, at least a pixel wide
        let rw = smoothstep(3., 10., sg / max(px, 1e-6));
        let a1 = sg * sg * 0.12; let b1 = a1 + px * px * 0.6;
        let a2 = sg * sg; let b2 = a2 + px * px * 0.6;
        var e = tint * L * (0.6 * exp(-d2 / b1) * a1 / b1 + 0.1 * exp(-d2 / b2) * a2 / b2 + 0.015 * a2 / (a2 + d2)) * (1. - 0.5 * rw);
        if (rw > 0.) {
            let sp = px * 0.75;
            for (var k = 0u; k < 48u; k++) {
                let u = v3(hn(&s), hn(&s), hn(&s));
                // Plummer radius
                let r = sg * 1.2 / sqrt(max(pow(u.x * 0.97 + 0.01, -0.6667) - 1., 1e-3));
                let z = u.y * 2. - 1.; let a = u.z * tau; let sz = sqrt(1. - z * z);
                let ps = c + v3(sz * cos(a), sz * sin(a), z) * r;
                let q = o + d * dot(ps - o, d) - ps;
                let bT = star_T(hn(&s) * 0.85 + 0.15, hn(&s), 0.);
                e += vivid(bb(bT), 1.2) * 0.15 * pow(1. - hn(&s) * 0.9, -0.6) * exp(-dot(q, q) / (sp * sp)) * L * rw;
            }
        }
        if (tc > depth) { e *= trf; }
        col += e;
    }
    return col * zcont(1. + p.z) * p.br;
}
// galactic wind (M82-like), lit by the core
fn wind_at(w: v3, oct: i32) -> v2 {
    let az = abs(w.z); let H = 0.9 * p.wlen;
    let rw = 0.05 * p.core + az * tan(p.wopen);
    let x0 = length(w.xy) / rw;
    if (az > H || x0 > 1.6) { return v2(0.); }
    // cloud noise along the cone
    var q = v3(w.xy / rw * 1.1, (log(az + 0.03) - tmyr() * 0.02) * 2.4) + select(0., 31., w.z < 0.) + v3(p.seed * 1.7, p.seed * 0.9, 0.);
    q += (v3(vnoise(q * 0.5 + 3.), vnoise(q * 0.5 + 11.), 0.) - 0.5) * (0.8 + 3.2 * az / H);
    // rough edge, bright walls
    let x = x0 * (1. + 0.6 * (vnoise(q * 0.3 + 5.) - 0.5));
    let shell = smoothstep(1.1, 0.55, x) * (0.15 + 0.85 * smoothstep(0.2, 0.85, x));
    if (shell <= 0.) { return v2(0.); }
    let fpn = FP * max(1.1 / rw, 2.4 / (az + 0.03));
    let cloud = cloud_noise(q, oct, 46., fpn);
    var fine = 1.;
    if (oct > 3) { fine = cloud_noise(q * 2.8 + 21., 3, 30., fpn * 2.8); }
    let n = cloud * (0.5 + 0.85 * fine);
    let fall = exp(-az / (0.22 * H)) * (1. - smoothstep(0.55 * H, H, az)) * smoothstep(0., 0.02, az);
    let g = smoothstep(0.33, 1.05, n) * shell * fall * 0.35 * p.wind;
    let d = smoothstep(0.75, 1.3, n) * shell * exp(-az / (0.1 * H)) * 0.12 * p.wind * p.dust;
    return v2(g, d);
}
// inside the wind's cylinder
fn wind_span(o: v3, dr: v3) -> v2 {
    let H = 0.9 * p.wlen; let Rw = (0.03 * p.core + H * tan(p.wopen)) * 1.3;
    var ta = 0.; var tb = 1e4;
    if (abs(dr.z) > 1e-6) { let a = (-H - o.z) / dr.z; let b = (H - o.z) / dr.z; ta = max(ta, min(a, b)); tb = min(tb, max(a, b)); }
    else if (abs(o.z) > H) { return v2(1., 0.); }
    let A = dot(dr.xy, dr.xy); let B = dot(o.xy, dr.xy); let C = dot(o.xy, o.xy) - Rw * Rw;
    if (A > 1e-8) { let h = B * B - A * C; if (h < 0.) { return v2(1., 0.); } ta = max(ta, (-B - sqrt(h)) / A); tb = min(tb, (-B + sqrt(h)) / A); }
    else if (C > 0.) { return v2(1., 0.); }
    return v2(ta, tb);
}
// wind lighting, same as the disk's clouds
fn wind_seg(o: v3, dr: v3, ta: f32, tb: f32, dith: f32, pa: f32, tr: ptr<function, v3>, dw: ptr<function, f32>, dsum: ptr<function, f32>) -> v3 {
    var c = v3(0.);
    if (p.wind <= 0. || tb <= ta) { return c; }
    let N = 64; let dt = (tb - ta) / f32(N);
    let ext = pow(v3(1., 1.33, 1.76), v3(p.redd));
    let jw = clamp(p.jwst, 0., 1.);
    for (var i = 0; i < N; i++) {
        if (max((*tr).r, max((*tr).g, (*tr).b)) < 0.01) { break; }
        let t = ta + (f32(i) + dith) * dt;
        let w = o + dr * t;
        FP = t * pa;
        let d = wind_at(w, 5);
        if (d.x + d.y < 1e-4) { continue; }
        let r3 = max(length(w), 1e-4); let ld = -w / r3;
        let Lc = core_L(r3);
        let tcore = core_tau(w) * 0.6 * p.shd * ext;
        let s1 = wind_at(w + ld * 0.015, 3); let s2 = wind_at(w + ld * 0.045, 3);
        let tl = ((s1.x * 9.6 + s1.y * 22.) * 0.015 + (s2.x * 9.6 + s2.y * 22.) * 0.03) * 1.6 * p.shd;
        let cs = dot(dr, -ld);
        var lit = v3(0.); var ai = 1.; var bi = 1.; var gi = 0.65; var ws = 0.;
        for (var k = 0; k < 3; k++) {
            lit += ai * mix(1., hg(cs, gi) * 12.566, 0.6) * exp(-(tcore + tl) * bi);
            ws += ai; ai *= 0.7 * p.msc; bi *= 0.3; gi *= 0.5;
        }
        lit *= Lc / ws;
        let occ = (exp(-tl) + 0.7 * p.msc * exp(-0.3 * tl)) / (1. + 0.7 * p.msc);
        VL = clamp(dot(-ld, RD) * 1.5, -1., 1.);
        var em = lit * (d.x * v3(0.75, 0.85, 1.0) * 1.6 + d.y * v3(1.0, 0.72, 0.45) * 0.4);
        em += Lc * exp(-tcore) * hg(cs, 0.8) * 12.566 * d.y * v3(1.0, 0.7, 0.42) * 0.04 * p.fwd;
        let rho = length(w.xy);
        let amb = exp(-rho / 0.27) * 0.6 / (1. + w.z * w.z / 0.02) + 0.5 / (1. + r3 * r3 / (0.02 * p.core * p.core)) + 0.03;
        em += (d.x * v3(0.7, 0.78, 0.9) * 0.6 + d.y * v3(0.62, 0.4, 0.25) * 0.45) * amb * mix(1., occ, 0.5) * p.dalb;
        em = em * zcont(zfac()) + lc(v3(1.0, 0.3, 0.35), 0.05) * d.x * dot(lit, LUMA) * 0.3 * p.ion * (1. + jw);
        let ce = em * dt * 1.6 * (*tr) * p.br;
        c += ce; let l = dot(ce, LUMA); *dw += l; *dsum += l * t;
        *tr *= exp(-(d.x * 9.6 * v3(1., 1.1, 1.25) + d.y * 22. * ext * (1. - 0.75 * jw)) * dt * 1.6);
    }
    return c;
}
// core light colour (vs 4500 K)
fn core_tint() -> v3 { return bb(p.ctemp * 1000.) / bb(4500.); }
// bulge light at r3
fn core_L(r3: f32) -> v3 { return v3(1.0, 0.84, 0.62) * core_tint() * 18. * p.glow * p.core * pow(1. + r3 * r3 / (0.01 * p.core * p.core), -0.75); }
fn ign(px: v2) -> f32 { return fract(52.9829189 * fract(dot(px, v2(0.06711056, 0.00583715)))); }

// P1: galaxy, stars, sky; t0 = shadow map
@compute @workgroup_size(16,16,1)
fn scene(@builtin(global_invocation_id) id: u3) {
    let d = vec2<u32>(textureDimensions(out)); if (id.x >= d.x || id.y >= d.y) { return; }
    let R = v2(d);
    init_lines();
    // TAA jitter (R2)
    let jit = select((fract(f32(u_t.frame) * v2(0.7548776662, 0.5698402910)) - 0.5), v2(0.), p.taw <= 0.001);
    let uv = (v2(id.xy) + 0.5 + jit) / R;
    let s0 = v2((uv.x * 2. - 1.) * R.x / R.y, 1. - uv.y * 2.);
    let sr = v2(s0.x * cos(-p.tlt) - s0.y * sin(-p.tlt), s0.x * sin(-p.tlt) + s0.y * cos(-p.tlt));
    let B = m3(cam_rot(v3(1., 0., 0.)), cam_rot(v3(0., 1., 0.)), cam_rot(v3(0., 0., 1.)));
    let ro = cam_pos();
    let rd = cam_rot(normalize(v3(sr / (1.7 * p.zm), 1.)));
    let Rs = p.size; let tt = u_t.time * 0.25 * p.anim;
    // pattern frame
    let ang = -0.12 * tt;
    let o = rotz(ro / Rs, -ang); let dr = rotz(rd, -ang);
    RD = dr; SB = 0.; RING = 0.;
    let pa = 2. / (R.y * 1.7 * p.zm);
    var col = v3(0.); var tr = v3(1.);
    // depth for TAA
    var dw = 0.; var dsum = 0.;

    // wind in front / behind
    let wsp = wind_span(o, dr);
    var tmid = 1e9; if (abs(dr.z) > 1e-6 && -o.z / dr.z > 0.) { tmid = -o.z / dr.z; }
    let wdi = fract(ign(v2(id.xy)) + f32(u_t.frame % 64u) * 0.618034 + 0.5);
    col += wind_seg(o, dr, wsp.x, min(wsp.y, tmid), wdi, pa, &tr, &dw, &dsum);

    // march the disk ellipsoid; bulge over the whole ray
    let zmax = max(0.32 * p.core, 0.22 * p.thick);
    let sc = v3(1. / 1.32, 1. / 1.32, 1. / zmax);
    let o2 = o * sc; let d2 = dr * sc;
    let qa = dot(d2, d2); let qb = dot(o2, d2); let qc = dot(o2, o2) - 1.;
    let hq = qb * qb - qa * qc;
    var tb = 0.;
    if (hq > 0.) {
        let t0 = max((-qb - sqrt(hq)) / qa, 0.); let t1 = (-qb + sqrt(hq)) / qa;
        if (t1 > t0) {
            col += bulge(o, dr, 0., t0);
            // IGN + golden ratio dither
            let dither = fract(ign(v2(id.xy)) + f32(u_t.frame % 64u) * 0.618034);
            let az = max(abs(dr.z), 0.02);
            let hz0 = 0.045 * p.thick * (1. + 0.8 * p.relief);
            // star DDA, two detail levels
            let kf = clamp(log2(3.2 / max(length(o), 0.05)), 0., 2.5);
            let ka = floor(kf); let fw = kf - ka;
            var wa = walk_at(o, dr, t0, 48. * exp2(ka), ka);
            var wb = walk_at(o, dr, t0, 96. * exp2(ka), ka + 1.);
            var cring = select(-1., 0., p.gflow <= 0.);
            var lcell = v3(1e9);
            var t = t0; var first = true;
            for (var i = 0; i < 320; i++) {
                if (t >= t1 || max(tr.r, max(tr.g, tr.b)) < 0.01) { break; }
                let wt = o + dr * t;
                // adaptive steps
                let hz = hz0 * (0.3 + length(wt.xy));
                // smaller steps near the camera
                var dt = clamp(0.45 * hz / az, clamp(t * 0.002, 0.0005, 0.0025), clamp(t * 0.02, 0.002, 0.014));
                if (abs(wt.z) > 3. * hz) { dt = max(dt, (abs(wt.z) - 2.5 * hz) / az); }
                if (first) { dt *= max(dither, 0.05); first = false; }
                dt = min(dt, t1 - t);
                let w = o + dr * (t + dt * 0.5);
                FP = t * pa;
                // nearby clusters
                var Ll = v3(0.); var Le = v3(0.);
                if (p.clus > 0. && abs(w.z) < 3. * hz) {
                  // own ring + neighbour
                  let rr0 = ring_of(w); let rr1 = rr0 + select(-1., 1., fract(length(w.xy) / RW) > 0.5);
                  for (var j = 0; j < select(2, 1, p.gflow <= 0.); j++) {
                    let ri = select(rr0, rr1, j == 1); let sb = ring_sb(ri);
                    let wm = rotz(w, -sb);
                    let c2 = floor(wm.xy * G2);
                    for (var k = 0; k < 9; k++) {
                        let cl = cluster(c2 + v2(f32(k % 3 - 1), f32(k / 3 - 1)), wm, true, sb, ri);
                        if (cl.p.w <= 0.) { continue; }
                        let dv = w - cl.p.xyz; let d2 = dot(dv, dv);
                        let wf = (1. - d2 / (LR * LR)) * (1. - d2 / (LR * LR));
                        let I = cl.p.w * wf / (1. + d2 / 0.0003) * mix(1., hg(dot(dr, dv) / sqrt(d2 + 1e-8), 0.5) * 12.566, 0.5);
                        Ll += cl.c * I * 5.; Le += cl.e * I * 3.;
                        // supernova light
                        if (cl.s.x + cl.s.z > 0.001) {
                            Ll += v3(0.85, 0.9, 1.) * cl.s.x * 12. * cl.p.w * wf / (1. + d2 / 0.0003);
                            let rr = (sqrt(d2) - cl.s.y) / 0.0025;
                            Le += lc(v3(0.9, 0.35, 0.5), 0.5) * cl.s.z * exp(-rr * rr) * 6.;
                        }
                    }
                    if (j == 0 && any(v3(c2, ri) != lcell)) {
                        lcell = v3(c2, ri);
                        let cl = cluster(c2, wm, false, sb, ri);
                        if (cl.p.w > 0.) {
                            let tc = dot(cl.p.xyz - o, dr);
                            let dd = length(o + dr * tc - cl.p.xyz); let sg = max(tc, 0.) * pa * 0.9;
                            col += (cl.c * cl.p.w * 1.2 + v3(0.85, 0.9, 1.) * cl.s.x * 30.) * exp(-dd * dd / (sg * sg + 1e-9)) * tr * p.br;
                        }
                    }
                  }
                }
                if (p.stars > 0. || p.hii > 0.) {
                    // ring crossing
                    let ra = select(ring_of(o + dr * t), 0., p.gflow <= 0.);
                    let rb = select(ring_of(o + dr * (t + dt)), 0., p.gflow <= 0.);
                    col += ring_stars(&wa, &wb, &cring, ra, o, dr, pa, t, t + dt, ka, fw) * tr * p.br;
                    if (rb != ra) { col += ring_stars(&wa, &wb, &cring, rb, o, dr, pa, t, t + dt, ka, fw) * tr * p.br; }
                }
                col += bulge(o, dr, t, t + dt) * tr;
                if (abs(w.z) < 3.5 * hz) {
                    let g = gsample(w, dr, Ll, Le);
                    let ce = g.em * dt * 1.6 * tr * p.br;
                    col += ce;
                    let cl_ = dot(ce, LUMA); dw += cl_; dsum += cl_ * t;
                    // wavelength extinction
                    if (g.ext.x > 0.) { tr *= exp(-g.ext * dt * 1.6); }
                }
                t += dt;
            }
            tb = t;
        }
    }
    col += wind_seg(o, dr, max(wsp.x, tmid), wsp.y, wdi, pa, &tr, &dw, &dsum);
    col += bulge(o, dr, tb, 1e4) * tr;
    col += globulars(ro / Rs, rd, select(1e4, dsum / dw, dw > 1e-4), tr, pa);
    col += (starfield(rd, pa) + skyband(rd)) * tr + suns(sr, B);
    textureStore(out, vec2<i32>(id.xy), v4(max(col, v3(0.)), select(1e4, dsum / dw, dw > 1e-4)));
}

// clamp half a texel (repeat sampler)
fn cuv(uv: v2, d: v2) -> v2 { return clamp(uv, 0.5/d, 1.-0.5/d); }
fn tap0(uv: v2) -> v3 { let d=v2(textureDimensions(t0)); return textureSampleLevel(t0,s0,cuv(uv,d),0.).rgb; }
fn tap1(uv: v2) -> v3 { let d=v2(textureDimensions(t1)); return textureSampleLevel(t1,s1,cuv(uv,d),0.).rgb; }

// Catmull-Rom history (9 taps)
fn hist_cr(uv: v2) -> v3 {
    let dm = v2(textureDimensions(t1));
    let sp = uv * dm; let tp = floor(sp - 0.5) + 0.5; let f = sp - tp;
    let w0 = f * (-0.5 + f * (1. - 0.5 * f)); let w1 = 1. + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2. - 1.5 * f)); let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let a = (tp - 1.) / dm; let b = (tp + w2 / w12) / dm; let c = (tp + 2.) / dm;
    var r = (tap1(v2(a.x, a.y)) * w0.x + tap1(v2(b.x, a.y)) * w12.x + tap1(v2(c.x, a.y)) * w3.x) * w0.y;
    r += (tap1(v2(a.x, b.y)) * w0.x + tap1(v2(b.x, b.y)) * w12.x + tap1(v2(c.x, b.y)) * w3.x) * w12.y;
    r += (tap1(v2(a.x, c.y)) * w0.x + tap1(v2(b.x, c.y)) * w12.x + tap1(v2(c.x, c.y)) * w3.x) * w3.y;
    return max(r, v3(0.));
}
// reprojection
fn reproject(uv: v2, R: v2, depth: f32) -> v2 {
    let s0 = v2((uv.x * 2. - 1.) * R.x / R.y, 1. - uv.y * 2.);
    let sr = v2(s0.x * cos(-p.tlt) - s0.y * sin(-p.tlt), s0.x * sin(-p.tlt) + s0.y * cos(-p.tlt));
    let rd = cam_rot(normalize(v3(sr / (1.7 * p.zm), 1.)));
    // engine step, exact on export
    let tp = u_t.time - u_t.delta;
    let sp = cam_state(tp, p.pyaw, p.pview, p.pdist);
    var v = rd;
    if (depth < 1e3) {
        let w = cam_pos() / p.size + rd * depth;
        let da = 0.12 * 0.25 * p.anim * u_t.delta;
        v = rotz(w, da) - cam_pose(pan() - v3(0., 0., sp.z), sp.x, sp.y) / p.size;
    }
    let q = v3(dot(v, cam_pose(v3(1., 0., 0.), sp.x, sp.y)), dot(v, cam_pose(v3(0., 1., 0.), sp.x, sp.y)), dot(v, cam_pose(v3(0., 0., 1.), sp.x, sp.y)));
    let srp = q.xy / max(q.z, 1e-4) * 1.7 * p.pzm;
    let s0p = v2(srp.x * cos(p.tlt) - srp.y * sin(p.tlt), srp.x * sin(p.tlt) + srp.y * cos(p.tlt));
    return v2(s0p.x * R.y / R.x * 0.5 + 0.5, 0.5 - s0p.y * 0.5);
}

// P3: TAA resolve (gelami/mrange)
@compute @workgroup_size(16,16,1)
fn taa(@builtin(global_invocation_id) id: u3) {
    let d=vec2<u32>(textureDimensions(out)); if(id.x>=d.x||id.y>=d.y){return;}
    let Rl=v2(d); let uv=(v2(id.xy)+0.5)/Rl;
    let cur=r2y(tap0(uv));
    // variance clip
    var m1=v3(0.); var m2=v3(0.); var dep=1e4;
    for(var y=-1;y<=1;y++){ for(var x=-1;x<=1;x++){
        let tu=uv+v2(f32(x),f32(y))/Rl;
        let n=r2y(tap0(tu)); m1+=n; m2+=n*n;
        dep=min(dep, textureSampleLevel(t0,s0,cuv(tu,Rl),0.).a);
    }}
    m1/=9.; m2/=9.;
    let sd=sqrt(max(m2-m1*m1,v3(0.)))*1.25;
    let pu=reproject(uv, Rl, dep);
    let inside=all(pu>=v2(0.)) && all(pu<=v2(1.));
    let hc=clamp(r2y(hist_cr(pu)),m1-sd,m1+sd);
    // tonemapped weights
    let bl=select(p.taw,0.,u_t.frame<4u || !inside);
    let wc=(1.-bl)/(1.+cur.x); let wh=bl/(1.+max(hc.x,0.));
    textureStore(out, vec2<i32>(id.xy), v4(max(v3(0.), y2r((cur*wc+hc*wh)/(wc+wh))),1.));
}

// P2: DoF (Gustafsson spiral)
// max disc (% height)
fn coc_max(R: v2) -> f32 { return max(p.brad * 0.01 * R.y, 1.); }
// focus range
fn coc(dep: f32, f: f32, R: v2) -> f32 {
    let dz=dep-f; let e=f+sign(dz)*max(abs(dz)-p.frng,0.);
    return min(p.ap * 0.05 * R.y * p.zm * abs(1. / f - 1. / max(e, 1e-3)), coc_max(R));
}
@compute @workgroup_size(16,16,1)
fn dof(@builtin(global_invocation_id) id: u3) {
    let d=vec2<u32>(textureDimensions(out)); if(id.x>=d.x||id.y>=d.y){return;}
    let R=v2(d); let ip=vec2<i32>(id.xy);
    let c0=textureLoad(t0,ip,0);
    if(p.ap<=0.){ textureStore(out,ip,c0); return; }
    // focus 1 = galaxy centre
    let f=max(cam_focus_dist()/p.size*p.focus,1e-3);
    let cs=coc(c0.a,f,R);
    // more taps for bigger discs
    let mr=coc_max(R); let rs=mr*mr/(2.*clamp(mr*mr*0.6,60.,1000.)); let hi=vec2<i32>(d)-1;
    var acc=c0.rgb; var tot=1.; var r=rs;
    var a=fract(ign(v2(id.xy))+f32(u_t.frame%64u)*0.618034)*tau;
    for(var i=0;i<1100;i++){
        if(r>=mr){break;}
        let off=v2(cos(a),sin(a))*r;
        let q=clamp(ip+vec2<i32>(round(off)),vec2<i32>(0),hi);
        let dep=textureLoad(t0,q,0).a;
        let sc=textureSampleLevel(t0,s0,cuv((v2(ip)+0.5+off)/R,R),0.).rgb;
        // no bleed over sharper pixels
        var ss=coc(dep,f,R);
        if(dep>c0.a){ ss=min(ss,cs*2.); }
        let m=smoothstep(r-0.5,r+0.5,ss);
        let hw=1.+m*p.bokeh*min(dot(sc,LUMA),8.);
        acc+=mix(acc/tot,sc,m)*hw; tot+=hw;
        a+=2.39996323; r+=rs/r;
    }
    textureStore(out,ip,v4(acc/tot,c0.a));
}

// bloom mip chain (Karis on first step)
fn karis(c: v3) -> f32 { return 1./(1.+dot(c,v3(0.2126,0.7152,0.0722))); }
fn down13(uv: v2, first: bool) -> v3 {
    let e=1./v2(textureDimensions(t0));
    let a=tap0(uv+e*v2(-2.,-2.)); let b=tap0(uv+e*v2(0.,-2.)); let c=tap0(uv+e*v2(2.,-2.));
    let d=tap0(uv+e*v2(-1.,-1.)); let f=tap0(uv+e*v2(1.,-1.));
    let g=tap0(uv+e*v2(-2.,0.)); let h=tap0(uv); let i=tap0(uv+e*v2(2.,0.));
    let j=tap0(uv+e*v2(-1.,1.)); let k=tap0(uv+e*v2(1.,1.));
    let l=tap0(uv+e*v2(-2.,2.)); let m=tap0(uv+e*v2(0.,2.)); let n=tap0(uv+e*v2(2.,2.));
    // 13-tap downsample
    let q0=(d+f+j+k)*0.25; let q1=(a+b+g+h)*0.25; let q2=(b+c+h+i)*0.25; let q3=(g+h+l+m)*0.25; let q4=(h+i+m+n)*0.25;
    if(first){
        let w0=karis(q0)*0.5; let w1=karis(q1)*0.125; let w2=karis(q2)*0.125; let w3=karis(q3)*0.125; let w4=karis(q4)*0.125;
        return (q0*w0+q1*w1+q2*w2+q3*w3+q4*w4)/(w0+w1+w2+w3+w4);
    }
    return q0*0.5+(q1+q2+q3+q4)*0.125;
}
// soft-knee threshold
fn bright_part(c: v3) -> v3 {
    let t=max(p.bthr,0.01)/max(p.expo,0.01); let knee=0.7*t;
    let br=max(c.r,max(c.g,c.b));
    var soft=clamp(br-t+knee, 0., 2.*knee); soft=soft*soft/(4.*knee+1e-4);
    return c*(max(soft, br-t)/max(br,1e-4));
}
// tent upsample
fn up9(uv: v2) -> v3 {
    let e=1./v2(textureDimensions(t0));
    var c=tap0(uv)*4.;
    c+=(tap0(uv+v2(e.x,0.))+tap0(uv-v2(e.x,0.))+tap0(uv+v2(0.,e.y))+tap0(uv-v2(0.,e.y)))*2.;
    c+=tap0(uv+e)+tap0(uv-e)+tap0(uv+v2(e.x,-e.y))+tap0(uv+v2(-e.x,e.y));
    return c/16.*(0.3+0.4*p.halo)+tap1(uv);
}
fn px_uv(id: u3) -> v2 { return (v2(id.xy)+0.5)/v2(textureDimensions(out)); }
// bloom levels: down to 1/16, then back up
@compute @workgroup_size(16,16,1)
fn bloom_d1(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    let uv=px_uv(id);
    let avg=down13(uv, true);
    var c=bright_part(avg)*p.blm;
    // stars always feed the bloom
    if(p.sglow>0.){
        let e=0.5/v2(textureDimensions(t0));
        let a=tap0(uv+v2(-e.x,-e.y)); let b=tap0(uv+v2(e.x,-e.y)); let f=tap0(uv+v2(-e.x,e.y)); let g=tap0(uv+v2(e.x,e.y));
        var pk=a; if(dot(b,LUMA)>dot(pk,LUMA)){pk=b;} if(dot(f,LUMA)>dot(pk,LUMA)){pk=f;} if(dot(g,LUMA)>dot(pk,LUMA)){pk=g;}
        let lp=dot(pk,LUMA); let lm=dot(avg,LUMA);
        c+=pk*clamp((lp-2.*lm)/max(lp,1e-4),0.,1.)*p.sglow;
    }
    textureStore(out, vec2<i32>(id.xy), v4(c,1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_d2(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(down13(px_uv(id), false),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_d3(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(down13(px_uv(id), false),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_d4(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(down13(px_uv(id), false),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_d5(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(down13(px_uv(id), false),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_d6(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(down13(px_uv(id), false),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_u5(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(up9(px_uv(id)),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_u4(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(up9(px_uv(id)),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_u3(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(up9(px_uv(id)),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_u2(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(up9(px_uv(id)),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_u1(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(up9(px_uv(id)),1.));
}

// post: t0 = TAA, t1 = bloom
// ACES + hue-preserving blend
fn tonemap(c: v3) -> v3 {
    let m = max(c.r, max(c.g, c.b));
    let h = c * (aces(v3(m)).x / max(m, 1e-5));
    return mix(aces(c), h, p.hpres);
}
fn grade(uv: v2, Rl: v2) -> v3 {
    var col = tap0(uv) + tap1(uv) * 0.45;
    // log density
    if(p.logd>0.){
        let k=8.; let L=max(max(col.r,col.g),max(col.b,1e-5));
        col*=mix(1., log(1.+k*L)/(k*L)*(k/log(1.+k)), p.logd);
    }
    col*=p.expo;
    let vc=(uv-0.5)*v2(Rl.x/Rl.y,1.);
    col*=1.-p.vig*smoothstep(0.3,1.1,length(vc));
    let l=dot(col,LUMA); col=mix(v3(l),col,p.sat);
    col=tonemap(max(col,v3(0.)));
    return pow(max(col,v3(0.)),v3(1./p.gam));
}
@compute @workgroup_size(16,16,1)
fn main_image(@builtin(global_invocation_id) id: u3) {
    let d=vec2<u32>(textureDimensions(out)); if(id.x>=d.x||id.y>=d.y){return;}
    let Rl=v2(d); let uv=(v2(id.xy)+0.5)/Rl;
    var col=grade(uv,Rl);
    // luma sharpen
    if(p.shp>0.001){
        let e=1./Rl;
        let bl=(grade(uv+v2(e.x,0.),Rl)+grade(uv-v2(e.x,0.),Rl)+grade(uv+v2(0.,e.y),Rl)+grade(uv-v2(0.,e.y),Rl))*0.25;
        let l=dot(col,LUMA); let dl=clamp((l-dot(bl,LUMA))*p.shp*2.5,-0.2,0.2);
        col*=clamp((l+dl)/max(l,1e-4),0.5,2.);
    }
    textureStore(out, vec2<i32>(id.xy), v4(clamp(col,v3(0.),v3(1.)),1.));
}
