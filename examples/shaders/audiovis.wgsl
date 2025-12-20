// MIT License, Enes Altun, 2025

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}

struct AudioVisParams {
    red_power: f32,
    green_power: f32,
    blue_power: f32,
    green_boost: f32,
    contrast: f32,
    gamma: f32,
    glow: f32,
    _padding: f32,
}

// Group 0: Per-Frame Data
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Group 1: Primary I/O & Parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: AudioVisParams;

// Group 2: Engine Resources (audio spectrum as storage buffer)
@group(2) @binding(0) var<storage, read> audio_spectrum: array<f32>;

// Type aliases
alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;

// Hash function for procedural generation
fn h21(p: v2) -> f32 {
    let p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    return fract(p3.x * p3.y * 43758.5453);
}

// Smooth noise
fn sN(p: v2) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(h21(i), h21(i + v2(1.0, 0.0)), u.x),
        mix(h21(i + v2(0.0, 1.0)), h21(i + v2(1.0, 1.0)), u.x),
        u.y
    );
}

// Fractal Brownian Motion
fn fbm(p: v2, o: i32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var f = 1.0;
    var p2 = p;
    
    for (var i = 0; i < o; i++) {
        v += a * sN(p2);
        p2 *= 2.0;
        a *= 0.5;
    }
    
    return v;
}

// Smoke effect
fn sE(uv: v2, t: f32, i: f32) -> v3 {
    let s1 = fbm(uv * 4.0 + v2(t * 0.05, 0.0), 3);
    let s2 = fbm(uv * 2.0 - v2(0.0, t * 0.035), 2);
    let cs = mix(s1, s2, 0.5 + 0.5 * sin(t * 0.1));
    return v3(0.01, 0.01, 0.03) * cs * i;
}

// HSV to RGB conversion
fn hsv2rgb(h: f32, s: f32, v: f32) -> v3 {
    let c = v * s;
    let hp = h * 12.0;
    let x = c * (1.0 - abs(fract(hp / 2.0) * 2.0 - 1.0));
    
    var rgb: v3;
    if(hp < 1.0) { rgb = v3(c, x, 0.0); }
    else if(hp < 2.0) { rgb = v3(x, c, 0.0); }
    else if(hp < 3.0) { rgb = v3(0.0, c, x); }
    else if(hp < 4.0) { rgb = v3(0.0, x, c); }
    else if(hp < 5.0) { rgb = v3(x, 0.0, c); }
    else { rgb = v3(c, 0.0, x); }
    
    return rgb + v3(v - c);
}

// Beam color cycling
fn bC(x: f32, t: f32) -> v3 {
    let xc = (x - (t / 8.0)) * 3.0;
    let mc = fract(xc) * 3.0;
    var c = v3(0.25);
    
    if (mc < 1.0) {
        c.r += 1.0 - mc;
        c.g += mc;
    } else if (mc < 2.0) {
        let ac = mc - 1.0;
        c.g += 1.0 - ac;
        c.b += ac;
    } else {
        let ac = mc - 2.0;
        c.b += 1.0 - ac;
        c.r += ac;
    }
    
    return c;
}

// Neon glow
fn nG(c: v3, i: f32) -> v3 {
    return c * (1.0 + i * i * 4.0);
}

// Bloom effect
fn bloomEffect(c: v3) -> v3 {
    let bp = max(v3(0.0), c - v3(0.7));
    return c + bp * bp * 0.5;
}

// Get audio value at any frequency (0-1 range)
fn gAV(f: f32) -> f32 {
    let idx = f * 64.0;
    let i = u32(idx);
    let fp = idx - f32(i);
    
    if (i >= 64u) { return 0.0; }
    
    let v1 = audio_spectrum[i];
    
    if (i >= 63u) { return v1; }
    
    let v2 = audio_spectrum[i + 1u];
    return mix(v1, v2, fp);
}

// Get audio range energy
fn gAR(l: f32, h: f32) -> f32 {
    var s = 0.0;
    var c = 0.0;
    for(var i = l; i <= h; i += 0.02) {
        s += gAV(i);
        c += 1.0;
    }
    return s / c;
}

// Audio spectrum buffer layout:
//   - Indices 0-63: frequency spectrum magnitudes (RMS-normalized)
//   - Index 64: BPM value (beats per minute)
//   - Index 65: bass energy (pre-computed on CPU)
//   - Index 66: mid energy (pre-computed on CPU)
//   - Index 67: high energy (pre-computed on CPU)
//   - Index 68: total energy (pre-computed on CPU)
fn getBPM() -> f32 { return audio_spectrum[64]; }
fn getBassEnergy() -> f32 { return audio_spectrum[65]; }
fn getMidEnergy() -> f32 { return audio_spectrum[66]; }
fn getHighEnergy() -> f32 { return audio_spectrum[67]; }
fn getTotalEnergy() -> f32 { return audio_spectrum[68]; }

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let tc = v2(f32(gid.x), f32(gid.y)) / v2(f32(dims.x), f32(dims.y));
    let t = time_data.time;

    // Audio energy - pre-computed on CPU in spectrum.rs
    let bE = getBassEnergy();
    let mE = getMidEnergy();
    let hE = getHighEnergy();
    let tE = getTotalEnergy();
    
    // Final color - start with dark background
    var fc = v3(0.005, 0.005, 0.01);
    fc += sE(tc, t, 0.5 + tE * 0.5); 
    
    // Bar visualization
    let eB = 0.15;  // eq bottom
    let eH = 0.7;   // eq height
    let bW = 1.0 / 66.0;  // band width
    let bS = bW * 0.3;  // band spacing
    
    let rF = 1.0 - eB;  // reflection floor
    let rD = 0.1;  // reflection depth

    // BPM bar
    let bpX = 0.015;  // bpm bar X
    let bpW = 0.012;  // bpm bar width
    let bpB = 0.15;   // bpm bar bottom
    let bpH = 0.7;    // bpm bar height

    let bpm = getBPM();
    let bps = bpm / 60.0;  // beats per second
    let bPh = fract(t * bps);  // beat phase
    let bRi = smoothstep(0.0, 0.1, bPh);  // rise
    let bDe = exp(-5.0 * max(0.0, bPh - 0.1));  // decay
    let bPu = select(bDe, bRi, bPh < 0.1) * (0.6 + bE * 0.4);  // pulse

    if (tc.x >= bpX && tc.x < bpX + bpW) {
        let bY = 1.0 - tc.y;
        let bN = (bY - bpB) / bpH;
        if (bN >= 0.0 && bN <= 1.0) {
            if (bN <= bPu) {
                let gT = bN / max(bPu, 0.01);
                let bCol = mix(v3(1.0, 0.2, 0.1), v3(1.0, 0.9, 0.2), gT);
                fc = nG(bCol, 0.5 + bPu * 0.5) * (0.8 + bPu * 0.4);
            }
            let eD = min(min(abs(tc.x - bpX), abs(tc.x - (bpX + bpW))), min(abs(bN), abs(bN - 1.0)) * bpW);
            if (eD < 0.002) {
                fc = mix(fc, v3(0.5, 0.5, 0.6), 0.5);
            }
        }
    }
    if (tc.x >= bpX && tc.x < bpX + bpW && tc.y < 0.12 && tc.y > 0.08) {
        fc += v3(0.8, 0.4, 0.2) * (0.3 + bPu * 0.7) * 0.3;
    }

    let fBO = 0.03;  // freq bar offset

    for (var i = 0; i < 64; i++) {
        let bX = fBO + (f32(i) + 1.0) * bW;  // band X position
        let fT = f32(i) / 64.0;  // frequency factor
        
        // Frequency smoothing 
        let rawA = (gAV(max(0.0, fT - 0.01)) + gAV(fT) * 2.0 + gAV(min(1.0, fT + 0.01))) * 0.25;
        //  adaptive curve
        let adaptFactor = smoothstep(0.2, 0.7, rawA);
        let adaptMult = mix(mix(0.95, 1.3, fT), 1.0, adaptFactor);
        let enhanced = pow(rawA, mix(0.85, 0.6, fT)) * adaptMult;
        let softLimit = mix(0.9, 0.65, fT);
        let bA = softLimit * (1.0 - exp(-enhanced / softLimit * 1.5));

        let bH = bA * eH;  // bar height
        let bB = 1.0 - eB - bH;  // bar bottom
        let bT = 1.0 - eB;  // bar top
        
        let inX = tc.x >= bX && tc.x < bX + bW - bS;  // in X bounds
        let inY = tc.y <= bT && tc.y >= bB;  // in Y bounds
        
        let dX = min(abs(tc.x - bX), abs(tc.x - (bX + bW - bS)));  // distance to X edge
        let dY = min(abs(tc.y - bT), abs(tc.y - bB));  // distance to Y edge
        let eD = min(dX, dY);  // edge distance
        
        // Draw bar
        if (inX && inY) {
            let hT = (bT - tc.y) / max(bH, 0.001);  // height factor
            
            let tE = smoothstep(0.5, 0.0, length(v2(
                (tc.x - (bX + (bW - bS) * 0.5)) / (bW - bS),
                (tc.y - (bB + bH * 0.5)) / bH
            )) * 2.0 - 0.5);
            
            // Neon color
            let nC = hsv2rgb(fT, 0.9, 0.7 + 0.3 * (1.0 - hT)) * (0.7 + 0.3 * tE);
            let fl = 0.95 + 0.05 * sin(t * (1.0 + fT * 2.0) + f32(i));  // flicker
            
            fc = nG(nC, bA) * fl;
        }
        // Outer glow
        else if (eD < 0.01 && ((tc.x >= bX - 0.01 && tc.x < bX + bW - bS + 0.01) &&
                (tc.y <= bT + 0.01 && tc.y >= bB - 0.01))) {
            let gS = smoothstep(0.01, 0.0, eD) * bA;
            fc += hsv2rgb(fT, 0.9, 0.9) * gS * 0.5;
        }
        // Bar reflection
        if (inX && tc.y > rF && tc.y < rF + rD) {
            let rDist = (tc.y - rF) / rD;
            let rInt = (1.0 - rDist) * 3.2 * bA;
            let rCol = hsv2rgb(fT, 0.9, 0.8) * rInt;
            fc = mix(fc, rCol, smoothstep(rD, 0.0, tc.y - rF) * 0.5);
        }
    }
    
    // Waveform visualization
    let wY = 0.1;  // waveform Y position
    let wH = 0.06;  // waveform height
    let wX = tc.x;  // waveform X
    let wV = gAV(wX) * 0.8;  // waveform value
    let wP = wY + sin(wX * 100.0) * wV * wH * (0.8 + tE * 0.4);  // wave
    let dW = abs(tc.y - wP);  // distance to wave
    if (tc.y < wY + wH * 2.0 && tc.y > 0.0) {
        let bI = abs(1.0 / (30.0 * dW * (1.0 + mE * 2.0)));
        let bCol = bC(wX, t);
        fc += bCol * bI * (0.2 + tE * 0.8);
        if (dW < 0.0025) {
            fc = mix(fc, bC(wX - t * 0.05, t) * 1.5, 0.7);
        }
    }
    
    // Scanline effect
    fc *= 1.0 - 0.04 + 0.04 * sin(tc.y * 100.0 + t);
    
    // Color adjustments
    fc.r = pow(fc.r, 1.0 / params.red_power);
    fc.g = pow(fc.g, 1.0 / params.green_power) * (1.0 + params.green_boost);
    fc.b = pow(fc.b, 1.0 / params.blue_power);
    fc = pow(max(fc, v3(0.0)), v3(params.gamma));
    fc = (fc - 0.5) * params.contrast + 0.5;
    fc = bloomEffect(fc) * (1.0 + params.glow * 0.7);
    
    fc *= mix(1.0, smoothstep(0.5, 0.0, length(tc - 0.5) - 0.2), 0.85);
    
    textureStore(output, gid.xy, v4(fc, 1.0));
}