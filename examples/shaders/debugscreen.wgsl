// Cuneus Debug Screen
// note that, WebGPU only supports a maximum of 4 groups (0-3). But you can use more bindings :)

// Group 0: Time
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

// Group 1: Output
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

// Group 2: Engine Resources
struct MouseUniform {
    position: vec2<f32>,
    click_position: vec2<f32>,
    wheel: vec2<f32>,
    buttons: vec2<u32>,
}
@group(2) @binding(0) var<uniform> u_mouse: MouseUniform;

struct FontTextureUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    grid_size: vec2<f32>,
}
@group(2) @binding(1) var<uniform> u_font_texture: FontTextureUniforms;
@group(2) @binding(2) var t_font_texture_atlas: texture_2d<f32>;
@group(2) @binding(3) var<storage, read_write> audio_buffer: array<f32>;

// Group 3: Multi-pass inputs
@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;
@group(3) @binding(4) var input_texture2: texture_2d<f32>;
@group(3) @binding(5) var input_sampler2: sampler;


fn sample_input(coord: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(input_texture0, 0));
    let clamped = clamp(coord, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(input_texture0, clamped, 0);
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs((h / 1.047197551) % 2.0 - 1.0));
    let m = v - c;

    var rgb = vec3<f32>(0.0);

    if (h < 1.047197551) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h < 2.094395102) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h < 3.141592654) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h < 4.188790205) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h < 5.235987756) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + vec3<f32>(m);
}


const FONT_SPACING: f32 = 2.0;

const CHAR_SPACE: u32 = 32u;
const CHAR_EXCLAMATION: u32 = 33u;
const CHAR_COMMA: u32 = 44u;
const CHAR_MINUS: u32 = 45u;
const CHAR_PERIOD: u32 = 46u;
const CHAR_COLON: u32 = 58u;

const CHAR_0: u32 = 48u;
const CHAR_1: u32 = 49u;
const CHAR_2: u32 = 50u;
const CHAR_3: u32 = 51u;
const CHAR_4: u32 = 52u;
const CHAR_5: u32 = 53u;
const CHAR_6: u32 = 54u;
const CHAR_7: u32 = 55u;
const CHAR_8: u32 = 56u;
const CHAR_9: u32 = 57u;

const CHAR_A: u32 = 65u;
const CHAR_B: u32 = 66u;
const CHAR_C: u32 = 67u;
const CHAR_D: u32 = 68u;
const CHAR_E: u32 = 69u;
const CHAR_F: u32 = 70u;
const CHAR_G: u32 = 71u;
const CHAR_H: u32 = 72u;
const CHAR_I: u32 = 73u;
const CHAR_P: u32 = 80u;
const CHAR_R: u32 = 82u;
const CHAR_S: u32 = 83u;
const CHAR_T: u32 = 84u;

const CHAR_a: u32 = 97u;
const CHAR_b: u32 = 98u;
const CHAR_c: u32 = 99u;
const CHAR_d: u32 = 100u;
const CHAR_e: u32 = 101u;
const CHAR_g: u32 = 103u;
const CHAR_h: u32 = 104u;
const CHAR_i: u32 = 105u;
const CHAR_l: u32 = 108u;
const CHAR_m: u32 = 109u;
const CHAR_n: u32 = 110u;
const CHAR_o: u32 = 111u;
const CHAR_r: u32 = 114u;
const CHAR_s: u32 = 115u;
const CHAR_u: u32 = 117u;
const CHAR_x: u32 = 120u;

// render single character
fn ch(pp: vec2<f32>, pos: vec2<f32>, code: u32, size: f32) -> f32 {
    let char_size_pixels = vec2<f32>(size, size);
    let relative_pos = pp - pos;

    // Check bounds
    if (relative_pos.x < 0.0 || relative_pos.x >= char_size_pixels.x ||
        relative_pos.y < 0.0 || relative_pos.y >= char_size_pixels.y) {
        return 0.0;
    }

    // Calculate UV coordinates within the character cell
    let local_uv = relative_pos / char_size_pixels;

    // calc char pos in atlas grid (16x16)
    let grid_x = code % 16u;
    let grid_y = code / 16u;

    // padding to avoid cell bleeding
    let padding = 0.05;
    let padded_uv = local_uv * (1.0 - 2.0 * padding) + vec2<f32>(padding);

    // atlas UV coords
    let cell_size_uv = vec2<f32>(1.0 / 16.0, 1.0 / 16.0);
    let cell_offset = vec2<f32>(f32(grid_x), f32(grid_y)) * cell_size_uv;
    let final_uv = cell_offset + padded_uv * cell_size_uv;

    let atlas_coord = vec2<i32>(
        i32(final_uv.x * u_font_texture.atlas_size.x),
        i32(final_uv.y * u_font_texture.atlas_size.y)
    );
    let sample = textureLoad(t_font_texture_atlas, atlas_coord, 0);

    // red channel font data + anti-alias
    let font_alpha = sample.r * 0.8;
    return smoothstep(0.1, 0.9, font_alpha);
}

// char spacing
fn adv(size: f32) -> f32 {
    return size * (1.0 / FONT_SPACING);
}

// word rendering
fn word(pp: vec2<f32>, pos: vec2<f32>, chars: array<u32, 32>, length: u32, size: f32) -> f32 {
    let char_advance = adv(size);
    var alpha = 0.0;

    for (var i = 0u; i < min(length, 32u); i++) {
        let char_pos = pos + vec2<f32>(f32(i) * char_advance, 0.0);
        let char_alpha = ch(pp, char_pos, chars[i], size);
        alpha = max(alpha, char_alpha);
    }

    return alpha;
}

// render number
fn num(pp: vec2<f32>, pos: vec2<f32>, number: u32, size: f32) -> f32 {
    let char_advance = adv(size);
    var alpha = 0.0;
    var temp_num = number;
    var digit_count = 0u;

    // Count digits
    if (temp_num == 0u) {
        digit_count = 1u;
    } else {
        var count_temp = temp_num;
        while (count_temp > 0u) {
            count_temp = count_temp / 10u;
            digit_count++;
        }
    }

    // Render digits from right to left
    temp_num = number;
    for (var i = 0u; i < digit_count; i++) {
        let digit = temp_num % 10u;
        let digit_char_code = CHAR_0 + digit;
        let digit_pos = pos + vec2<f32>(f32(digit_count - 1u - i) * char_advance, 0.0);
        let char_alpha = ch(pp, digit_pos, digit_char_code, size);
        alpha = max(alpha, char_alpha);
        temp_num = temp_num / 10u;
    }

    return alpha;
}

fn digit_count(n: u32) -> u32 {
    if (n == 0u) { return 1u; }
    var count = 0u;
    var v = n;
    while (v > 0u) {
        v = v / 10u;
        count++;
    }
    return count;
}

fn float1(pp: vec2<f32>, pos: vec2<f32>, value: f32, size: f32) -> f32 {
    let char_advance = adv(size);
    var alpha = 0.0;
    var current_pos = pos;

    let whole_part = u32(value);
    let whole_alpha = num(pp, current_pos, whole_part, size);
    alpha = max(alpha, whole_alpha);
    current_pos.x += f32(digit_count(whole_part)) * char_advance;

    let dot_alpha = ch(pp, current_pos, CHAR_PERIOD, size);
    alpha = max(alpha, dot_alpha);
    current_pos.x += char_advance;

    let decimal_part = u32((value - f32(whole_part)) * 10.0);
    let decimal_alpha = ch(pp, current_pos, CHAR_0 + decimal_part, size);
    alpha = max(alpha, decimal_alpha);

    return alpha;
}

// animated hello cuneus text
fn render_hello_cuneus_animated(pixel_pos: vec2<f32>, screen_center: vec2<f32>) -> vec3<f32> {
    let base_size = 100.0;
    let size_pulse = sin(u_time.time * 2.0) * 20.0;
    let size = base_size + size_pulse;

    let char_advance = adv(size);
    var result = vec3<f32>(0.0);

    let hello_cuneus = array<u32, 32>(
        CHAR_H, CHAR_e, CHAR_l, CHAR_l, CHAR_o, CHAR_SPACE, CHAR_C, CHAR_u, CHAR_n, CHAR_e, CHAR_u, CHAR_s,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    );

    let text_width = 12.0 * char_advance;
    let start_x = screen_center.x - text_width * 0.5;
    let start_y = screen_center.y - size * 0.5;

    let wave_amplitude = 30.0;
    let wave_frequency = 0.5;
    for (var i = 0u; i < 12u; i++) {
        let wave_offset = sin(u_time.time * 3.0 + f32(i) * wave_frequency) * wave_amplitude;

        let hue = (u_time.time + f32(i) * 0.3) % 6.28318;
        let char_color = hsv_to_rgb(hue, 0.8, 1.0);

        let char_pos = vec2<f32>(
            start_x + f32(i) * char_advance,
            start_y + wave_offset
        );
        let char_alpha = ch(pixel_pos, char_pos, hello_cuneus[i], size);
        result += char_color * char_alpha;
    }

    return result;
}

fn print_debug_text(pixel_pos: vec2<f32>, text_pos: vec2<f32>) -> f32 {
    let size = 32.0;
    let debug_text = array<u32, 32>(
        CHAR_D, CHAR_e, CHAR_b, CHAR_u, CHAR_g,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    );
    return word(pixel_pos, text_pos, debug_text, 5u, size);
}

fn print_time_display(pixel_pos: vec2<f32>, text_pos: vec2<f32>) -> f32 {
    let size = 32.0;
    let char_advance = adv(size);
    var alpha = 0.0;

    let time_label = array<u32, 32>(
        CHAR_T, CHAR_i, CHAR_m, CHAR_e, CHAR_COLON, CHAR_SPACE,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    );
    alpha = max(alpha, word(pixel_pos, text_pos, time_label, 6u, size));

    let time_number_pos = text_pos + vec2<f32>(6.0 * char_advance, 0.0);
    alpha = max(alpha, float1(pixel_pos, time_number_pos, u_time.time, size));

    return alpha;
}

fn print_fps_display(pixel_pos: vec2<f32>, text_pos: vec2<f32>) -> f32 {
    let size = 32.0;
    let char_advance = adv(size);
    var alpha = 0.0;

    let fps_label = array<u32, 32>(
        CHAR_F, CHAR_P, CHAR_S, CHAR_COLON, CHAR_SPACE,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    );
    alpha = max(alpha, word(pixel_pos, text_pos, fps_label, 5u, size));

    let fps = u32(1.0 / max(u_time.delta, 0.001));
    let fps_number_pos = text_pos + vec2<f32>(5.0 * char_advance, 0.0);
    alpha = max(alpha, num(pixel_pos, fps_number_pos, fps, size));

    return alpha;
}

fn print_frame_display(pixel_pos: vec2<f32>, text_pos: vec2<f32>) -> f32 {
    let size = 32.0;
    let char_advance = adv(size);
    var alpha = 0.0;

    // "Frame: "
    let frame_label = array<u32, 32>(
        CHAR_F, CHAR_r, CHAR_a, CHAR_m, CHAR_e, CHAR_COLON, CHAR_SPACE,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    );
    alpha = max(alpha, word(pixel_pos, text_pos, frame_label, 7u, size));

    let frame_number_pos = text_pos + vec2<f32>(7.0 * char_advance, 0.0);
    alpha = max(alpha, num(pixel_pos, frame_number_pos, u_time.frame, size));

    return alpha;
}

fn print_res_display(pixel_pos: vec2<f32>, text_pos: vec2<f32>, dims: vec2<u32>) -> f32 {
    let size = 32.0;
    let char_advance = adv(size);
    var alpha = 0.0;

    // "Res: "
    let res_label = array<u32, 32>(
        CHAR_R, CHAR_e, CHAR_s, CHAR_COLON, CHAR_SPACE,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    );
    alpha = max(alpha, word(pixel_pos, text_pos, res_label, 5u, size));

    var current_x = text_pos.x + 5.0 * char_advance;

    // Width number
    alpha = max(alpha, num(pixel_pos, vec2<f32>(current_x, text_pos.y), dims.x, size));
    current_x += f32(digit_count(dims.x)) * char_advance;

    // "x"
    alpha = max(alpha, ch(pixel_pos, vec2<f32>(current_x, text_pos.y), CHAR_x, size));
    current_x += char_advance;

    // Height number
    alpha = max(alpha, num(pixel_pos, vec2<f32>(current_x, text_pos.y), dims.y, size));

    return alpha;
}

fn print_rgb_display(pixel_pos: vec2<f32>, text_pos: vec2<f32>, color: vec3<f32>) -> f32 {
    let size = 24.0;
    let char_advance = adv(size);
    var alpha = 0.0;
    var current_pos = text_pos;

    let rgb_label = array<u32, 32>(
        CHAR_R, CHAR_G, CHAR_B, CHAR_COLON, CHAR_SPACE,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    );
    alpha = max(alpha, word(pixel_pos, current_pos, rgb_label, 5u, size));
    current_pos.x += 5.0 * char_advance;
    let r_val = u32(color.r * 255.0);
    let g_val = u32(color.g * 255.0);
    let b_val = u32(color.b * 255.0);

    alpha = max(alpha, num(pixel_pos, current_pos, r_val, size));
    current_pos.x += f32(digit_count(r_val)) * char_advance;

    alpha = max(alpha, ch(pixel_pos, current_pos, CHAR_COMMA, size));
    current_pos.x += char_advance;

    alpha = max(alpha, num(pixel_pos, current_pos, g_val, size));
    current_pos.x += f32(digit_count(g_val)) * char_advance;

    alpha = max(alpha, ch(pixel_pos, current_pos, CHAR_COMMA, size));
    current_pos.x += char_advance;

    alpha = max(alpha, num(pixel_pos, current_pos, b_val, size));

    return alpha;
}

fn print_mouse_debug(pixel_pos: vec2<f32>, text_pos: vec2<f32>, sampled_color: vec3<f32>) -> f32 {
    let size = 20.0;
    let char_advance = adv(size);
    var alpha = 0.0;

    let mouse_label = array<u32, 32>(
        CHAR_m, CHAR_o, CHAR_u, CHAR_s, CHAR_e, CHAR_COLON, CHAR_SPACE,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    );
    alpha = max(alpha, word(pixel_pos, text_pos, mouse_label, 7u, size));

    let color_pos = text_pos + vec2<f32>(7.0 * char_advance, 0.0);
    alpha = max(alpha, print_rgb_display(pixel_pos, color_pos, sampled_color));

    return alpha;
}

//simple feedback
@compute @workgroup_size(16, 16, 1)
fn effect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }

    let uv = vec2<f32>(
        f32(global_id.x) / f32(dimensions.x),
        f32(global_id.y) / f32(dimensions.y)
    );

    // Read previous frame (self-feedback via multi-pass)
    let prev = sample_input(vec2<i32>(global_id.xy));

    // Animated rainbow background
    let base_col = 0.5 + 0.5 * cos(
        u_time.time +
        uv.xyx * 1.0 +
        vec3<f32>(0.0, 2.0, 4.0)
    );

    // Mouse circle interaction
    let mouse_dist = distance(uv, u_mouse.position);
    let circle_radius = 0.1 + sin(u_time.time) * 0.05;
    let circle_effect = smoothstep(circle_radius, circle_radius - 0.02, mouse_dist);

    let circle_col = vec3<f32>(
        0.5 + 0.5 * sin(u_time.time * 1.1),
        0.5 + 0.5 * sin(u_time.time * 0.7),
        0.5 + 0.5 * sin(u_time.time * 0.5)
    );

    let left_button_pressed = (u_mouse.buttons.x & 1u) != 0u;
    var current = mix(base_col, circle_col, circle_effect);
    if (left_button_pressed && mouse_dist < circle_radius) {
        current = vec3<f32>(1.0) - current;
    }

    // Wheel effect
    let wheel_effect = abs(u_mouse.wheel.y) * 0.2;
    let pulse = sin(u_time.time * 5.0 + mouse_dist * 20.0) * wheel_effect;
    if (wheel_effect > 0.01) {
        current = current * (1.0 + pulse);
    }

    // Trail blend: decay old + blend in new
    let trail_decay = 0.96;
    let blend = max(0.2, circle_effect);
    var result = mix(prev.rgb * trail_decay, current, blend);

    textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(result, 1.0));
}

// Pass 2: Read effect output, overlay all text + info

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }

    let pixel_pos = vec2<f32>(f32(global_id.x), f32(global_id.y));
    let screen_center = vec2<f32>(f32(dimensions.x) * 0.5, f32(dimensions.y) * 0.5);

    var final_col = sample_input(vec2<i32>(global_id.xy)).rgb;
    let hello_color = render_hello_cuneus_animated(pixel_pos, screen_center);
    final_col += hello_color;

    let debug_alpha = print_debug_text(pixel_pos, vec2<f32>(20.0, 20.0));
    if (debug_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.9, 0.9, 0.9), debug_alpha * 0.8);
    }

    let time_alpha = print_time_display(pixel_pos, vec2<f32>(20.0, 70.0));
    if (time_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.9, 0.9, 0.2), time_alpha * 0.8);
    }

    let fps_alpha = print_fps_display(pixel_pos, vec2<f32>(20.0, 120.0));
    if (fps_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.2, 0.9, 0.2), fps_alpha * 0.8);
    }

    let frame_alpha = print_frame_display(pixel_pos, vec2<f32>(20.0, 170.0));
    if (frame_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.2, 0.7, 0.9), frame_alpha * 0.8);
    }

    let res_alpha = print_res_display(pixel_pos, vec2<f32>(20.0, 220.0), dimensions);
    if (res_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.9, 0.5, 0.2), res_alpha * 0.8);
    }

    let color_debug_alpha = print_rgb_display(pixel_pos, vec2<f32>(20.0, 270.0), final_col);
    if (color_debug_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.7, 0.7, 0.7), color_debug_alpha * 0.8);
    }

    let mouse_sampled_color = final_col;
    let mouse_debug_alpha = print_mouse_debug(pixel_pos, vec2<f32>(20.0, 320.0), mouse_sampled_color);
    if (mouse_debug_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.6, 0.8, 0.9), mouse_debug_alpha * 0.8);
    }

    // Audio generation (writes at pixel 0,0 only)
    if (global_id.x == 0u && global_id.y == 0u) {
        let base_frequency = 261.63;
        let note_frequency = base_frequency * (1.0 + sin(u_time.time * 0.5) * 0.1);
        let note_amplitude = 0.2;
        let waveform_type = 0.0;

        audio_buffer[0] = note_frequency;
        audio_buffer[1] = note_amplitude;
        audio_buffer[2] = waveform_type;

        let base_frequencies = array<f32, 9>(
            261.63, 293.66, 329.63, 349.23, 392.00,
            440.00, 493.88, 523.25, 587.33
        );

        for (var i = 0u; i < 9u; i++) {
            audio_buffer[3u + i] = base_frequencies[i];
        }
    }

    textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(final_col, 1.0));
}