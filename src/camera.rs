//! Orbit / pan / fly camera with smoothing, shared by 3D examples.
//!
//! Mouse: left drag orbits, right or middle drag (or shift + left) pans, wheel zooms.
//! Keys: WASD or arrows move, Q/E up and down, shift for speed, R returns to the framed view.
//! Skip `handle_event` and call `rotate` / `pan` / `zoom` or set `goal` to use your own input.

use crate::math::{Mat4, Vec3};
use std::collections::HashSet;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, NamedKey};

#[derive(Clone, Copy, Debug)]
pub struct OrbitPose {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl OrbitPose {
    /// Direction from the target to the eye
    pub fn offset(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy)
    }
    pub fn eye(&self) -> Vec3 {
        self.target + self.offset() * self.distance
    }
    pub fn forward(&self) -> Vec3 {
        -self.offset()
    }
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward())
    }
    pub fn view_from_world(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }
}

pub struct OrbitCamera {
    /// Where input steers to
    pub goal: OrbitPose,
    /// What is rendered, follows `goal` smoothly
    pub pose: OrbitPose,
    /// Pose restored by `reset`
    pub home: OrbitPose,
    /// Vertical field of view in degrees
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    /// Fly: dragging looks around the eye and WASD moves along the view
    pub fly: bool,
    /// Turntable speed in radians per second
    pub auto_rotate: f32,
    /// Higher is snappier
    pub smoothing: f32,
    /// Scale used for movement speeds and zoom limits
    pub scene_radius: f32,
    drag_orbit: bool,
    drag_pan: bool,
    shift: bool,
    last_mouse: Option<[f32; 2]>,
    keys: HashSet<String>,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new()
    }
}

impl OrbitCamera {
    pub fn new() -> Self {
        let p = OrbitPose { yaw: 0.0, pitch: 0.0, distance: 3.0, target: Vec3::ZERO };
        Self {
            goal: p, pose: p, home: p,
            fov: 45.0, near: 0.01, far: 1000.0,
            fly: false, auto_rotate: 0.0, smoothing: 14.0, scene_radius: 1.0,
            drag_orbit: false, drag_pan: false, shift: false, last_mouse: None, keys: HashSet::new(),
        }
    }

    /// Frame a sphere so it fills the view, and make that the home pose
    pub fn frame(&mut self, center: Vec3, radius: f32) {
        self.scene_radius = radius.max(1e-4);
        let distance = self.scene_radius / (self.fov.to_radians() * 0.5).sin() * 1.05;
        self.home = OrbitPose { yaw: 0.0, pitch: 0.2, distance, target: center };
        self.near = self.scene_radius * 0.01;
        self.far = distance + self.scene_radius * 100.0;
        self.goal = self.home;
        self.pose = self.home;
    }

    pub fn reset(&mut self) {
        self.goal = self.home;
    }

    pub fn eye(&self) -> Vec3 {
        self.pose.eye()
    }

    pub fn view_from_world(&self) -> Mat4 {
        self.pose.view_from_world()
    }

    /// Current pose turned by `auto_rotate * time`, for exports
    pub fn turntable(&self, time: f32) -> OrbitPose {
        OrbitPose { yaw: self.pose.yaw + self.auto_rotate * time, ..self.pose }
    }

    /// Perspective with wgpu's 0..1 depth range
    pub fn clip_from_view(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov.to_radians(), aspect.max(1e-4), self.near, self.far)
    }

    pub fn clip_from_world(&self, aspect: f32) -> Mat4 {
        self.clip_from_view(aspect) * self.view_from_world()
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        if self.fly {
            let eye = self.goal.eye();
            self.goal.yaw -= dx * 0.004;
            self.goal.pitch = (self.goal.pitch + dy * 0.004).clamp(-1.55, 1.55);
            self.goal.target = eye - self.goal.offset() * self.goal.distance;
        } else {
            self.goal.yaw -= dx * 0.008;
            self.goal.pitch = (self.goal.pitch + dy * 0.008).clamp(-1.55, 1.55);
        }
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let k = self.goal.distance * 0.0012;
        let (r, u) = (self.goal.right(), self.goal.up());
        self.goal.target += (-dx * r + dy * u) * k;
    }

    pub fn zoom(&mut self, amount: f32) {
        if self.fly {
            self.goal.target += self.goal.forward() * self.scene_radius * 0.08 * amount;
        } else {
            let f = (1.0 - amount * 0.1).clamp(0.5, 2.0);
            self.goal.distance = (self.goal.distance * f).clamp(self.scene_radius * 0.02, self.scene_radius * 200.0);
        }
    }

    /// Feed window events; returns true when the camera used the event
    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::ModifiersChanged(m) => {
                self.shift = m.state().shift_key();
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => { self.drag_orbit = down; true }
                    MouseButton::Right | MouseButton::Middle => { self.drag_pan = down; true }
                    _ => false,
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = [position.x as f32, position.y as f32];
                let last = self.last_mouse.replace(p);
                let Some(l) = last else { return false };
                let (dx, dy) = (p[0] - l[0], p[1] - l[1]);
                if self.drag_pan || (self.drag_orbit && self.shift) {
                    self.pan(dx, dy);
                    true
                } else if self.drag_orbit {
                    self.rotate(dx, dy);
                    true
                } else {
                    false
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32 / 100.0).clamp(-3.0, 3.0),
                };
                self.zoom(d);
                true
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // arrows act as WASD
                let key = match &event.logical_key {
                    Key::Character(ch) => ch.as_str().to_lowercase(),
                    Key::Named(NamedKey::ArrowUp) => "w".into(),
                    Key::Named(NamedKey::ArrowLeft) => "a".into(),
                    Key::Named(NamedKey::ArrowDown) => "s".into(),
                    Key::Named(NamedKey::ArrowRight) => "d".into(),
                    _ => return false,
                };
                if !matches!(key.as_str(), "w" | "a" | "s" | "d" | "q" | "e" | "r") {
                    return false;
                }
                if event.state == ElementState::Pressed {
                    if key == "r" { self.reset(); } else { self.keys.insert(key); }
                } else {
                    self.keys.remove(&key);
                }
                true
            }
            _ => false,
        }
    }

    /// Advance movement and smoothing; call once per frame with the real frame time
    pub fn update(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        let mut speed = self.scene_radius * 0.8 * dt;
        if self.shift { speed *= 4.0; }
        let f = if self.fly { self.goal.forward() } else { (self.goal.forward() * Vec3::new(1.0, 0.0, 1.0)).normalize_or_zero() };
        let r = self.goal.right();
        let mut m = Vec3::ZERO;
        for k in &self.keys {
            m += match k.as_str() {
                "w" => f, "s" => -f, "d" => r, "a" => -r, "q" => Vec3::Y, "e" => -Vec3::Y,
                _ => Vec3::ZERO,
            };
        }
        self.goal.target += m * speed;
        self.goal.yaw += self.auto_rotate * dt;

        let k = 1.0 - (-dt * self.smoothing).exp();
        let lerp = |a: f32, b: f32| if (b - a).abs() < 1e-6 { b } else { a + (b - a) * k };
        self.pose.yaw = lerp(self.pose.yaw, self.goal.yaw);
        self.pose.pitch = lerp(self.pose.pitch, self.goal.pitch);
        self.pose.distance = lerp(self.pose.distance, self.goal.distance);
        self.pose.target = Vec3::new(
            lerp(self.pose.target.x, self.goal.target.x),
            lerp(self.pose.target.y, self.goal.target.y),
            lerp(self.pose.target.z, self.goal.target.z),
        );
    }
}
