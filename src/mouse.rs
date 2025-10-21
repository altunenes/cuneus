use crate::UniformProvider;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MouseUniform {
    pub position: [f32; 2],
    pub click_position: [f32; 2],
    pub wheel: [f32; 2],
    pub buttons: [u32; 2],
}

impl Default for MouseUniform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            click_position: [0.0, 0.0],
            wheel: [0.0, 0.0],
            buttons: [0, 0],
        }
    }
}

impl UniformProvider for MouseUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

pub struct MouseTracker {
    pub uniform: MouseUniform,
    pub raw_position: [f32; 2],
    pub is_inside_window: bool,
}

impl Default for MouseTracker {
    fn default() -> Self {
        Self {
            uniform: MouseUniform::default(),
            raw_position: [0.0, 0.0],
            is_inside_window: false,
        }
    }
}

impl MouseTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle_mouse_input(
        &mut self,
        event: &WindowEvent,
        window_size: [f32; 2],
        ui_handled: bool,
    ) -> bool {
        // If UI already handled the event, don't update mouse for shader
        if ui_handled {
            return false;
        }

        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x as f32;
                let y = position.y as f32;
                self.raw_position = [x, y];

                self.uniform.position[0] = x / window_size[0];
                self.uniform.position[1] = y / window_size[1];
                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                use winit::event::{ElementState, MouseButton};

                let pressed = *state == ElementState::Pressed;
                let bit_mask = match button {
                    MouseButton::Left => 1,
                    MouseButton::Right => 2,
                    MouseButton::Middle => 4,
                    MouseButton::Back => 8,
                    MouseButton::Forward => 16,
                    MouseButton::Other(b) => {
                        if *b < 27 {
                            1 << (b + 5)
                        } else {
                            0
                        }
                    }
                };

                if pressed {
                    self.uniform.buttons[0] |= bit_mask;
                    self.uniform.click_position = self.uniform.position;
                } else {
                    self.uniform.buttons[0] &= !bit_mask;
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;

                match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        self.uniform.wheel[0] += *x;
                        self.uniform.wheel[1] += *y;
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        self.uniform.wheel[0] += pos.x as f32 / 100.0;
                        self.uniform.wheel[1] += pos.y as f32 / 100.0;
                    }
                }
                true
            }
            WindowEvent::CursorLeft { .. } => {
                self.is_inside_window = false;
                true
            }
            WindowEvent::CursorEntered { .. } => {
                self.is_inside_window = true;
                true
            }
            _ => false,
        }
    }

    pub fn reset_wheel(&mut self) {
        self.uniform.wheel = [0.0, 0.0];
    }
}
