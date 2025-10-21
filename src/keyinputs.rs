use winit::event::{ElementState, KeyEvent};
use winit::keyboard::Key;
use winit::window::Window;

pub struct KeyInputHandler {
    is_fullscreen: bool,
    pub show_ui: bool,
}
impl KeyInputHandler {
    pub fn new() -> Self {
        Self {
            is_fullscreen: false,
            show_ui: true,
        }
    }
    pub fn handle_keyboard_input(&mut self, window: &Window, event: &KeyEvent) -> bool {
        if event.state == ElementState::Pressed && !event.repeat {
            if let Key::Character(ch) = &event.logical_key {
                match ch.as_str() {
                    "f" | "F" => {
                        self.toggle_fullscreen(window);
                        return true;
                    }
                    "h" | "H" => {
                        self.show_ui = !self.show_ui;
                        return true;
                    }
                    _ => {}
                }
            }
        }
        false
    }
    fn toggle_fullscreen(&mut self, window: &Window) {
        if !self.is_fullscreen {
            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        } else {
            window.set_fullscreen(None);
        }
        self.is_fullscreen = !self.is_fullscreen;
    }
}
