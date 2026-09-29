use winit::event::{ElementState, KeyEvent};
use winit::keyboard::Key;
use winit::window::Window;

pub struct KeyInputHandler {
    is_fullscreen: bool,
    pub show_ui: bool,
    /// Toggles fullscreen (F by default); `None` frees the key
    pub fullscreen_key: Option<Key>,
    /// Shows / hides the UI (H by default); `None` frees the key
    pub ui_key: Option<Key>,
}
impl Default for KeyInputHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyInputHandler {
    pub fn new() -> Self {
        Self {
            is_fullscreen: false,
            show_ui: true,
            fullscreen_key: Some(Key::Character("f".into())),
            ui_key: Some(Key::Character("h".into())),
        }
    }
    pub fn handle_keyboard_input(&mut self, window: &Window, event: &KeyEvent) -> bool {
        if event.state != ElementState::Pressed || event.repeat {
            return false;
        }
        if matches_key(&self.fullscreen_key, &event.logical_key) {
            self.toggle_fullscreen(window);
            return true;
        }
        if matches_key(&self.ui_key, &event.logical_key) {
            self.show_ui = !self.show_ui;
            return true;
        }
        false
    }
    pub fn toggle_fullscreen(&mut self, window: &Window) {
        if !self.is_fullscreen {
            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        } else {
            window.set_fullscreen(None);
        }
        self.is_fullscreen = !self.is_fullscreen;
    }
    pub fn is_fullscreen(&self) -> bool {
        self.is_fullscreen
    }
}

// letters match either case
fn matches_key(bound: &Option<Key>, pressed: &Key) -> bool {
    match (bound, pressed) {
        (Some(Key::Character(a)), Key::Character(b)) => a.eq_ignore_ascii_case(b),
        (Some(k), p) => k == p,
        (None, _) => false,
    }
}
