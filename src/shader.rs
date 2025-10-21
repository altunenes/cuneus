use crate::Core;
use winit::event::WindowEvent;

pub trait ShaderManager {
    fn init(core: &Core) -> Self
    where
        Self: Sized;
    fn resize(&mut self, _core: &Core) {}
    fn update(&mut self, _core: &Core) {}
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError>;
    fn handle_input(&mut self, _core: &Core, _event: &WindowEvent) -> bool {
        false
    }
}
