use image::ImageError;
use std::path::PathBuf;
use std::sync::mpsc;

#[derive(Debug)]
pub enum ExportError {
    IoError(std::io::Error),
    ImageError(ImageError),
}

impl From<std::io::Error> for ExportError {
    fn from(err: std::io::Error) -> Self {
        ExportError::IoError(err)
    }
}

impl From<ImageError> for ExportError {
    fn from(err: ImageError) -> Self {
        ExportError::ImageError(err)
    }
}
#[derive(Debug, Clone)]
pub struct ExportSettings {
    pub export_path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub start_time: f32,
    pub total_time: f32,
    pub fps: u32,
    pub is_exporting: bool,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            export_path: PathBuf::from("./export"),
            width: 1920,
            height: 1080,
            start_time: 0.0,
            total_time: 5.0,
            fps: 60,
            is_exporting: false,
        }
    }
}
#[derive(Clone)]
pub struct ExportUiRequest {
    pub width: u32,
    pub height: u32,
    pub start_time: f32,
    pub total_time: f32,
    pub fps: u32,
    pub path: PathBuf,
    pub is_exporting: bool,
}
#[derive(Default)]
pub struct ExportUiState {
    pub show_window: bool,
    pub temp_width: u32,
    pub temp_height: u32,
    pub temp_start_time: f32,
    pub temp_total_time: f32,
    pub temp_fps: u32,
}

/// Manages the export process and UI state
pub struct ExportManager {
    settings: ExportSettings,
    export_channel: Option<mpsc::Receiver<(u32, f32)>>,
    ui_state: ExportUiState,
    temp_state: TempExportState,
}

#[derive(Clone)]
struct TempExportState {
    width: u32,
    height: u32,
    start_time: f32,
    total_time: f32,
    fps: u32,
    path: PathBuf,
}

impl ExportManager {
    pub fn new() -> Self {
        let settings = ExportSettings::default();
        let ui_state = ExportUiState::default();
        let temp_state = TempExportState {
            width: settings.width,
            height: settings.height,
            start_time: settings.start_time,
            total_time: settings.total_time,
            fps: settings.fps,
            path: settings.export_path.clone(),
        };

        Self {
            settings,
            export_channel: None,
            ui_state,
            temp_state,
        }
    }
    pub fn get_ui_request(&self) -> ExportUiRequest {
        ExportUiRequest {
            width: self.temp_state.width,
            height: self.temp_state.height,
            start_time: self.temp_state.start_time,
            total_time: self.temp_state.total_time,
            fps: self.temp_state.fps,
            path: self.temp_state.path.clone(),
            is_exporting: self.settings.is_exporting,
        }
    }
    pub fn apply_ui_request(&mut self, request: ExportUiRequest) {
        self.temp_state.width = request.width;
        self.temp_state.height = request.height;
        self.temp_state.start_time = request.start_time;
        self.temp_state.total_time = request.total_time;
        self.temp_state.fps = request.fps;
        self.temp_state.path = request.path;
    }
    /// Returns a reference to the current export settings
    pub fn settings(&self) -> &ExportSettings {
        &self.settings
    }

    /// Returns whether an export is currently in progress
    pub fn is_exporting(&self) -> bool {
        self.settings.is_exporting
    }
    pub fn settings_mut(&mut self) -> &mut ExportSettings {
        &mut self.settings
    }
    /// Attempts to get the next frame for export
    pub fn try_get_next_frame(&mut self) -> Option<(u32, f32)> {
        self.export_channel.as_ref()?.try_recv().ok()
    }

    pub fn start_export(&mut self) {
        if self.settings.is_exporting {
            return;
        }

        // Apply the temporary state to settings before starting export
        self.settings.width = self.temp_state.width;
        self.settings.height = self.temp_state.height;
        self.settings.start_time = self.temp_state.start_time;
        self.settings.total_time = self.temp_state.total_time;
        self.settings.fps = self.temp_state.fps;
        self.settings.export_path = self.temp_state.path.clone();

        // Then start the export process
        self.settings.is_exporting = true;
        let settings = self.settings.clone();
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let total_frames = (settings.total_time * settings.fps as f32) as u32;

            for frame in 0..total_frames {
                let time = settings.start_time + (frame as f32 / settings.fps as f32);
                if tx.send((frame, time)).is_err() {
                    break;
                }
            }
        });

        self.export_channel = Some(rx);
    }

    /// Completes the export process
    pub fn complete_export(&mut self) {
        self.settings.is_exporting = false;
        self.export_channel = None;
    }

    /// Returns references to both UI state and settings for the UI to use
    pub fn get_ui_elements(&mut self) -> (&mut ExportUiState, &mut ExportSettings) {
        (&mut self.ui_state, &mut self.settings)
    }
    pub fn render_export_ui_widget(ui: &mut egui::Ui, request: &mut ExportUiRequest) -> bool {
        let mut should_start_export = false;

        ui.separator();
        ui.collapsing("Export", |ui| {
            if !request.is_exporting {
                // Resolution section
                ui.collapsing("Resolution", |ui| {
                    ui.add(
                        egui::DragValue::new(&mut request.width)
                            .range(1..=7680)
                            .prefix("Width: "),
                    );

                    ui.add(
                        egui::DragValue::new(&mut request.height)
                            .range(1..=4320)
                            .prefix("Height: "),
                    );
                });
                ui.collapsing("Time Settings", |ui| {
                    ui.add(
                        egui::DragValue::new(&mut request.start_time)
                            .prefix("Start Time: ")
                            .speed(0.1),
                    );

                    ui.add(
                        egui::DragValue::new(&mut request.total_time)
                            .prefix("Total Time: ")
                            .speed(0.1),
                    );

                    ui.add(
                        egui::DragValue::new(&mut request.fps)
                            .range(1..=240)
                            .prefix("FPS: "),
                    );
                });
                ui.collapsing("Output", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Export Path:");
                        if ui.button("Browse").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .set_directory(&request.path)
                                .pick_folder()
                            {
                                request.path = path;
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        let path_text = if let Some(path_str) = request.path.to_str() {
                            path_str
                        } else {
                            "Invalid path"
                        };
                        ui.add(egui::Label::new(
                            egui::RichText::new(path_text)
                                .monospace()
                                .weak()
                                .color(egui::Color32::from_rgb(150, 150, 150)),
                        ));
                    });
                    if !request.path.exists() {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("⚠ Path doesn't exist!")
                                    .color(egui::Color32::from_rgb(255, 190, 0)),
                            );
                        });
                    }
                });
                ui.separator();
                if ui.button("Start Export").clicked() {
                    should_start_export = true;
                }
            } else {
                ui.label("Exporting...");
            }
        });
        should_start_export
    }
    pub fn handle_export<F, E>(&mut self, capture_fn: F)
    where
        F: FnMut(u32, f32) -> Result<Vec<u8>, E>,
        E: std::fmt::Debug,
    {
        let mut capture_fn = capture_fn;
        if let Some((frame, time)) = self.try_get_next_frame() {
            match capture_fn(frame, time) {
                Ok(data) => {
                    let settings = self.settings();
                    if let Err(e) = save_frame(data, frame, settings) {
                        eprintln!("Error saving frame: {:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Error capturing frame: {:?}", e);
                }
            }
        } else {
            self.complete_export();
        }
    }
}
#[allow(unused_mut)]
pub fn save_frame(
    mut data: Vec<u8>,
    frame: u32,
    settings: &ExportSettings,
) -> Result<(), ExportError> {
    let frame_path = settings.export_path.join(format!("frame_{:05}.png", frame));

    if let Some(parent) = frame_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(target_os = "macos")]
    {
        for chunk in data.chunks_mut(4) {
            chunk.swap(0, 2);
        }
    }
    let image = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
        settings.width,
        settings.height,
        data,
    )
    .ok_or_else(|| {
        ImageError::Parameter(image::error::ParameterError::from_kind(
            image::error::ParameterErrorKind::Generic("Failed to create image buffer".to_string()),
        ))
    })?;

    image.save(&frame_path)?;
    Ok(())
}
