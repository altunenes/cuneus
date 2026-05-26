use std::collections::VecDeque;
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Clone, Debug)]
pub enum RemoteCommand {
    SetF32 { id: String, value: f32 },
    SetColor3 { id: String, value: [f32; 3] },
    Pulse { velocity: f32 },
    Note { pitch: f32, velocity: f32 },
    Transport { bpm: f32, beat: f32, measure: f32 },
}

#[derive(Clone, Default)]
pub struct RemoteControl {
    commands: Arc<Mutex<VecDeque<RemoteCommand>>>,
}

impl RemoteControl {
    pub fn from_env() -> Option<Self> {
        let port = std::env::var("CUNEUS_REMOTE_PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())?;
        Self::listen(port).ok()
    }

    pub fn listen(port: u16) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(("127.0.0.1", port))?;
        socket.set_nonblocking(false)?;

        let remote = Self::default();
        let commands = remote.commands.clone();

        thread::Builder::new()
            .name(format!("cuneus-remote-{port}"))
            .spawn(move || {
                let mut buffer = [0_u8; 1024];
                loop {
                    let Ok((len, _addr)) = socket.recv_from(&mut buffer) else {
                        continue;
                    };
                    let Ok(text) = std::str::from_utf8(&buffer[..len]) else {
                        continue;
                    };
                    if let Some(command) = parse_command(text.trim()) {
                        if let Ok(mut queue) = commands.lock() {
                            queue.push_back(command);
                        }
                    }
                }
            })?;

        Ok(remote)
    }

    pub fn drain(&self) -> Vec<RemoteCommand> {
        let Ok(mut queue) = self.commands.lock() else {
            return Vec::new();
        };
        queue.drain(..).collect()
    }
}

fn parse_command(text: &str) -> Option<RemoteCommand> {
    let mut parts = text.split_whitespace();
    match parts.next()? {
        "set_f32" => Some(RemoteCommand::SetF32 {
            id: parts.next()?.to_string(),
            value: parts.next()?.parse().ok()?,
        }),
        "set_color3" => Some(RemoteCommand::SetColor3 {
            id: parts.next()?.to_string(),
            value: [
                parts.next()?.parse().ok()?,
                parts.next()?.parse().ok()?,
                parts.next()?.parse().ok()?,
            ],
        }),
        "pulse" => Some(RemoteCommand::Pulse {
            velocity: parts.next()?.parse().ok()?,
        }),
        "note" => Some(RemoteCommand::Note {
            pitch: parts.next()?.parse().ok()?,
            velocity: parts.next()?.parse().ok()?,
        }),
        "transport" => Some(RemoteCommand::Transport {
            bpm: parts.next()?.parse().ok()?,
            beat: parts.next()?.parse().ok()?,
            measure: parts.next()?.parse().ok()?,
        }),
        _ => None,
    }
}
