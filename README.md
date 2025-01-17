[![Shader Binary Release](https://github.com/altunenes/cuneus/actions/workflows/release.yaml/badge.svg)](https://github.com/altunenes/cuneus/actions/workflows/release.yaml) [![crates.io](https://img.shields.io/crates/v/Cuneus.svg)](https://crates.io/crates/Cuneus)

# Cuneus 🌈

A tool for experimenting with WGSL shaders, it uses `wgpu` for rendering, `egui` for the UI, `winit` for windowing, and `notify` for hot-reload. :-)

### Current Features

- Hot shader reloading
- Multi-pass, atomics etc
- Interactive parameter adjustment, ez Texture loading through egui
- Export HQ frames via egui

## Current look

  <a href="https://github.com/user-attachments/assets/7eea9b94-875a-4e01-9204-3da978d3cd65">
    <img src="https://github.com/user-attachments/assets/7eea9b94-875a-4e01-9204-3da978d3cd65" width="300" alt="Cuneus IDE Interface"/>
  </a>

## Keys

- `F` full screen/minimal screen, `H` hide egui

#### Usage

- If you want to try your own shaders, check out the [usage.md](usage.md).

#### Open my shaders

- cargo run --release --bin *file*
- Or download on the [releases](https://github.com/altunenes/cuneus/releases)

# Gallery

| **Sinh** | **Signed Distance** | **Satan** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/adbb0938-1824-4024-b6aa-21d6fdde8b0d"><img src="https://github.com/user-attachments/assets/adbb0938-1824-4024-b6aa-21d6fdde8b0d" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/sinh.rs) | <a href="https://github.com/user-attachments/assets/1847c374-5719-4fee-b74d-3418e5fa4d7b"><img src="https://github.com/user-attachments/assets/1847c374-5719-4fee-b74d-3418e5fa4d7b" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/sdvert.rs) | <a href="https://github.com/user-attachments/assets/8f86a3b4-8d31-499f-b9fa-8b23266291ae"><img src="https://github.com/user-attachments/assets/8f86a3b4-8d31-499f-b9fa-8b23266291ae" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/satan.rs) |

| **Mandelbulb** | **Lich** | **Galaxy** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/2405334c-f13e-4d8d-863f-bab7dcc676ab"><img src="https://github.com/user-attachments/assets/2405334c-f13e-4d8d-863f-bab7dcc676ab" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/mandelbulb.rs) | <a href="https://github.com/user-attachments/assets/9589d2ec-43b8-4373-8dce-9cd2c74d862f"><img src="https://github.com/user-attachments/assets/9589d2ec-43b8-4373-8dce-9cd2c74d862f" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/lich.rs) | <a href="https://github.com/user-attachments/assets/a2647904-55bd-4912-9713-4558203ee6aa"><img src="https://github.com/user-attachments/assets/a2647904-55bd-4912-9713-4558203ee6aa" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/galaxy.rs) |

| **Xmas** | **Droste** | **Clifford** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/4f1f0cc0-12a5-4158-90e1-ac205fa2d28a"><img src="https://github.com/user-attachments/assets/4f1f0cc0-12a5-4158-90e1-ac205fa2d28a" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/xmas.rs) | <a href="https://github.com/user-attachments/assets/ffe1e193-9a9a-4784-8193-177d6b8648af"><img src="https://github.com/user-attachments/assets/ffe1e193-9a9a-4784-8193-177d6b8648af" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/droste.rs) | <a href="https://github.com/user-attachments/assets/42868686-bad9-4ce3-b5bd-346d880c8540"><img src="https://github.com/user-attachments/assets/42868686-bad9-4ce3-b5bd-346d880c8540" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/clifford.rs) |


| **orbits** | **dna** | **genuary6** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/8aadd685-e11b-4929-809b-61c950fc2a3d"><img src="https://github.com/user-attachments/assets/8aadd685-e11b-4929-809b-61c950fc2a3d" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/orbits.rs) | <a href="https://github.com/user-attachments/assets/fe88f9e3-de98-4b03-a3d5-e3219632a6df"><img src="https://github.com/user-attachments/assets/fe88f9e3-de98-4b03-a3d5-e3219632a6df" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/dna.rs) | <a href="https://github.com/user-attachments/assets/be2e132a-a473-462d-8b5b-2277336c7e78"><img src="https://github.com/user-attachments/assets/be2e132a-a473-462d-8b5b-2277336c7e78" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/genuary2025_6.rs) |
