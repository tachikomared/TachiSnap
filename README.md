# TachiSnap

Pixel Snapper for animation pixel artists.

Made by TachikomaRed and smolemaru.

TachiSnap is a free, client-side Rust + WebAssembly tool for cleaning up AI-generated pixel art. Images are processed locally in the browser; they are not uploaded to a server.

## What it does

AI image models often produce pixel art with uneven blocks, drifting grids, fuzzy colors, or unwanted backgrounds. TachiSnap cleans that up by:

- snapping visual pixels to a uniform grid
- quantizing colors to a strict palette with k-means++ in CIELAB color space
- optionally removing backgrounds with flood fill or global color removal
- upscaling with crisp nearest-neighbor scaling
- snapping animated GIF frames
- converting sprite sheets into animated GIFs
- bulk processing multiple images or a selected/dropped folder

Supported input formats: PNG, JPEG, GIF, BMP, and WebP.

Output formats: PNG for still images, GIF for animated output, and ZIP for bulk downloads.

## Local development

### Prerequisites

- Rust
- wasm-pack

### Build

```bash
./build.sh
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

The build script compiles the Rust library to WebAssembly and copies the static web files into the project root.

## CLI usage

```bash
cargo run --release -- snap input.png output.png --k 16 --pixel-size 4 --upscale 4
cargo run --release -- animate sheet.png output.gif --cols 4 --rows 2 --fps 12
cargo run --release -- bulk ./ai-art ./clean-art --recursive --k 16 --upscale 4 --remove-bg --json
```

Use `--remove-bg`, `--bg-tolerance`, `--bg-mode flood|global`, and `--bg-color "#RRGGBB"` for background removal.

## Agentic native usage

TachiSnap includes a folder-oriented CLI command for Codex, Claude Code, and other local coding agents:

```bash
tachi-snap bulk ./input-ai-pixel-art ./output-clean \
  --recursive \
  --k 16 \
  --pixel-size 0 \
  --upscale 4 \
  --remove-bg \
  --bg-mode flood \
  --json
```

The command:

- scans PNG, JPEG, GIF, BMP, and WebP files
- preserves nested folder structure in the output folder
- writes still images as PNG and animated GIFs as GIF
- suffixes outputs with `_tachisnap`
- prints a machine-readable JSON report with per-file status, output paths, sizes, and errors

Example agent prompt:

```text
Use TachiSnap to clean every AI-generated pixel art file in ./raw-art and put the results in ./clean-art. Use recursive mode and return the JSON summary.
```

## Deployment

This is a static app after build. Host the generated HTML, JavaScript, and WASM files on any static host, including GitHub Pages or Vercel.

For GitHub Pages, the included workflow builds WASM and publishes the static artifact on pushes to `main`.

For GitHub Releases, push a version tag:

```bash
git tag v0.3.1
git push origin v0.3.1
```

The release workflow publishes:

- a static web app archive
- native `tachi-snap` CLI ZIPs for Linux, Windows, and macOS

For Vercel, use a build command equivalent to:

```bash
wasm-pack build --target web --out-dir pkg --release && mkdir -p dist && cp -r web/* dist/ && cp -r pkg dist/pkg
```

Set the output directory to `dist`.

## Algorithm

1. Color quantization reduces the image to a fixed palette.
2. Gradient profiles estimate the pixel grid.
3. A grid walker snaps cuts to strong edges.
4. Stabilization smooths inconsistent rows and columns.
5. Majority-vote resampling produces clean pixel cells.
6. Optional nearest-neighbor scaling enlarges the result without blur.

## Credits

Forked from [Hugo-Dz/spritefusion-pixel-snapper](https://github.com/Hugo-Dz/spritefusion-pixel-snapper), MIT License.
