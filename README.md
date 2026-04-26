# 🎮 Pixel Snapper

> Snap messy AI-generated pixel art to a perfect grid — in your browser.

[![Deploy to GitHub Pages](https://github.com/YOUR_USERNAME/pixel-snapper/actions/workflows/deploy.yml/badge.svg)](https://github.com/YOUR_USERNAME/pixel-snapper/actions)

**[Live Demo →](https://YOUR_USERNAME.github.io/pixel-snapper)**

![Screenshot](static/hero.png)

---

## What it does

Current AI image models can't understand grid-based pixel art:

- Pixels are inconsistent in size and position
- The grid resolution drifts over time
- Colors aren't tied to a strict palette

**Pixel Snapper fixes this:**

- ✅ Pixels snapped to a perfect uniform grid
- ✅ Colors quantized to a strict palette (k-means++)
- ✅ Optional nearest-neighbor upscale (1×, 2×, 4×, 8×)
- ✅ Runs 100% in-browser via WebAssembly

---

## Improvements over original

| Feature | Original | This fork |
|---------|----------|-----------|
| Forced pixel size | ❌ | ✅ Manual override slider |
| Upscale output | ❌ | ✅ 1×/2×/4×/8× |
| Detect pixel size (WASM) | ❌ | ✅ `detect_pixel_size()` |
| Web UI | ❌ (external) | ✅ Included |
| GitHub Pages deploy | ❌ | ✅ Auto CI/CD |
| Image formats | PNG/JPEG | PNG/JPEG/GIF/BMP/WebP |

---

## Local development

### Prerequisites

- [Rust](https://rustup.rs/)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/)

### Build

```bash
git clone https://github.com/YOUR_USERNAME/pixel-snapper
cd pixel-snapper
./build.sh
python3 -m http.server 8080
# open http://localhost:8080
```

### CLI usage

```bash
cargo run --release -- input.png output.png [k_colors] [pixel_size] [upscale]

# Examples:
cargo run --release -- sprite.png fixed.png          # auto settings
cargo run --release -- sprite.png fixed.png 16       # 16-color palette
cargo run --release -- sprite.png fixed.png 16 4     # force 4px pixel size
cargo run --release -- sprite.png fixed.png 16 4 4   # + 4× upscale
```

---

## Deploy to GitHub Pages

1. Fork / push to GitHub
2. Go to **Settings → Pages → Source**: set to **GitHub Actions**
3. Push to `main` — the CI workflow builds WASM and deploys automatically

---

## Algorithm

1. **Color quantization** — k-means++ reduces image to `k` colors
2. **Gradient profiles** — horizontal/vertical Sobel edge sums detect grid lines
3. **Peak estimation** — median peak distance gives pixel size
4. **Grid walker** — elastic walker snaps cuts to strongest edges
5. **Stabilization** — two-pass cross-axis validation fixes skewed grids
6. **Resample** — majority-vote per cell produces clean output
7. **Upscale** — optional nearest-neighbor scaling

---

## Credits

Forked from [Hugo-Dz/spritefusion-pixel-snapper](https://github.com/Hugo-Dz/spritefusion-pixel-snapper) — MIT License.

Original project by [Hugo Duprez](https://www.hugoduprez.com/) / [Sprite Fusion](https://spritefusion.com).
