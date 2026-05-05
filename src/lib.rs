#![allow(dead_code)]

use image::{GenericImageView, ImageBuffer, Rgba, RgbaImage};
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, WeightedIndex};
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fmt;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// ─── Config ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Config {
    pub k_colors: usize,
    k_seed: u64,
    #[allow(dead_code)]
    input_path: String,
    #[allow(dead_code)]
    output_path: String,
    max_kmeans_iterations: usize,
    peak_threshold_multiplier: f64,
    peak_distance_filter: usize,
    walker_search_window_ratio: f64,
    walker_min_search_window: f64,
    walker_strength_threshold: f64,
    min_cuts_per_axis: usize,
    fallback_target_segments: usize,
    max_step_ratio: f64,
    pub forced_pixel_size: usize,
    pub upscale_factor: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            k_colors: 16,
            k_seed: 42,
            input_path: String::new(),
            output_path: String::new(),
            max_kmeans_iterations: 25,
            peak_threshold_multiplier: 0.2,
            peak_distance_filter: 4,
            walker_search_window_ratio: 0.35,
            walker_min_search_window: 2.0,
            walker_strength_threshold: 0.5,
            min_cuts_per_axis: 4,
            fallback_target_segments: 64,
            max_step_ratio: 1.8,
            forced_pixel_size: 0,
            upscale_factor: 1,
        }
    }
}

// ─── Errors ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum PixelSnapperError {
    ImageError(image::ImageError),
    InvalidInput(String),
    ProcessingError(String),
}

impl fmt::Display for PixelSnapperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PixelSnapperError::ImageError(e) => write!(f, "Image error: {}", e),
            PixelSnapperError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            PixelSnapperError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl Error for PixelSnapperError {}

impl From<image::ImageError> for PixelSnapperError {
    fn from(error: image::ImageError) -> Self {
        PixelSnapperError::ImageError(error)
    }
}

#[cfg(target_arch = "wasm32")]
impl From<PixelSnapperError> for wasm_bindgen::JsValue {
    fn from(err: PixelSnapperError) -> wasm_bindgen::JsValue {
        wasm_bindgen::JsValue::from_str(&err.to_string())
    }
}

type Result<T> = std::result::Result<T, PixelSnapperError>;

// ─── WASM public API ────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn detect_pixel_size(input_bytes: &[u8]) -> std::result::Result<u32, wasm_bindgen::JsValue> {
    let config = Config::default();
    let img = image::load_from_memory(input_bytes).map_err(PixelSnapperError::from)?;
    let (width, height) = img.dimensions();
    validate_image_dimensions(width, height)?;
    let rgba_img = img.to_rgba8();
    let quantized = quantize_image(&rgba_img, &config)?;
    let (px, py) = compute_profiles(&quantized, &config)?;
    let sx = estimate_step_size(&px, &config);
    let sy = estimate_step_size(&py, &config);
    let (step_x, step_y) = resolve_step_sizes(sx, sy, width, height, &config);
    let detected = ((step_x + step_y) / 2.0).round() as u32;
    Ok(detected.max(1))
}

/// Process image with full options.
/// remove_bg: auto-detect and remove background before snapping
/// bg_tolerance: color distance tolerance for BG removal (0–80, default 20)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn process_image(
    input_bytes: &[u8],
    k_colors: Option<u32>,
    forced_pixel_size: Option<u32>,
    upscale_factor: Option<u32>,
    remove_bg: Option<bool>,
    bg_tolerance: Option<u8>,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let mut config = Config::default();

    if let Some(k) = k_colors {
        if k == 0 {
            return Err(wasm_bindgen::JsValue::from_str("k_colors must be > 0"));
        }
        config.k_colors = k as usize;
    }
    if let Some(ps) = forced_pixel_size {
        config.forced_pixel_size = ps as usize;
    }
    if let Some(uf) = upscale_factor {
        config.upscale_factor = (uf as usize).max(1).min(16);
    }

    let do_remove_bg = remove_bg.unwrap_or(false);
    let tolerance = bg_tolerance.unwrap_or(20);

    process_image_bytes_common(input_bytes, Some(config), do_remove_bg, tolerance)
        .map_err(|e| wasm_bindgen::JsValue::from(e))
}

/// Remove background from an image by flood-filling transparent from edges.
/// Returns PNG bytes with background pixels set to alpha=0.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn remove_background(
    input_bytes: &[u8],
    tolerance: Option<u8>,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let img = image::load_from_memory(input_bytes).map_err(PixelSnapperError::from)?;
    let (w, h) = img.dimensions();
    validate_image_dimensions(w, h)?;
    let mut rgba = img.to_rgba8();
    if let Some(bg) = detect_background_color(&rgba) {
        remove_background_flood(&mut rgba, bg, tolerance.unwrap_or(20));
    }
    let mut buf = Vec::new();
    rgba.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .map_err(PixelSnapperError::ImageError)?;
    Ok(buf)
}

/// Detect the sprite grid layout of a sprite sheet.
/// Returns JSON: {"cols":4,"rows":2,"frame_w":32,"frame_h":32,"pixel_size":4,"sheet_w":128,"sheet_h":64}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn detect_sprite_grid(
    input_bytes: &[u8],
) -> std::result::Result<String, wasm_bindgen::JsValue> {
    let img = image::load_from_memory(input_bytes).map_err(PixelSnapperError::from)?;
    let (w, h) = img.dimensions();
    validate_image_dimensions(w, h)?;
    let rgba = img.to_rgba8();
    let config = Config::default();

    let quantized = quantize_image(&rgba, &config)?;
    let (px, py) = compute_profiles(&quantized, &config)?;
    let sx = estimate_step_size(&px, &config);
    let sy = estimate_step_size(&py, &config);
    let (step_x, step_y) = resolve_step_sizes(sx, sy, w, h, &config);

    let cols = (w as f64 / step_x).round() as u32;
    let rows = (h as f64 / step_y).round() as u32;
    let cols = cols.max(1);
    let rows = rows.max(1);
    let frame_w = w / cols;
    let frame_h = h / rows;
    let pixel_size = ((step_x + step_y) / 2.0).round() as u32;

    Ok(format!(
        r#"{{"cols":{},"rows":{},"frame_w":{},"frame_h":{},"pixel_size":{},"sheet_w":{},"sheet_h":{}}}"#,
        cols, rows, frame_w, frame_h, pixel_size, w, h
    ))
}

/// Convert a sprite sheet to an animated GIF.
/// Snaps each frame through the full pipeline, then encodes as animated GIF.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn spritesheet_to_gif(
    input_bytes: &[u8],
    cols: u32,
    rows: u32,
    fps: u32,
    k_colors: Option<u32>,
    forced_pixel_size: Option<u32>,
    upscale_factor: Option<u32>,
    remove_bg: Option<bool>,
    bg_tolerance: Option<u8>,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let mut config = Config::default();
    if let Some(k) = k_colors {
        if k > 0 {
            config.k_colors = k as usize;
        }
    }
    if let Some(ps) = forced_pixel_size {
        config.forced_pixel_size = ps as usize;
    }
    if let Some(uf) = upscale_factor {
        config.upscale_factor = (uf as usize).max(1).min(16);
    }

    let do_remove_bg = remove_bg.unwrap_or(false);
    let tolerance = bg_tolerance.unwrap_or(20);

    let img = image::load_from_memory(input_bytes).map_err(PixelSnapperError::from)?;
    let (w, h) = img.dimensions();
    validate_image_dimensions(w, h)?;
    let mut rgba = img.to_rgba8();

    if do_remove_bg {
        if let Some(bg) = detect_background_color(&rgba) {
            remove_background_flood(&mut rgba, bg, tolerance);
        }
    }

    let raw_frames =
        split_spritesheet(&rgba, cols, rows).map_err(wasm_bindgen::JsValue::from)?;

    let mut snapped_frames: Vec<RgbaImage> = Vec::with_capacity(raw_frames.len());
    for frame in &raw_frames {
        let snapped =
            snap_frame(frame, &config).map_err(wasm_bindgen::JsValue::from)?;
        snapped_frames.push(snapped);
    }

    frames_to_gif(&snapped_frames, fps, 0).map_err(wasm_bindgen::JsValue::from)
}

/// Returns the library version string.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn version() -> String {
    "0.3.1".to_string()
}

/// Pack already-rendered frames (from a canvas sprite sheet) into an animated GIF.
/// The sheet is split into cols×rows cells and each cell becomes one frame.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn encode_gif_from_sheet(
    input_bytes: &[u8],
    cols: u32,
    rows: u32,
    fps: u32,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let img = image::load_from_memory(input_bytes).map_err(PixelSnapperError::from)?;
    let (w, h) = img.dimensions();
    validate_image_dimensions(w, h)?;
    let rgba = img.to_rgba8();
    let frames = split_spritesheet(&rgba, cols, rows).map_err(wasm_bindgen::JsValue::from)?;
    frames_to_gif(&frames, fps, 0).map_err(wasm_bindgen::JsValue::from)
}

/// Snap each frame of an animated GIF and return a new snapped GIF.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn snap_gif(
    input_bytes: &[u8],
    k_colors: Option<u32>,
    forced_pixel_size: Option<u32>,
    upscale_factor: Option<u32>,
    remove_bg: Option<bool>,
    bg_tolerance: Option<u8>,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let mut config = Config::default();
    if let Some(k) = k_colors {
        if k > 0 { config.k_colors = k as usize; }
    }
    if let Some(ps) = forced_pixel_size {
        config.forced_pixel_size = ps as usize;
    }
    if let Some(uf) = upscale_factor {
        config.upscale_factor = (uf as usize).max(1).min(16);
    }
    let do_remove_bg = remove_bg.unwrap_or(false);
    let tolerance = bg_tolerance.unwrap_or(20);
    snap_gif_internal(input_bytes, &config, do_remove_bg, tolerance)
        .map_err(wasm_bindgen::JsValue::from)
}

// ─── CLI ────────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
use clap::{Parser, Subcommand};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[command(
    name = "pixel-snapper",
    version = "0.3.0",
    about = "Snap AI-generated pixel art to a perfect grid"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    /// Output JSON to stdout for agent/script use
    #[arg(long, global = true)]
    json: bool,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Subcommand)]
enum Commands {
    /// Snap a single image to a pixel grid
    Snap {
        input: String,
        output: String,
        /// Palette size (number of colors)
        #[arg(long, default_value_t = 16)]
        k: usize,
        /// Pixel size override (0 = auto-detect)
        #[arg(long, default_value_t = 0)]
        pixel_size: usize,
        /// Upscale factor (1–16)
        #[arg(long, default_value_t = 1)]
        upscale: usize,
        /// Auto-remove background
        #[arg(long)]
        remove_bg: bool,
        /// Background removal tolerance (0–80)
        #[arg(long, default_value_t = 20)]
        bg_tolerance: u8,
    },
    /// Convert a sprite sheet to an animated GIF
    Animate {
        input: String,
        output: String,
        /// Number of columns in the sprite sheet
        #[arg(long)]
        cols: u32,
        /// Number of rows in the sprite sheet
        #[arg(long)]
        rows: u32,
        /// Frames per second
        #[arg(long, default_value_t = 12)]
        fps: u32,
        /// Palette size per frame
        #[arg(long, default_value_t = 16)]
        k: usize,
        /// Pixel size override (0 = auto-detect)
        #[arg(long, default_value_t = 0)]
        pixel_size: usize,
        /// Upscale factor (1–16)
        #[arg(long, default_value_t = 1)]
        upscale: usize,
        /// Auto-remove background
        #[arg(long)]
        remove_bg: bool,
        /// Background removal tolerance (0–80)
        #[arg(long, default_value_t = 20)]
        bg_tolerance: u8,
    },
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let cli = Cli::parse();
        if let Err(e) = run_cli(cli) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_cli(cli: Cli) -> Result<()> {
    let use_json = cli.json;
    match cli.command {
        Commands::Snap {
            input,
            output,
            k,
            pixel_size,
            upscale,
            remove_bg,
            bg_tolerance,
        } => {
            let mut config = Config {
                input_path: input.clone(),
                output_path: output.clone(),
                ..Default::default()
            };
            if k > 0 {
                config.k_colors = k;
            }
            config.forced_pixel_size = pixel_size;
            config.upscale_factor = upscale.max(1).min(16);

            if !use_json {
                println!("Snapping: {}", input);
            }

            let t0 = std::time::Instant::now();
            let img_bytes = std::fs::read(&input).map_err(|e| {
                PixelSnapperError::ProcessingError(format!("Failed to read: {}", e))
            })?;
            let output_bytes =
                process_image_bytes_common(&img_bytes, Some(config.clone()), remove_bg, bg_tolerance)?;

            // Detect pixel size for reporting
            let detected_px = {
                let img = image::load_from_memory(&img_bytes)?;
                let (w, h) = img.dimensions();
                let rgba = img.to_rgba8();
                let q = quantize_image(&rgba, &config)?;
                let (px, py) = compute_profiles(&q, &config)?;
                let sx = estimate_step_size(&px, &config);
                let sy = estimate_step_size(&py, &config);
                let (sx, sy) = resolve_step_sizes(sx, sy, w, h, &config);
                ((sx + sy) / 2.0).round() as u32
            };

            let out_img = image::load_from_memory(&output_bytes)?;
            let (ow, oh) = out_img.dimensions();
            let elapsed = t0.elapsed().as_millis();

            std::fs::write(&output, &output_bytes).map_err(|e| {
                PixelSnapperError::ProcessingError(format!("Failed to write: {}", e))
            })?;

            if use_json {
                println!(
                    r#"{{"status":"ok","output":"{}","pixel_size":{},"output_w":{},"output_h":{},"elapsed_ms":{}}}"#,
                    output, detected_px, ow, oh, elapsed
                );
            } else {
                println!(
                    "Done: {} → {} ({}×{}, pixel size {}px, {}ms)",
                    input, output, ow, oh, detected_px, elapsed
                );
            }
            Ok(())
        }

        Commands::Animate {
            input,
            output,
            cols,
            rows,
            fps,
            k,
            pixel_size,
            upscale,
            remove_bg,
            bg_tolerance,
        } => {
            let mut config = Config {
                input_path: input.clone(),
                output_path: output.clone(),
                ..Default::default()
            };
            if k > 0 {
                config.k_colors = k;
            }
            config.forced_pixel_size = pixel_size;
            config.upscale_factor = upscale.max(1).min(16);

            if !use_json {
                println!("Animating: {} ({}×{} grid, {}fps)", input, cols, rows, fps);
            }

            let t0 = std::time::Instant::now();
            let img_bytes = std::fs::read(&input).map_err(|e| {
                PixelSnapperError::ProcessingError(format!("Failed to read: {}", e))
            })?;

            let img = image::load_from_memory(&img_bytes)?;
            let (w, h) = img.dimensions();
            validate_image_dimensions(w, h)?;
            let mut rgba = img.to_rgba8();

            if remove_bg {
                if let Some(bg) = detect_background_color(&rgba) {
                    remove_background_flood(&mut rgba, bg, bg_tolerance);
                }
            }

            let raw_frames = split_spritesheet(&rgba, cols, rows)?;
            let frame_count = raw_frames.len();

            let mut snapped_frames: Vec<RgbaImage> = Vec::with_capacity(frame_count);
            for frame in &raw_frames {
                snapped_frames.push(snap_frame(frame, &config)?);
            }

            let gif_bytes = frames_to_gif(&snapped_frames, fps, 0)?;
            let elapsed = t0.elapsed().as_millis();

            std::fs::write(&output, &gif_bytes).map_err(|e| {
                PixelSnapperError::ProcessingError(format!("Failed to write: {}", e))
            })?;

            let (fw, fh) = snapped_frames[0].dimensions();

            if use_json {
                println!(
                    r#"{{"status":"ok","output":"{}","frames":{},"frame_w":{},"frame_h":{},"fps":{},"size_bytes":{},"elapsed_ms":{}}}"#,
                    output, frame_count, fw, fh, fps, gif_bytes.len(), elapsed
                );
            } else {
                println!(
                    "Done: {} → {} ({} frames, {}×{}px, {}fps, {}KB, {}ms)",
                    input,
                    output,
                    frame_count,
                    fw,
                    fh,
                    fps,
                    gif_bytes.len() / 1024,
                    elapsed
                );
            }
            Ok(())
        }
    }
}

// ─── Core pipeline ──────────────────────────────────────────────────────────

fn process_image_bytes_common(
    input_bytes: &[u8],
    config: Option<Config>,
    remove_bg: bool,
    bg_tolerance: u8,
) -> Result<Vec<u8>> {
    let config = config.unwrap_or_default();

    let img = image::load_from_memory(input_bytes)?;
    let (width, height) = img.dimensions();
    validate_image_dimensions(width, height)?;

    let mut rgba_img = img.to_rgba8();

    // Step 0: background removal before quantization
    if remove_bg {
        if let Some(bg) = detect_background_color(&rgba_img) {
            remove_background_flood(&mut rgba_img, bg, bg_tolerance);
        }
    }

    let final_img = snap_frame(&rgba_img, &config)?;

    let mut output_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut output_bytes);
    final_img
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(PixelSnapperError::ImageError)?;

    Ok(output_bytes)
}

/// Run the full snap pipeline on a single frame/image.
fn snap_frame(rgba_img: &RgbaImage, config: &Config) -> Result<RgbaImage> {
    let (width, height) = rgba_img.dimensions();
    let quantized_img = quantize_image(rgba_img, config)?;

    let (profile_x, profile_y) = compute_profiles(&quantized_img, config)?;

    let (step_x, step_y) = if config.forced_pixel_size > 0 {
        let s = config.forced_pixel_size as f64;
        (s, s)
    } else {
        let sx = estimate_step_size(&profile_x, config);
        let sy = estimate_step_size(&profile_y, config);
        resolve_step_sizes(sx, sy, width, height, config)
    };

    let raw_col_cuts = walk(&profile_x, step_x, width as usize, config)?;
    let raw_row_cuts = walk(&profile_y, step_y, height as usize, config)?;

    let (col_cuts, row_cuts) = stabilize_both_axes(
        &profile_x,
        &profile_y,
        raw_col_cuts,
        raw_row_cuts,
        width as usize,
        height as usize,
        config,
    );

    let output_img = resample(&quantized_img, &col_cuts, &row_cuts)?;

    let final_img = if config.upscale_factor > 1 {
        upscale_nearest(&output_img, config.upscale_factor)
    } else {
        output_img
    };

    Ok(final_img)
}

// ─── CIELAB color space ──────────────────────────────────────────────────────

#[inline]
fn rgb_to_lab(r: f32, g: f32, b: f32) -> [f32; 3] {
    // sRGB → linear (gamma expand)
    fn lin(c: f32) -> f32 {
        let c = c / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055_f32).powf(2.4)
        }
    }
    let (rl, gl, bl) = (lin(r), lin(g), lin(b));

    // sRGB (D65) → XYZ using IEC 61966-2-1 matrix
    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;

    // Normalize by D65 white point
    let xn = x / 0.95047;
    let yn = y / 1.00000;
    let zn = z / 1.08883;

    fn f(t: f32) -> f32 {
        if t > 0.008856 {
            t.powf(1.0 / 3.0)
        } else {
            7.787 * t + 16.0 / 116.0
        }
    }
    let (fx, fy, fz) = (f(xn), f(yn), f(zn));

    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

#[inline]
fn lab_to_rgb(lab: &[f32; 3]) -> (u8, u8, u8) {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = lab[1] / 500.0 + fy;
    let fz = fy - lab[2] / 200.0;

    fn finv(t: f32) -> f32 {
        let t3 = t * t * t;
        if t3 > 0.008856 {
            t3
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    }

    let x = finv(fx) * 0.95047;
    let y = finv(fy) * 1.00000;
    let z = finv(fz) * 1.08883;

    // XYZ → linear sRGB
    let rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

    fn gamma(c: f32) -> u8 {
        let c = c.clamp(0.0, 1.0);
        let g = if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (g * 255.0).round() as u8
    }
    (gamma(rl), gamma(gl), gamma(bl))
}

#[inline]
fn lab_dist_sq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dl = a[0] - b[0];
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    dl * dl + da * da + db * db
}

// ─── Upscale (nearest-neighbor) ─────────────────────────────────────────────

fn upscale_nearest(img: &RgbaImage, factor: usize) -> RgbaImage {
    let (w, h) = img.dimensions();
    let nw = w * factor as u32;
    let nh = h * factor as u32;
    let mut out = ImageBuffer::new(nw, nh);
    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y);
            for dy in 0..factor as u32 {
                for dx in 0..factor as u32 {
                    out.put_pixel(x * factor as u32 + dx, y * factor as u32 + dy, *px);
                }
            }
        }
    }
    out
}

// ─── Validation ─────────────────────────────────────────────────────────────

fn validate_image_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Image dimensions cannot be zero".to_string(),
        ));
    }
    if width > 10000 || height > 10000 {
        return Err(PixelSnapperError::InvalidInput(
            "Image too large (max 10000×10000)".to_string(),
        ));
    }
    Ok(())
}

// ─── Color quantization (k-means++ in CIELAB) ────────────────────────────────

fn quantize_image(img: &RgbaImage, config: &Config) -> Result<RgbaImage> {
    if config.k_colors == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "k_colors must be > 0".to_string(),
        ));
    }

    // Collect opaque pixels as LAB values for perceptual quantization
    let opaque_pixels: Vec<[f32; 3]> = img
        .pixels()
        .filter_map(|p| {
            if p[3] == 0 {
                None
            } else {
                Some(rgb_to_lab(p[0] as f32, p[1] as f32, p[2] as f32))
            }
        })
        .collect();

    let n_pixels = opaque_pixels.len();
    if n_pixels == 0 {
        return Ok(img.clone());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(config.k_seed);
    let k = config.k_colors.min(n_pixels);

    // k-means++ init
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    let first_idx = rng.gen_range(0..n_pixels);
    centroids.push(opaque_pixels[first_idx]);
    let mut distances = vec![f32::MAX; n_pixels];

    for _ in 1..k {
        let last_c = centroids.last().unwrap();
        let mut sum_sq = 0.0f32;
        for (i, p) in opaque_pixels.iter().enumerate() {
            let d = lab_dist_sq(p, last_c);
            if d < distances[i] {
                distances[i] = d;
            }
            sum_sq += distances[i];
        }
        if sum_sq <= 0.0 {
            centroids.push(opaque_pixels[rng.gen_range(0..n_pixels)]);
        } else {
            let dist = WeightedIndex::new(&distances).map_err(|e| {
                PixelSnapperError::ProcessingError(format!("k-means++ sample failed: {}", e))
            })?;
            centroids.push(opaque_pixels[dist.sample(&mut rng)]);
        }
    }

    // Lloyd iterations
    let mut prev_centroids = centroids.clone();
    for iteration in 0..config.max_kmeans_iterations {
        let mut sums = vec![[0.0f32; 3]; k];
        let mut counts = vec![0usize; k];

        for p in &opaque_pixels {
            let best_k = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    lab_dist_sq(p, a)
                        .partial_cmp(&lab_dist_sq(p, b))
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            sums[best_k][0] += p[0];
            sums[best_k][1] += p[1];
            sums[best_k][2] += p[2];
            counts[best_k] += 1;
        }

        for i in 0..k {
            if counts[i] > 0 {
                let fc = counts[i] as f32;
                centroids[i] = [sums[i][0] / fc, sums[i][1] / fc, sums[i][2] / fc];
            }
        }

        if iteration > 0 {
            let max_move = centroids
                .iter()
                .zip(prev_centroids.iter())
                .map(|(n, o)| lab_dist_sq(n, o))
                .fold(0.0f32, f32::max);
            // 0.1 threshold in LAB space ≈ imperceptible color shift
            if max_move < 0.1 {
                break;
            }
        }

        prev_centroids.copy_from_slice(&centroids);
    }

    // Map every pixel to nearest centroid, convert LAB centroid back to RGB
    let mut new_img = RgbaImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel[3] == 0 {
            new_img.put_pixel(x, y, *pixel);
            continue;
        }
        let p = rgb_to_lab(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
        let best = centroids
            .iter()
            .min_by(|a, b| {
                lab_dist_sq(&p, a)
                    .partial_cmp(&lab_dist_sq(&p, b))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap();
        let (cr, cg, cb) = lab_to_rgb(best);
        new_img.put_pixel(x, y, Rgba([cr, cg, cb, pixel[3]]));
    }
    Ok(new_img)
}

// ─── Background removal ───────────────────────────────────────────────────────

/// Sample 4 corners (3×3 patch each) to find the dominant background color.
/// Returns None when all corners are transparent (e.g. pre-cut sprites).
fn detect_background_color(img: &RgbaImage) -> Option<[u8; 4]> {
    let (w, h) = img.dimensions();
    let patch = 3_u32.min(w).min(h);
    let corners = [
        (0u32, 0u32),
        (w.saturating_sub(patch), 0),
        (0, h.saturating_sub(patch)),
        (w.saturating_sub(patch), h.saturating_sub(patch)),
    ];
    let mut counts: HashMap<[u8; 4], usize> = HashMap::new();
    for (cx, cy) in corners {
        for dy in 0..patch {
            for dx in 0..patch {
                let px = img.get_pixel(cx + dx, cy + dy).0;
                if px[3] > 0 {
                    *counts.entry(px).or_insert(0) += 1;
                }
            }
        }
    }
    counts.into_iter().max_by_key(|(_, c)| *c).map(|(c, _)| c)
}

#[inline]
fn color_distance_sq(a: &[u8; 4], b: &[u8; 4]) -> u32 {
    let dr = a[0] as i32 - b[0] as i32;
    let dg = a[1] as i32 - b[1] as i32;
    let db = a[2] as i32 - b[2] as i32;
    (dr * dr + dg * dg + db * db) as u32
}

/// BFS flood-fill from all four edges; pixels within tolerance become transparent.
fn remove_background_flood(img: &mut RgbaImage, bg_color: [u8; 4], tolerance: u8) {
    let (w, h) = img.dimensions();
    // tol*tol*3 scales single-channel tolerance to 3-channel Euclidean squared
    let tol_sq = (tolerance as u32) * (tolerance as u32) * 3;
    let mut visited = vec![false; (w * h) as usize];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();

    for x in 0..w {
        queue.push_back((x, 0));
        queue.push_back((x, h - 1));
    }
    for y in 1..h.saturating_sub(1) {
        queue.push_back((0, y));
        queue.push_back((w - 1, y));
    }

    while let Some((x, y)) = queue.pop_front() {
        let idx = (y * w + x) as usize;
        if visited[idx] {
            continue;
        }
        visited[idx] = true;
        let px = img.get_pixel(x, y).0;
        if px[3] == 0 || color_distance_sq(&px, &bg_color) > tol_sq {
            continue;
        }
        img.put_pixel(x, y, Rgba([px[0], px[1], px[2], 0]));
        if x > 0 {
            queue.push_back((x - 1, y));
        }
        if x < w - 1 {
            queue.push_back((x + 1, y));
        }
        if y > 0 {
            queue.push_back((x, y - 1));
        }
        if y < h - 1 {
            queue.push_back((x, y + 1));
        }
    }
}

// ─── Sprite sheet splitting ───────────────────────────────────────────────────

fn split_spritesheet(img: &RgbaImage, cols: u32, rows: u32) -> Result<Vec<RgbaImage>> {
    if cols == 0 || rows == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "cols and rows must be > 0".to_string(),
        ));
    }
    let (w, h) = img.dimensions();
    let fw = w / cols;
    let fh = h / rows;
    if fw == 0 || fh == 0 {
        return Err(PixelSnapperError::InvalidInput(format!(
            "Frame size too small: {}×{} — try fewer columns/rows",
            fw, fh
        )));
    }
    let mut frames = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let x0 = col * fw;
            let y0 = row * fh;
            let frame = image::imageops::crop_imm(img, x0, y0, fw, fh).to_image();
            frames.push(frame);
        }
    }
    Ok(frames)
}

// ─── Animated GIF encoding ────────────────────────────────────────────────────

fn frames_to_gif(frames: &[RgbaImage], fps: u32, loop_count: u16) -> Result<Vec<u8>> {
    if frames.is_empty() {
        return Err(PixelSnapperError::InvalidInput("No frames to encode".to_string()));
    }
    let fps = fps.max(1).min(50);
    let delay_cs = (100 / fps) as u16; // GIF delay is in centiseconds

    let (w, h) = frames[0].dimensions();
    let mut buf: Vec<u8> = Vec::new();

    {
        let mut encoder = gif::Encoder::new(&mut buf, w as u16, h as u16, &[])
            .map_err(|e| PixelSnapperError::ProcessingError(e.to_string()))?;

        let repeat = if loop_count == 0 {
            gif::Repeat::Infinite
        } else {
            gif::Repeat::Finite(loop_count)
        };
        encoder
            .set_repeat(repeat)
            .map_err(|e| PixelSnapperError::ProcessingError(e.to_string()))?;

        for rgba_frame in frames {
            // Resize frame to match first frame if needed
            let frame_img = if rgba_frame.dimensions() != (w, h) {
                image::imageops::resize(rgba_frame, w, h, image::imageops::FilterType::Nearest)
            } else {
                rgba_frame.clone()
            };

            let mut frame_pixels: Vec<u8> = frame_img
                .pixels()
                .flat_map(|p| [p[0], p[1], p[2], p[3]])
                .collect();

            // Speed 10: good quality/speed balance (1=best, 30=fastest)
            let mut gif_frame = gif::Frame::from_rgba_speed(
                w as u16,
                h as u16,
                &mut frame_pixels,
                10,
            );
            gif_frame.delay = delay_cs;

            encoder
                .write_frame(&gif_frame)
                .map_err(|e| PixelSnapperError::ProcessingError(e.to_string()))?;
        }
    }

    Ok(buf)
}

// ─── GIF snap ────────────────────────────────────────────────────────────────

/// Decode all frames of an animated GIF. Returns (frame RGBA, delay in centiseconds).
fn read_gif_frames(input_bytes: &[u8]) -> Result<Vec<(RgbaImage, u16)>> {
    use image::AnimationDecoder;
    let cursor = std::io::Cursor::new(input_bytes);
    let decoder = image::codecs::gif::GifDecoder::new(cursor)
        .map_err(|e| PixelSnapperError::ProcessingError(format!("GIF decode: {}", e)))?;
    let mut out = Vec::new();
    for frame in decoder.into_frames() {
        let frame = frame
            .map_err(|e| PixelSnapperError::ProcessingError(format!("GIF frame: {}", e)))?;
        let (num, denom) = frame.delay().numer_denom_ms();
        // delay in centiseconds (GIF unit); avoid divide-by-zero
        let delay_cs = if denom == 0 { 10u16 } else {
            ((num as f64 / denom as f64) / 10.0).round().max(1.0) as u16
        };
        out.push((frame.into_buffer(), delay_cs));
    }
    if out.is_empty() {
        return Err(PixelSnapperError::InvalidInput("GIF has no frames".to_string()));
    }
    Ok(out)
}

/// Encode frames with individual per-frame delays (centiseconds).
fn encode_gif_with_delays(frames: &[RgbaImage], delays: &[u16]) -> Result<Vec<u8>> {
    if frames.is_empty() {
        return Err(PixelSnapperError::InvalidInput("No frames".to_string()));
    }
    let (w, h) = frames[0].dimensions();
    let mut buf = Vec::new();
    {
        let mut enc = gif::Encoder::new(&mut buf, w as u16, h as u16, &[])
            .map_err(|e| PixelSnapperError::ProcessingError(e.to_string()))?;
        enc.set_repeat(gif::Repeat::Infinite)
            .map_err(|e| PixelSnapperError::ProcessingError(e.to_string()))?;
        for (i, rgba_frame) in frames.iter().enumerate() {
            let resized = if rgba_frame.dimensions() != (w, h) {
                image::imageops::resize(rgba_frame, w, h, image::imageops::FilterType::Nearest)
            } else {
                rgba_frame.clone()
            };
            let mut pixels: Vec<u8> = resized.pixels().flat_map(|p| [p[0], p[1], p[2], p[3]]).collect();
            let mut f = gif::Frame::from_rgba_speed(w as u16, h as u16, &mut pixels, 10);
            f.delay = delays.get(i).copied().unwrap_or(10);
            enc.write_frame(&f)
                .map_err(|e| PixelSnapperError::ProcessingError(e.to_string()))?;
        }
    }
    Ok(buf)
}

fn snap_gif_internal(
    input_bytes: &[u8],
    config: &Config,
    remove_bg: bool,
    bg_tolerance: u8,
) -> Result<Vec<u8>> {
    let frames_with_delays = read_gif_frames(input_bytes)?;
    let (fw, fh) = frames_with_delays[0].0.dimensions();
    validate_image_dimensions(fw, fh)?;

    let mut snapped: Vec<RgbaImage> = Vec::with_capacity(frames_with_delays.len());
    let mut delays: Vec<u16> = Vec::with_capacity(frames_with_delays.len());

    for (mut frame, delay) in frames_with_delays {
        if remove_bg {
            if let Some(bg) = detect_background_color(&frame) {
                remove_background_flood(&mut frame, bg, bg_tolerance);
            }
        }
        snapped.push(snap_frame(&frame, config)?);
        delays.push(delay);
    }

    encode_gif_with_delays(&snapped, &delays)
}

// ─── Gradient profiles ───────────────────────────────────────────────────────

/// Computes horizontal and vertical gradient projection profiles over all opaque pixels.
fn compute_profiles(img: &RgbaImage, _config: &Config) -> Result<(Vec<f64>, Vec<f64>)> {
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return Err(PixelSnapperError::InvalidInput(
            "Image too small (minimum 3×3)".to_string(),
        ));
    }

    let mut col_proj = vec![0.0f64; w as usize];
    let mut row_proj = vec![0.0f64; h as usize];

    let lum = |x: u32, y: u32| -> Option<f64> {
        let p = img.get_pixel(x, y);
        if p[3] < 32 { return None; }
        Some(0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64)
    };

    for y in 0..h {
        for x in 1..w - 1 {
            if let (Some(gn), Some(gp)) = (lum(x + 1, y), lum(x - 1, y)) {
                col_proj[x as usize] += (gn - gp).abs();
            }
        }
    }
    for x in 0..w {
        for y in 1..h - 1 {
            if let (Some(gn), Some(gp)) = (lum(x, y + 1), lum(x, y - 1)) {
                row_proj[y as usize] += (gn - gp).abs();
            }
        }
    }

    Ok((col_proj, row_proj))
}

// ─── Step size estimation ────────────────────────────────────────────────────

fn estimate_step_size(profile: &[f64], config: &Config) -> Option<f64> {
    if profile.is_empty() {
        return None;
    }
    let max_val = profile.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val == 0.0 {
        return None;
    }
    let threshold = max_val * config.peak_threshold_multiplier;

    let mut peaks: Vec<usize> = (1..profile.len() - 1)
        .filter(|&i| {
            profile[i] > threshold && profile[i] > profile[i - 1] && profile[i] > profile[i + 1]
        })
        .collect();

    if peaks.len() < 2 {
        return None;
    }

    let mut clean = vec![peaks[0]];
    for &p in peaks.iter().skip(1) {
        if p - clean.last().unwrap() > config.peak_distance_filter - 1 {
            clean.push(p);
        }
    }
    peaks = clean;

    if peaks.len() < 2 {
        return None;
    }

    let mut diffs: Vec<f64> = peaks.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Some(diffs[diffs.len() / 2])
}

fn resolve_step_sizes(
    sx: Option<f64>,
    sy: Option<f64>,
    width: u32,
    height: u32,
    config: &Config,
) -> (f64, f64) {
    match (sx, sy) {
        (Some(x), Some(y)) => {
            let ratio = if x > y { x / y } else { y / x };
            if ratio > config.max_step_ratio {
                let s = x.min(y);
                (s, s)
            } else {
                let avg = (x + y) / 2.0;
                (avg, avg)
            }
        }
        (Some(x), None) => (x, x),
        (None, Some(y)) => (y, y),
        (None, None) => {
            let s = (width.min(height) as f64 / config.fallback_target_segments as f64).max(1.0);
            (s, s)
        }
    }
}

// ─── Grid walker ─────────────────────────────────────────────────────────────

fn walk(profile: &[f64], step_size: f64, limit: usize, config: &Config) -> Result<Vec<usize>> {
    if profile.is_empty() {
        return Err(PixelSnapperError::ProcessingError(
            "Cannot walk empty profile".to_string(),
        ));
    }

    let mut cuts = vec![0usize];
    let mut pos = 0.0f64;
    let window =
        (step_size * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val: f64 = profile.iter().sum::<f64>() / profile.len() as f64;

    while pos < limit as f64 {
        let target = pos + step_size;
        if target >= limit as f64 {
            cuts.push(limit);
            break;
        }
        let start = ((target - window) as usize).max((pos + 1.0) as usize);
        let end = ((target + window) as usize).min(limit);

        if end <= start {
            pos = target;
            continue;
        }

        let (max_idx, max_val) = (start..end)
            .map(|i| (i, profile[i]))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or((start, -1.0));

        if max_val > mean_val * config.walker_strength_threshold {
            cuts.push(max_idx);
            pos = max_idx as f64;
        } else {
            cuts.push(target as usize);
            pos = target;
        }
    }
    Ok(cuts)
}

// ─── Stabilization ───────────────────────────────────────────────────────────

fn stabilize_both_axes(
    px: &[f64],
    py: &[f64],
    raw_cols: Vec<usize>,
    raw_rows: Vec<usize>,
    w: usize,
    h: usize,
    config: &Config,
) -> (Vec<usize>, Vec<usize>) {
    let cols1 = stabilize_cuts(px, raw_cols.clone(), w, &raw_rows, h, config);
    let rows1 = stabilize_cuts(py, raw_rows.clone(), h, &raw_cols, w, config);

    let col_cells = cols1.len().saturating_sub(1).max(1);
    let row_cells = rows1.len().saturating_sub(1).max(1);
    let col_step = w as f64 / col_cells as f64;
    let row_step = h as f64 / row_cells as f64;
    let ratio = if col_step > row_step {
        col_step / row_step
    } else {
        row_step / col_step
    };

    if ratio > config.max_step_ratio {
        let target = col_step.min(row_step);
        let fc = if col_step > target * 1.2 {
            snap_uniform_cuts(px, w, target, config, config.min_cuts_per_axis)
        } else {
            cols1
        };
        let fr = if row_step > target * 1.2 {
            snap_uniform_cuts(py, h, target, config, config.min_cuts_per_axis)
        } else {
            rows1
        };
        (fc, fr)
    } else {
        (cols1, rows1)
    }
}

fn stabilize_cuts(
    profile: &[f64],
    cuts: Vec<usize>,
    limit: usize,
    sibling_cuts: &[usize],
    sibling_limit: usize,
    config: &Config,
) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }
    let cuts = sanitize_cuts(cuts, limit);
    let min_req = config.min_cuts_per_axis.max(2).min(limit.saturating_add(1));
    let axis_cells = cuts.len().saturating_sub(1);
    let sib_cells = sibling_cuts.len().saturating_sub(1);
    let sib_has_grid =
        sibling_limit > 0 && sib_cells >= min_req.saturating_sub(1) && sib_cells > 0;
    let skewed = sib_has_grid && axis_cells > 0 && {
        let ax_step = limit as f64 / axis_cells as f64;
        let sib_step = sibling_limit as f64 / sib_cells as f64;
        let r = ax_step / sib_step;
        r > config.max_step_ratio || r < 1.0 / config.max_step_ratio
    };

    if cuts.len() >= min_req && !skewed {
        return cuts;
    }

    let mut target_step = if sib_has_grid {
        sibling_limit as f64 / sib_cells as f64
    } else if config.fallback_target_segments > 1 {
        limit as f64 / config.fallback_target_segments as f64
    } else if axis_cells > 0 {
        limit as f64 / axis_cells as f64
    } else {
        limit as f64
    };
    if !target_step.is_finite() || target_step <= 0.0 {
        target_step = 1.0;
    }
    snap_uniform_cuts(profile, limit, target_step, config, min_req)
}

fn sanitize_cuts(mut cuts: Vec<usize>, limit: usize) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }
    let mut has_zero = false;
    let mut has_limit = false;
    for v in cuts.iter_mut() {
        if *v == 0 {
            has_zero = true;
        }
        if *v >= limit {
            *v = limit;
        }
        if *v == limit {
            has_limit = true;
        }
    }
    if !has_zero {
        cuts.push(0);
    }
    if !has_limit {
        cuts.push(limit);
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

fn snap_uniform_cuts(
    profile: &[f64],
    limit: usize,
    target_step: f64,
    config: &Config,
    min_required: usize,
) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }
    if limit == 1 {
        return vec![0, 1];
    }

    let mut desired = if target_step.is_finite() && target_step > 0.0 {
        (limit as f64 / target_step).round() as usize
    } else {
        0
    };
    desired = desired
        .max(min_required.saturating_sub(1))
        .max(1)
        .min(limit);

    let cell_w = limit as f64 / desired as f64;
    let window =
        (cell_w * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val = if profile.is_empty() {
        0.0
    } else {
        profile.iter().sum::<f64>() / profile.len() as f64
    };

    let mut cuts = vec![0usize];
    for idx in 1..desired {
        let target = cell_w * idx as f64;
        let prev = *cuts.last().unwrap();
        if prev + 1 >= limit {
            break;
        }
        let start = ((target - window).floor() as isize)
            .max(prev as isize + 1)
            .max(0) as usize;
        let end = ((target + window).ceil() as isize).min(limit as isize - 1) as usize;
        let (mut best_idx, mut best_val) = (start.min(profile.len().saturating_sub(1)), -1.0f64);
        for i in start..=end.min(profile.len().saturating_sub(1)) {
            let v = profile.get(i).copied().unwrap_or(0.0);
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        if best_val < mean_val * config.walker_strength_threshold {
            let fi = (target.round() as isize)
                .max(prev as isize + 1)
                .min(limit as isize - 1) as usize;
            best_idx = fi;
        }
        cuts.push(best_idx);
    }
    if *cuts.last().unwrap() != limit {
        cuts.push(limit);
    }
    sanitize_cuts(cuts, limit)
}

// ─── Resample ────────────────────────────────────────────────────────────────

fn resample(img: &RgbaImage, cols: &[usize], rows: &[usize]) -> Result<RgbaImage> {
    if cols.len() < 2 || rows.len() < 2 {
        return Err(PixelSnapperError::ProcessingError(
            "Insufficient grid cuts for resampling".to_string(),
        ));
    }

    let out_w = (cols.len() - 1) as u32;
    let out_h = (rows.len() - 1) as u32;
    let mut out: RgbaImage = ImageBuffer::new(out_w, out_h);

    for (y_i, wy) in rows.windows(2).enumerate() {
        for (x_i, wx) in cols.windows(2).enumerate() {
            let (xs, xe, ys, ye) = (wx[0], wx[1], wy[0], wy[1]);
            if xe <= xs || ye <= ys {
                continue;
            }

            let mut counts: HashMap<[u8; 4], usize> = HashMap::new();
            for y in ys..ye {
                for x in xs..xe {
                    if x < img.width() as usize && y < img.height() as usize {
                        let p = img.get_pixel(x as u32, y as u32).0;
                        if p[3] >= 32 {
                            *counts.entry(p).or_insert(0) += 1;
                        }
                    }
                }
            }

            let best = counts
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
                .map(|(c, _)| c)
                .unwrap_or([0, 0, 0, 0]);

            out.put_pixel(x_i as u32, y_i as u32, Rgba(best));
        }
    }
    Ok(out)
}

