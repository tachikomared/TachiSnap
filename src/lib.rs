use image::{GenericImageView, ImageBuffer, Rgba, RgbaImage};
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, WeightedIndex};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[cfg(not(target_arch = "wasm32"))]
use std::env;

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
    /// NEW: if >0 forces this pixel size instead of auto-detecting
    pub forced_pixel_size: usize,
    /// NEW: upscale output by this factor (1 = no upscale)
    pub upscale_factor: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            k_colors: 16,
            k_seed: 42,
            input_path: String::new(),
            output_path: String::new(),
            max_kmeans_iterations: 15,
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

/// Returns the detected pixel size without processing the image.
/// Useful for showing the user what grid was found.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn detect_pixel_size(input_bytes: &[u8]) -> std::result::Result<u32, wasm_bindgen::JsValue> {
    let config = Config::default();
    let img = image::load_from_memory(input_bytes).map_err(PixelSnapperError::from)?;
    let (width, height) = img.dimensions();
    validate_image_dimensions(width, height)?;
    let rgba_img = img.to_rgba8();
    let quantized = quantize_image(&rgba_img, &config)?;
    let (px, py) = compute_profiles(&quantized)?;
    let sx = estimate_step_size(&px, &config);
    let sy = estimate_step_size(&py, &config);
    let (step_x, step_y) = resolve_step_sizes(sx, sy, width, height, &config);
    let detected = ((step_x + step_y) / 2.0).round() as u32;
    Ok(detected.max(1))
}

/// Process image with full options.
/// k_colors: palette size (default 16)
/// forced_pixel_size: 0 = auto detect, >0 = use this exact pixel size
/// upscale_factor: 1 = keep output small, 2/4/8 = upscale result
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn process_image(
    input_bytes: &[u8],
    k_colors: Option<u32>,
    forced_pixel_size: Option<u32>,
    upscale_factor: Option<u32>,
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

    process_image_bytes_common(input_bytes, Some(config))
        .map_err(|e| wasm_bindgen::JsValue::from(e))
}

/// Returns the library version string.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn version() -> String {
    "0.2.0".to_string()
}

// ─── CLI ────────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let config = parse_args().unwrap_or_default();
    process_image_cli(&config)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_args() -> Option<Config> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: pixel-snapper <input> <output> [k_colors] [pixel_size] [upscale]");
        return None;
    }
    let mut config = Config {
        input_path: args[1].clone(),
        output_path: args[2].clone(),
        ..Default::default()
    };
    if let Some(k) = args.get(3).and_then(|s| s.parse::<usize>().ok()) {
        if k > 0 { config.k_colors = k; }
    }
    if let Some(ps) = args.get(4).and_then(|s| s.parse::<usize>().ok()) {
        config.forced_pixel_size = ps;
    }
    if let Some(uf) = args.get(5).and_then(|s| s.parse::<usize>().ok()) {
        config.upscale_factor = uf.max(1).min(16);
    }
    Some(config)
}

#[cfg(not(target_arch = "wasm32"))]
fn process_image_cli(config: &Config) -> Result<()> {
    println!("Processing: {}", config.input_path);
    let img_bytes = std::fs::read(&config.input_path).map_err(|e| {
        PixelSnapperError::ProcessingError(format!("Failed to read input: {}", e))
    })?;
    let output_bytes = process_image_bytes_common(&img_bytes, Some(config.clone()))?;
    std::fs::write(&config.output_path, output_bytes).map_err(|e| {
        PixelSnapperError::ProcessingError(format!("Failed to write output: {}", e))
    })?;
    println!("Saved to: {}", config.output_path);
    Ok(())
}

// ─── Core pipeline ──────────────────────────────────────────────────────────

fn process_image_bytes_common(input_bytes: &[u8], config: Option<Config>) -> Result<Vec<u8>> {
    let config = config.unwrap_or_default();

    let img = image::load_from_memory(input_bytes)?;
    let (width, height) = img.dimensions();
    validate_image_dimensions(width, height)?;

    let rgba_img = img.to_rgba8();
    let quantized_img = quantize_image(&rgba_img, &config)?;
    let (profile_x, profile_y) = compute_profiles(&quantized_img)?;

    let (step_x, step_y) = if config.forced_pixel_size > 0 {
        let s = config.forced_pixel_size as f64;
        (s, s)
    } else {
        let sx = estimate_step_size(&profile_x, &config);
        let sy = estimate_step_size(&profile_y, &config);
        resolve_step_sizes(sx, sy, width, height, &config)
    };

    let raw_col_cuts = walk(&profile_x, step_x, width as usize, &config)?;
    let raw_row_cuts = walk(&profile_y, step_y, height as usize, &config)?;

    let (col_cuts, row_cuts) = stabilize_both_axes(
        &profile_x,
        &profile_y,
        raw_col_cuts,
        raw_row_cuts,
        width as usize,
        height as usize,
        &config,
    );

    let output_img = resample(&quantized_img, &col_cuts, &row_cuts)?;

    // Optional upscale
    let final_img = if config.upscale_factor > 1 {
        upscale_nearest(&output_img, config.upscale_factor)
    } else {
        output_img
    };

    let mut output_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut output_bytes);
    final_img
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(PixelSnapperError::ImageError)?;

    Ok(output_bytes)
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

// ─── Color quantization (k-means++) ─────────────────────────────────────────

fn quantize_image(img: &RgbaImage, config: &Config) -> Result<RgbaImage> {
    if config.k_colors == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "k_colors must be > 0".to_string(),
        ));
    }

    let opaque_pixels: Vec<[f32; 3]> = img
        .pixels()
        .filter_map(|p| {
            if p[3] == 0 {
                None
            } else {
                Some([p[0] as f32, p[1] as f32, p[2] as f32])
            }
        })
        .collect();

    let n_pixels = opaque_pixels.len();
    if n_pixels == 0 {
        return Ok(img.clone());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(config.k_seed);
    let k = config.k_colors.min(n_pixels);

    fn dist_sq(p: &[f32; 3], c: &[f32; 3]) -> f32 {
        let dr = p[0] - c[0];
        let dg = p[1] - c[1];
        let db = p[2] - c[2];
        dr * dr + dg * dg + db * db
    }

    // k-means++ init
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    let first_idx = rng.gen_range(0..n_pixels);
    centroids.push(opaque_pixels[first_idx]);
    let mut distances = vec![f32::MAX; n_pixels];

    for _ in 1..k {
        let last_c = centroids.last().unwrap();
        let mut sum_sq = 0.0;
        for (i, p) in opaque_pixels.iter().enumerate() {
            let d = dist_sq(p, last_c);
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
                    dist_sq(p, a)
                        .partial_cmp(&dist_sq(p, b))
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
                .map(|(n, o)| dist_sq(n, o))
                .fold(0.0f32, f32::max);
            if max_move < 0.01 {
                break;
            }
        }

        prev_centroids.copy_from_slice(&centroids);
    }

    // Map every pixel to nearest centroid
    let mut new_img = RgbaImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel[3] == 0 {
            new_img.put_pixel(x, y, *pixel);
            continue;
        }
        let p = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
        let best = centroids
            .iter()
            .min_by(|a, b| dist_sq(&p, a).partial_cmp(&dist_sq(&p, b)).unwrap_or(Ordering::Equal))
            .unwrap();
        new_img.put_pixel(
            x, y,
            Rgba([
                best[0].round() as u8,
                best[1].round() as u8,
                best[2].round() as u8,
                pixel[3],
            ]),
        );
    }
    Ok(new_img)
}

// ─── Gradient profiles ───────────────────────────────────────────────────────

fn compute_profiles(img: &RgbaImage) -> Result<(Vec<f64>, Vec<f64>)> {
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return Err(PixelSnapperError::InvalidInput(
            "Image too small (minimum 3×3)".to_string(),
        ));
    }

    let mut col_proj = vec![0.0; w as usize];
    let mut row_proj = vec![0.0; h as usize];

    let gray = |x: u32, y: u32| {
        let p = img.get_pixel(x, y);
        if p[3] == 0 {
            0.0
        } else {
            0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64
        }
    };

    for y in 0..h {
        for x in 1..w - 1 {
            col_proj[x as usize] += (gray(x + 1, y) - gray(x - 1, y)).abs();
        }
    }
    for x in 0..w {
        for y in 1..h - 1 {
            row_proj[y as usize] += (gray(x, y + 1) - gray(x, y - 1)).abs();
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

    // Remove peaks that are too close together
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
    let window = (step_size * config.walker_search_window_ratio).max(config.walker_min_search_window);
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
    let ratio = if col_step > row_step { col_step / row_step } else { row_step / col_step };

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
        if *v == 0 { has_zero = true; }
        if *v >= limit { *v = limit; }
        if *v == limit { has_limit = true; }
    }
    if !has_zero { cuts.push(0); }
    if !has_limit { cuts.push(limit); }
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
    if limit == 0 { return vec![0]; }
    if limit == 1 { return vec![0, 1]; }

    let mut desired = if target_step.is_finite() && target_step > 0.0 {
        (limit as f64 / target_step).round() as usize
    } else { 0 };
    desired = desired.max(min_required.saturating_sub(1)).max(1).min(limit);

    let cell_w = limit as f64 / desired as f64;
    let window = (cell_w * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val = if profile.is_empty() {
        0.0
    } else {
        profile.iter().sum::<f64>() / profile.len() as f64
    };

    let mut cuts = vec![0usize];
    for idx in 1..desired {
        let target = cell_w * idx as f64;
        let prev = *cuts.last().unwrap();
        if prev + 1 >= limit { break; }
        let start = ((target - window).floor() as isize).max(prev as isize + 1).max(0) as usize;
        let end = ((target + window).ceil() as isize).min(limit as isize - 1) as usize;
        let (mut best_idx, mut best_val) = (start.min(profile.len().saturating_sub(1)), -1.0f64);
        for i in start..=end.min(profile.len().saturating_sub(1)) {
            let v = profile.get(i).copied().unwrap_or(0.0);
            if v > best_val { best_val = v; best_idx = i; }
        }
        if best_val < mean_val * config.walker_strength_threshold {
            let fi = (target.round() as isize)
                .max(prev as isize + 1)
                .min(limit as isize - 1) as usize;
            best_idx = fi;
        }
        cuts.push(best_idx);
    }
    if *cuts.last().unwrap() != limit { cuts.push(limit); }
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
            if xe <= xs || ye <= ys { continue; }

            // Majority vote per cell
            let mut counts: HashMap<[u8; 4], usize> = HashMap::new();
            for y in ys..ye {
                for x in xs..xe {
                    if x < img.width() as usize && y < img.height() as usize {
                        let p = img.get_pixel(x as u32, y as u32).0;
                        *counts.entry(p).or_insert(0) += 1;
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
