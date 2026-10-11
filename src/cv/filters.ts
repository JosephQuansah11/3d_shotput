/**
 * cv/filters.ts — classical computer-vision operators, hand-written.
 *
 * These give the pipeline a non-learned "structural" view of the image that
 * runs alongside the neural network: edges, gradients, blur and a normalised
 * depth-style map.
 */

import { GrayImage } from './image';

/** 2D 'same' convolution of a single-channel image with a square kernel. */
export function convolve2D(
  img: GrayImage,
  kernel: Float64Array,
  k: number,
): Float32Array {
  const { data, width, height } = img;
  const out = new Float32Array(width * height);
  const half = Math.floor(k / 2);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const iy = Math.min(height - 1, Math.max(0, y + ky - half));
          const ix = Math.min(width - 1, Math.max(0, x + kx - half));
          sum += data[iy * width + ix] * kernel[ky * k + kx];
        }
      }
      out[y * width + x] = sum;
    }
  }
  return out;
}

/** Normalised Gaussian kernel of the given odd size and sigma. */
export function gaussianKernel(size: number, sigma: number): Float64Array {
  const k = new Float64Array(size * size);
  const half = Math.floor(size / 2);
  let sum = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - half;
      const dy = y - half;
      const v = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      k[y * size + x] = v;
      sum += v;
    }
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}

export function gaussianBlur(img: GrayImage, size = 5, sigma = 1.2): GrayImage {
  const out = convolve2D(img, gaussianKernel(size, sigma), size);
  return { data: out, width: img.width, height: img.height };
}

export interface Gradients {
  gx: Float32Array;
  gy: Float32Array;
  magnitude: Float32Array;
  width: number;
  height: number;
}

/** Sobel gradient operators. */
export function sobel(img: GrayImage): Gradients {
  const kx = new Float64Array([-1, 0, 1, -2, 0, 2, -1, 0, 1]);
  const ky = new Float64Array([-1, -2, -1, 0, 0, 0, 1, 2, 1]);
  const gx = convolve2D(img, kx, 3);
  const gy = convolve2D(img, ky, 3);
  const mag = new Float32Array(gx.length);
  for (let i = 0; i < gx.length; i++) mag[i] = Math.hypot(gx[i], gy[i]);
  return { gx, gy, magnitude: mag, width: img.width, height: img.height };
}

/** Laplacian (edge/curvature) operator. */
export function laplacian(img: GrayImage): Float32Array {
  const k = new Float64Array([0, 1, 0, 1, -4, 1, 0, 1, 0]);
  return convolve2D(img, k, 3);
}

/** Rescale a float buffer to [0,1] by its observed min/max. */
export function normalizeToUnit(values: Float32Array): Float32Array {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < lo) lo = values[i];
    if (values[i] > hi) hi = values[i];
  }
  const range = hi - lo || 1;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = (values[i] - lo) / range;
  return out;
}

/**
 * A cheap "shape-from-shading" pseudo-depth: darker pixels are treated as
 * farther away. Not a substitute for the learned model — it's a baseline the
 * UI shows for comparison.
 */
export function shadingPseudoDepth(img: GrayImage): Float32Array {
  const out = new Float32Array(img.data.length);
  for (let i = 0; i < out.length; i++) out[i] = 1 - img.data[i];
  return normalizeToUnit(out);
}
