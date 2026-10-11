/**
 * data/augment.ts — data augmentation for image → depth learning.
 *
 * A small real dataset overfits fast. Augmentation stretches it by applying
 * transformations that preserve the *image ↔ depth* correspondence:
 *
 *   - horizontal flip      (mirror image, depth and mask together)
 *   - affine jitter        (small translate / zoom / rotate, warped together)
 *   - photometric jitter   (brightness / contrast / gamma — image only)
 *   - additive noise       (image only)
 *   - cutout               (occlude a patch; the hidden pixels lose supervision)
 *
 * Everything is driven by a seeded PRNG so a run is reproducible.
 */

import { Sample } from './dataset';

export interface AugmentOptions {
  /** probability of a horizontal mirror */
  hflip: number;
  /** max brightness offset, in [0,1] units */
  brightness: number;
  /** max contrast deviation (0.2 => scale in [0.8, 1.2]) */
  contrast: number;
  /** max gamma deviation */
  gamma: number;
  /** std of additive Gaussian noise, in [0,1] units */
  noise: number;
  /** max translation as a fraction of the image size */
  translate: number;
  /** max zoom deviation (0.1 => scale in [0.9, 1.1]) */
  zoom: number;
  /** max rotation in radians */
  rotate: number;
  /** probability of a cutout patch */
  cutout: number;
}

export const DEFAULT_AUGMENT: AugmentOptions = {
  hflip: 0.5,
  brightness: 0.12,
  contrast: 0.15,
  gamma: 0.15,
  noise: 0.02,
  translate: 0.04,
  zoom: 0.08,
  rotate: 0.04,
  cutout: 0.15,
};

export const NO_AUGMENT: AugmentOptions = {
  hflip: 0,
  brightness: 0,
  contrast: 0,
  gamma: 0,
  noise: 0,
  translate: 0,
  zoom: 0,
  rotate: 0,
  cutout: 0,
};

/** Small deterministic PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Rng = () => number;

const rand = (rng: Rng, lo: number, hi: number) => lo + rng() * (hi - lo);

/* ------------------------------------------------------------------ */
/* Geometric warp                                                      */
/* ------------------------------------------------------------------ */

interface Affine {
  flip: boolean;
  rot: number;
  zoom: number;
  tx: number;
  ty: number;
}

function sampleAt(
  src: Float32Array, size: number, u: number, v: number, nearest: boolean,
): number {
  if (nearest) {
    const xi = Math.round(u);
    const yi = Math.round(v);
    if (xi < 0 || yi < 0 || xi >= size || yi >= size) return 0; // unlabelled outside
    return src[yi * size + xi];
  }
  const clamp = (i: number) => (i < 0 ? 0 : i >= size ? size - 1 : i);
  const x0 = Math.floor(u);
  const y0 = Math.floor(v);
  const fx = u - x0;
  const fy = v - y0;
  const a = src[clamp(y0) * size + clamp(x0)];
  const b = src[clamp(y0) * size + clamp(x0 + 1)];
  const c = src[clamp(y0 + 1) * size + clamp(x0)];
  const d = src[clamp(y0 + 1) * size + clamp(x0 + 1)];
  return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
}

function warp(src: Float32Array, size: number, t: Affine, nearest: boolean): Float32Array {
  const out = new Float32Array(size * size);
  const c = (size - 1) / 2;
  const cos = Math.cos(t.rot);
  const sin = Math.sin(t.rot);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let dx = (x - c) / t.zoom;
      let dy = (y - c) / t.zoom;
      // inverse rotation
      let u = dx * cos + dy * sin;
      let v = -dx * sin + dy * cos;
      u += c + t.tx;
      v += c + t.ty;
      if (t.flip) u = size - 1 - u;
      out[y * size + x] = sampleAt(src, size, u, v, nearest);
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Photometric helpers                                                 */
/* ------------------------------------------------------------------ */

function gaussian(rng: Rng): number {
  const u1 = rng() || 1e-12;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/* ------------------------------------------------------------------ */
/* The augmentation itself                                             */
/* ------------------------------------------------------------------ */

/**
 * Apply one random augmentation to a sample. Image, depth and mask are warped
 * by the *same* affine transform so their pixel correspondence is preserved;
 * photometric changes touch the image only.
 */
export function augmentSample(sample: Sample, opts: AugmentOptions, rng: Rng): Sample {
  const size = sample.size;
  let image: Float32Array = sample.image.slice();
  let depth: Float32Array = sample.depth.slice();
  let mask: Float32Array | null = sample.mask ? sample.mask.slice() : null;

  // --- geometric ---
  const affine: Affine = {
    flip: rng() < opts.hflip,
    rot: opts.rotate ? rand(rng, -opts.rotate, opts.rotate) : 0,
    zoom: opts.zoom ? 1 + rand(rng, -opts.zoom, opts.zoom) : 1,
    tx: opts.translate ? rand(rng, -opts.translate, opts.translate) * size : 0,
    ty: opts.translate ? rand(rng, -opts.translate, opts.translate) * size : 0,
  };
  if (affine.flip || affine.rot || affine.zoom !== 1 || affine.tx || affine.ty) {
    image = warp(image, size, affine, false);
    depth = warp(depth, size, affine, false);
    if (mask) mask = warp(mask, size, affine, true);
  }

  // --- photometric (image only) ---
  if (opts.brightness) {
    const b = rand(rng, -opts.brightness, opts.brightness);
    for (let i = 0; i < image.length; i++) image[i] += b;
  }
  if (opts.contrast) {
    const k = 1 + rand(rng, -opts.contrast, opts.contrast);
    for (let i = 0; i < image.length; i++) image[i] = (image[i] - 0.5) * k + 0.5;
  }
  if (opts.gamma) {
    const g = 1 + rand(rng, -opts.gamma, opts.gamma);
    for (let i = 0; i < image.length; i++) image[i] = Math.pow(Math.max(0, image[i]), g);
  }
  if (opts.noise) {
    for (let i = 0; i < image.length; i++) image[i] += gaussian(rng) * opts.noise;
  }

  // --- cutout (occlude a patch) ---
  if (rng() < opts.cutout) {
    const w = Math.max(2, Math.floor(size * rand(rng, 0.12, 0.3)));
    const h = Math.max(2, Math.floor(size * rand(rng, 0.12, 0.3)));
    const x0 = Math.floor(rng() * (size - w));
    const y0 = Math.floor(rng() * (size - h));
    if (!mask) mask = new Float32Array(size * size).fill(1);
    for (let y = y0; y < y0 + h; y++) {
      for (let x = x0; x < x0 + w; x++) {
        const i = y * size + x;
        image[i] = 0;
        mask[i] = 0; // hidden pixels are no longer supervised
      }
    }
  }

  // keep the image in [0,1]
  for (let i = 0; i < image.length; i++) {
    image[i] = image[i] < 0 ? 0 : image[i] > 1 ? 1 : image[i];
  }
  for (let i = 0; i < depth.length; i++) {
    depth[i] = depth[i] < 0 ? 0 : depth[i] > 1 ? 1 : depth[i];
  }

  return { image, depth, size, mask };
}

/**
 * Expand a dataset by `factor` (including the originals). Useful when you want
 * a fixed, larger dataset up front rather than on-the-fly augmentation.
 */
export function augmentDataset(
  samples: Sample[], opts: AugmentOptions, factor: number, seed = 1,
): Sample[] {
  const rng = mulberry32(seed);
  const out: Sample[] = [];
  for (const s of samples) {
    out.push(s);
    for (let i = 1; i < factor; i++) out.push(augmentSample(s, opts, rng));
  }
  return out;
}

