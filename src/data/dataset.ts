/**
 * data/dataset.ts — render scenes into supervised (image, depth) pairs.
 *
 * Every sample is a small square image plus the ground-truth depth map the
 * network is trained to reproduce. Because both come from the same ray tracer
 * they are perfectly aligned.
 */

import { v3, normalize3 } from '../math/vec';
import { Scene, randomScene, trace, shade } from './scene';

export interface Sample {
  image: Float32Array; // [size*size], luminance in [0,1]
  depth: Float32Array; // [size*size], 1 = near, 0 = far
  size: number;
}

export interface RenderOptions {
  size: number;
  fovY: number; // radians
  near: number;
  far: number;
}

export const DEFAULT_RENDER: RenderOptions = {
  size: 32,
  fovY: (50 * Math.PI) / 180,
  near: 0.2,
  far: 4.0,
};

/** Render one scene to an image + depth pair. */
export function renderScene(scene: Scene, opts: RenderOptions): Sample {
  const { size, fovY, near, far } = opts;
  const image = new Float32Array(size * size);
  const depth = new Float32Array(size * size);

  const focal = 1 / Math.tan(fovY / 2);
  const origin = v3(0, 0, 0);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      // map pixel to normalised device coords in [-1,1]
      const ndcX = ((x + 0.5) / size) * 2 - 1;
      const ndcY = 1 - ((y + 0.5) / size) * 2;
      const dir = normalize3(v3(ndcX / focal, ndcY / focal, -1));

      const hit = trace(scene, origin, dir);
      const idx = y * size + x;
      if (hit) {
        image[idx] = shade(scene, hit);
        const d = (far - hit.t) / (far - near);
        depth[idx] = Math.max(0, Math.min(1, d));
      } else {
        image[idx] = scene.background;
        depth[idx] = 0;
      }
    }
  }

  return { image, depth, size };
}

/** A brand new random (image, depth) sample. */
export function randomSample(opts: RenderOptions = DEFAULT_RENDER): Sample {
  return renderScene(randomScene(), opts);
}

/** Deterministic dataset from a fixed seed list of scenes. */
export function generateDataset(count: number, opts: RenderOptions = DEFAULT_RENDER): Sample[] {
  const out: Sample[] = [];
  for (let i = 0; i < count; i++) out.push(randomSample(opts));
  return out;
}

/* ------------------------------------------------------------------ */
/* Persistence                                                         */
/* ------------------------------------------------------------------ */

export interface SerializedSample {
  image: number[];
  depth: number[];
  size: number;
}

export const serializeSample = (s: Sample): SerializedSample => ({
  image: Array.from(s.image),
  depth: Array.from(s.depth),
  size: s.size,
});

export const deserializeSample = (s: SerializedSample): Sample => ({
  image: Float32Array.from(s.image),
  depth: Float32Array.from(s.depth),
  size: s.size,
});

/**
 * Render a depth buffer into an RGB ImageData for display, using a simple
 * "turbo"-like colour ramp so near objects are warm and far objects cool.
 */
export function depthToImageData(depth: Float32Array, size: number): ImageData {
  const img = new ImageData(size, size);
  for (let i = 0; i < depth.length; i++) {
    const d = Math.max(0, Math.min(1, depth[i]));
    // ramp: 0 -> blue, 0.5 -> green, 1 -> red
    const r = Math.round(255 * Math.min(1, Math.max(0, 2 * d - 0.5)) * 2 * 0.5 + 255 * Math.max(0, 2 * d - 1));
    const g = Math.round(255 * (1 - Math.abs(2 * d - 1)));
    const b = Math.round(255 * Math.max(0, 1 - 2 * d));
    const p = i * 4;
    img.data[p] = Math.min(255, r);
    img.data[p + 1] = Math.min(255, g);
    img.data[p + 2] = Math.min(255, b);
    img.data[p + 3] = 255;
  }
  return img;
}

/** Grayscale ImageData for an image buffer. */
export function grayToImageData(buf: Float32Array, size: number): ImageData {
  const img = new ImageData(size, size);
  for (let i = 0; i < buf.length; i++) {
    const v = Math.round(255 * Math.max(0, Math.min(1, buf[i])));
    const p = i * 4;
    img.data[p] = v;
    img.data[p + 1] = v;
    img.data[p + 2] = v;
    img.data[p + 3] = 255;
  }
  return img;
}
