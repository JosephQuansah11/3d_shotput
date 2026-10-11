/**
 * data/resize.ts — resampling for square float buffers.
 *
 * Images and depth maps are resized with bilinear filtering; supervision masks
 * use nearest-neighbour so they stay binary (a pixel is either labelled or not).
 */

export type ResizeMode = 'bilinear' | 'nearest';

export function resizeBuffer(
  data: Float32Array,
  from: number,
  to: number,
  mode: ResizeMode = 'bilinear',
): Float32Array {
  if (from === to) return data.slice();
  const out = new Float32Array(to * to);
  const scale = from / to;

  if (mode === 'nearest') {
    for (let y = 0; y < to; y++) {
      const sy = Math.min(from - 1, Math.floor(y * scale));
      for (let x = 0; x < to; x++) {
        const sx = Math.min(from - 1, Math.floor(x * scale));
        out[y * to + x] = data[sy * from + sx];
      }
    }
    return out;
  }

  for (let y = 0; y < to; y++) {
    const fy = (y + 0.5) * scale - 0.5;
    const y0 = Math.max(0, Math.floor(fy));
    const y1 = Math.min(from - 1, y0 + 1);
    const wy = Math.min(1, Math.max(0, fy - y0));
    for (let x = 0; x < to; x++) {
      const fx = (x + 0.5) * scale - 0.5;
      const x0 = Math.max(0, Math.floor(fx));
      const x1 = Math.min(from - 1, x0 + 1);
      const wx = Math.min(1, Math.max(0, fx - x0));

      const a = data[y0 * from + x0];
      const b = data[y0 * from + x1];
      const c = data[y1 * from + x0];
      const d = data[y1 * from + x1];
      out[y * to + x] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy;
    }
  }
  return out;
}

/** Resize an RGBA ImageData to a square RGBA ImageData (nearest neighbour). */
export function resizeImageData(img: ImageData, to: number): ImageData {
  const out = new ImageData(to, to);
  for (let y = 0; y < to; y++) {
    const sy = Math.min(img.height - 1, Math.floor((y * img.height) / to));
    for (let x = 0; x < to; x++) {
      const sx = Math.min(img.width - 1, Math.floor((x * img.width) / to));
      const sp = (sy * img.width + sx) * 4;
      const dp = (y * to + x) * 4;
      out.data[dp] = img.data[sp];
      out.data[dp + 1] = img.data[sp + 1];
      out.data[dp + 2] = img.data[sp + 2];
      out.data[dp + 3] = img.data[sp + 3];
    }
  }
  return out;
}
