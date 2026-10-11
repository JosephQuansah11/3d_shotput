/**
 * cv/image.ts — image ingestion and tensor conversion.
 *
 * Turns an arbitrary image (uploaded or drawn to a canvas) into the [1,H,W]
 * float tensor our network expects, with bilinear resampling done by hand.
 */

import { Tensor } from '../ai/tensor';

export interface GrayImage {
  data: Float32Array; // row-major, values in [0,1]
  width: number;
  height: number;
}

/** RGBA ImageData -> single-channel luminance in [0,1]. */
export function imageDataToGray(img: ImageData): GrayImage {
  const { width, height, data } = img;
  const out = new Float32Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    // Rec. 709 luma
    out[i] = (0.2126 * data[p] + 0.7152 * data[p + 1] + 0.0722 * data[p + 2]) / 255;
  }
  return { data: out, width, height };
}

/** Sample a grayscale image at normalised coordinates with bilinear filtering. */
export function sampleBilinear(img: GrayImage, u: number, v: number): number {
  const x = u * (img.width - 1);
  const y = v * (img.height - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, img.width - 1);
  const y1 = Math.min(y0 + 1, img.height - 1);
  const fx = x - x0;
  const fy = y - y0;
  const a = img.data[y0 * img.width + x0];
  const b = img.data[y0 * img.width + x1];
  const c = img.data[y1 * img.width + x0];
  const d = img.data[y1 * img.width + x1];
  return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
}

/** Resample to a fixed square size, preserving aspect ratio (letterbox). */
export function resizeGray(img: GrayImage, size: number): GrayImage {
  const out = new Float32Array(size * size);
  const scale = Math.min(size / img.width, size / img.height);
  const drawW = img.width * scale;
  const drawH = img.height * scale;
  const offX = (size - drawW) / 2;
  const offY = (size - drawH) / 2;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const sx = x - offX;
      const sy = y - offY;
      let val = 0;
      if (sx >= 0 && sx < drawW && sy >= 0 && sy < drawH) {
        val = sampleBilinear(img, sx / drawW, sy / drawH);
      }
      out[y * size + x] = val;
    }
  }
  return { data: out, width: size, height: size };
}

/** Grayscale image -> [1, H, W] tensor. */
export function grayToTensor(img: GrayImage): Tensor {
  return new Tensor(Float64Array.from(img.data), [1, img.height, img.width]);
}

/** [1, H, W] tensor -> grayscale image (clamped to [0,1]). */
export function tensorToGray(t: Tensor): GrayImage {
  const [, h, w] = t.shape;
  const out = new Float32Array(h * w);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.max(0, Math.min(1, t.data[i]));
  }
  return { data: out, width: w, height: h };
}

/** Load a File/Blob into an ImageData via a throwaway canvas. */
export async function fileToImageData(file: File, maxSize = 512): Promise<ImageData> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  return ctx.getImageData(0, 0, w, h);
}

/** Build a URL for an uploaded file so it can be shown in an <img>. */
export const objectUrl = (file: File): string => URL.createObjectURL(file);
