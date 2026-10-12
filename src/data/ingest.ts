/**
 * data/ingest.ts — turn uploaded files into working-resolution buffers.
 *
 * Real photos and their depth labels arrive as arbitrary image files. We decode
 * them, convert to luminance, and resample to the dataset's fixed working size
 * so every stored sample has identical, cheap-to-train dimensions.
 */

import { fileToImageData, imageDataToGray, resizeGray, GrayImage } from '../cv/image';
import { STORAGE_SIZE } from './realDataset';

/** Decode any image file to a working-resolution luminance buffer. */
export async function ingestImageFile(file: File): Promise<GrayImage> {
  const imgData = await fileToImageData(file, 1024);
  return resizeGray(imageDataToGray(imgData), STORAGE_SIZE);
}

/**
 * Decode a depth-map image to a [0,1] buffer where 1 = near.
 *
 * Convention: a normalised depth map is white = near. Many datasets store the
 * opposite (white = far), so `invert` flips the interpretation.
 */
export async function ingestDepthFile(file: File, invert: boolean): Promise<Float32Array> {
  const imgData = await fileToImageData(file, 1024);
  const gray = resizeGray(imageDataToGray(imgData), STORAGE_SIZE);
  const out = new Float32Array(gray.data.length);
  for (let i = 0; i < out.length; i++) {
    const v = gray.data[i];
    out[i] = invert ? 1 - v : v;
  }
  return out;
}

/** A dense supervision mask (every pixel labelled). */
export function denseMask(size = STORAGE_SIZE): Float32Array {
  return new Float32Array(size * size).fill(1);
}
