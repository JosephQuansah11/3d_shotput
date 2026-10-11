/**
 * data/realDataset.ts — real-photo samples, persisted in IndexedDB.
 *
 * A RealSample is a photo the user brought plus its depth supervision:
 *  - `depth` values in [0,1] (1 = near), and
 *  - `mask` marking which pixels are actually labelled. Dense depth maps set
 *    mask = all ones; brush-labelled photos set mask = 1 only under the strokes.
 *
 * We store everything at a fixed working resolution and resample to the model
 * resolution at training time, so changing the model size never invalidates
 * the dataset.
 */

import { Sample } from './dataset';
import { resizeBuffer } from './resize';

export const STORAGE_SIZE = 96;

export type RealSampleSource = 'depth-map' | 'brush';

export interface RealSample {
  id: string;
  name: string;
  createdAt: number;
  size: number; // working resolution of the buffers
  image: Float32Array; // size*size luminance in [0,1]
  depth: Float32Array; // size*size, 1 = near
  mask: Float32Array | null; // size*size, 1 = labelled
  thumbnail: string; // small data URL for the UI
  source: RealSampleSource;
}

export const newId = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/** Build a thumbnail data URL from a luminance buffer. */
export function makeThumbnail(image: Float32Array, size: number, thumb = 64): string {
  const canvas = document.createElement('canvas');
  canvas.width = thumb;
  canvas.height = thumb;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const img = ctx.createImageData(thumb, thumb);
  for (let y = 0; y < thumb; y++) {
    const sy = Math.min(size - 1, Math.floor((y * size) / thumb));
    for (let x = 0; x < thumb; x++) {
      const sx = Math.min(size - 1, Math.floor((x * size) / thumb));
      const v = Math.round(255 * Math.max(0, Math.min(1, image[sy * size + sx])));
      const p = (y * thumb + x) * 4;
      img.data[p] = v;
      img.data[p + 1] = v;
      img.data[p + 2] = v;
      img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL('image/png');
}

/** Turn a stored RealSample into a training Sample at the model resolution. */
export function toTrainingSample(sample: RealSample, modelSize: number): Sample {
  const image = resizeBuffer(sample.image, sample.size, modelSize, 'bilinear');
  const depth = resizeBuffer(sample.depth, sample.size, modelSize, 'bilinear');
  const mask = sample.mask
    ? resizeBuffer(sample.mask, sample.size, modelSize, 'nearest')
    : null;
  return { image, depth, size: modelSize, mask };
}

/** Average fraction of pixels that are labelled (1 = dense, ~0 = very sparse). */
export function labelledFraction(sample: RealSample): number {
  if (!sample.mask) return 1;
  let n = 0;
  for (let i = 0; i < sample.mask.length; i++) if (sample.mask[i] > 0) n++;
  return n / sample.mask.length;
}

/* ------------------------------------------------------------------ */
/* IndexedDB persistence                                               */
/* ------------------------------------------------------------------ */

const DB_NAME = 'shotput';
const STORE = 'real-samples';
const DB_VERSION = 1;

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDB().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const transaction = db.transaction(STORE, mode);
        const req = run(transaction.objectStore(STORE));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
        transaction.oncomplete = () => db.close();
      }),
  );
}

export async function putRealSample(sample: RealSample): Promise<void> {
  await tx('readwrite', (store) => store.put(sample) as IDBRequest<IDBValidKey>).then(() => undefined);
}

export function getAllRealSamples(): Promise<RealSample[]> {
  return tx<RealSample[]>('readonly', (store) => store.getAll() as IDBRequest<RealSample[]>);
}

export async function deleteRealSample(id: string): Promise<void> {
  await tx('readwrite', (store) => store.delete(id) as IDBRequest<undefined>).then(() => undefined);
}

export async function clearRealSamples(): Promise<void> {
  await tx('readwrite', (store) => store.clear() as IDBRequest<undefined>).then(() => undefined);
}
