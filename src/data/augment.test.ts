import { describe, it, expect } from 'vitest';
import { Sample } from './dataset';
import {
  augmentSample, augmentDataset, mulberry32, NO_AUGMENT, DEFAULT_AUGMENT,
} from './augment';

const ramp = (size: number, withMask: boolean): Sample => {
  const image = new Float32Array(size * size);
  const depth = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      image[y * size + x] = x / (size - 1);
      depth[y * size + x] = x / (size - 1);
    }
  }
  const mask = withMask ? new Float32Array(size * size).fill(1) : null;
  return { image, depth, size, mask };
};

describe('augmentation', () => {
  it('NO_AUGMENT leaves the sample untouched', () => {
    const s = ramp(8, true);
    const out = augmentSample(s, NO_AUGMENT, mulberry32(1));
    expect(Array.from(out.image)).toEqual(Array.from(s.image));
    expect(Array.from(out.depth)).toEqual(Array.from(s.depth));
    expect(Array.from(out.mask!)).toEqual(Array.from(s.mask!));
  });

  it('horizontal flip mirrors image and depth together', () => {
    const size = 8;
    const s = ramp(size, false);
    const out = augmentSample(s, { ...NO_AUGMENT, hflip: 1 }, mulberry32(7));
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        expect(out.image[y * size + x]).toBeCloseTo((size - 1 - x) / (size - 1), 5);
        expect(out.depth[y * size + x]).toBeCloseTo((size - 1 - x) / (size - 1), 5);
      }
    }
  });

  it('photometric jitter changes the image but not the depth', () => {
    const s = ramp(8, false);
    const out = augmentSample(s, { ...NO_AUGMENT, brightness: 0.3 }, mulberry32(3));
    expect(Array.from(out.depth)).toEqual(Array.from(s.depth));
    let changed = false;
    for (let i = 0; i < out.image.length; i++) {
      if (Math.abs(out.image[i] - s.image[i]) > 1e-6) changed = true;
      expect(out.image[i]).toBeGreaterThanOrEqual(0);
      expect(out.image[i]).toBeLessThanOrEqual(1);
    }
    expect(changed).toBe(true);
  });

  it('is reproducible for a given seed', () => {
    const s = ramp(12, true);
    const a = augmentSample(s, DEFAULT_AUGMENT, mulberry32(42));
    const b = augmentSample(s, DEFAULT_AUGMENT, mulberry32(42));
    expect(Array.from(a.image)).toEqual(Array.from(b.image));
    expect(Array.from(a.depth)).toEqual(Array.from(b.depth));
  });

  it('keeps masks binary through a geometric warp', () => {
    const s = ramp(16, true);
    const out = augmentSample(s, { ...NO_AUGMENT, hflip: 1, rotate: 0.3, zoom: 0.1, translate: 0.1 }, mulberry32(5));
    for (const v of out.mask!) expect(v === 0 || v === 1).toBe(true);
  });

  it('cutout zeroes an image patch and removes its supervision', () => {
    const size = 24;
    const s: Sample = {
      image: new Float32Array(size * size).fill(0.5),
      depth: new Float32Array(size * size).fill(0.5),
      size,
      mask: null,
    };
    const out = augmentSample(s, { ...NO_AUGMENT, cutout: 1 }, mulberry32(9));
    let zeros = 0;
    let unlabelled = 0;
    for (let i = 0; i < out.image.length; i++) {
      if (out.image[i] === 0) zeros++;
      if (out.mask && out.mask[i] === 0) unlabelled++;
    }
    expect(zeros).toBeGreaterThan(0);
    expect(unlabelled).toBe(zeros);
  });

  it('augmentDataset expands by the requested factor', () => {
    const data = [ramp(8, false), ramp(8, false)];
    expect(augmentDataset(data, DEFAULT_AUGMENT, 3, 1).length).toBe(6);
  });
});
