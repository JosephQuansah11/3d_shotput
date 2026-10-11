import { describe, it, expect } from 'vitest';
import { resizeBuffer } from './resize';
import { toTrainingSample, labelledFraction, RealSample } from './realDataset';

const makeSample = (size: number, withMask: boolean): RealSample => {
  const image = new Float32Array(size * size).fill(0.5);
  const depth = new Float32Array(size * size);
  for (let i = 0; i < depth.length; i++) depth[i] = (i % size) / size;
  const mask = withMask ? new Float32Array(size * size) : null;
  if (mask) {
    for (let i = 0; i < mask.length; i++) mask[i] = i % 2 === 0 ? 1 : 0;
  }
  return {
    id: 'test', name: 'x', createdAt: 0, size, image, depth, mask,
    thumbnail: '', source: withMask ? 'brush' : 'depth-map',
  };
};

describe('resizeBuffer', () => {
  it('is identity when sizes match', () => {
    const data = new Float32Array([1, 2, 3, 4]);
    expect(Array.from(resizeBuffer(data, 2, 2))).toEqual([1, 2, 3, 4]);
  });

  it('downsamples a linear ramp faithfully (bilinear)', () => {
    const from = 4;
    const data = new Float32Array(from * from);
    for (let y = 0; y < from; y++) for (let x = 0; x < from; x++) data[y * from + x] = x / (from - 1);
    const out = resizeBuffer(data, from, 2, 'bilinear');
    // left column ~0, right column ~1
    expect(out[0]).toBeLessThan(0.35);
    expect(out[1]).toBeGreaterThan(0.65);
  });

  it('nearest keeps mask values binary', () => {
    const mask = new Float32Array(9).fill(0);
    mask[4] = 1;
    const out = resizeBuffer(mask, 3, 6, 'nearest');
    for (const v of out) expect(v === 0 || v === 1).toBe(true);
  });
});

describe('real dataset helpers', () => {
  it('resamples a stored sample to the model size', () => {
    const s = makeSample(8, true);
    const t = toTrainingSample(s, 16);
    expect(t.size).toBe(16);
    expect(t.image.length).toBe(256);
    expect(t.mask?.length).toBe(256);
  });

  it('reports labelled fraction (1 for dense, ~0.5 for half-painted)', () => {
    expect(labelledFraction(makeSample(4, false))).toBe(1);
    expect(labelledFraction(makeSample(8, true))).toBeCloseTo(0.5, 5);
  });
});
