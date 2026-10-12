import { describe, it, expect } from 'vitest';
import { createDepthNet } from './depthnet';
import { train } from './trainer';
import { MSE } from './loss';
import { Adam } from './optim';
import { generateDataset } from '../data/dataset';

/**
 * End-to-end check: train the real network on ray-traced scenes and confirm the
 * loss actually goes down. This exercises the ray tracer, im2col convolution,
 * backprop and the optimizer together.
 */
describe('end-to-end training', () => {
  it('reduces loss on a small synthetic dataset', () => {
    const size = 16;
    const dataset = generateDataset(8, { size, fovY: (50 * Math.PI) / 180, near: 0.2, far: 4 });
    const net = createDepthNet({ size, baseChannels: 2 });
    const optimizer = new Adam();

    const before = train(net.model, new MSE(), optimizer, dataset, {
      epochs: 1, learningRate: 0.01,
    });
    const after = train(net.model, new MSE(), optimizer, dataset, {
      epochs: 4, learningRate: 0.01,
    });

    expect(before.history[0]).toBeGreaterThan(0);
    expect(after.history[after.history.length - 1]).toBeLessThan(before.history[0]);
  }, 120000);
});
