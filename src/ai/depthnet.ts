/**
 * ai/depthnet.ts — the model itself.
 *
 * A small encoder/decoder ("U-Net style") convolutional network that maps a
 * single-channel image to a single-channel depth map. Every layer is one of the
 * hand-written layers in layers.ts, so the whole forward and backward pass is
 * our own maths.
 *
 *   encoder: conv -> pool -> conv -> pool -> conv
 *   decoder: upsample -> conv -> upsample -> conv -> sigmoid
 */

import { Tensor } from './tensor';
import { Sequential } from './model';
import { Conv2D, ReLU, Sigmoid, MaxPool2D, Upsample2D } from './layers';
import { Sample } from '../data/dataset';

export interface DepthNetOptions {
  /** input/output resolution (must be divisible by 4) */
  size: number;
  baseChannels?: number;
}

export interface DepthNet {
  model: Sequential;
  size: number;
  describe(): string;
}

export function createDepthNet(opts: DepthNetOptions): DepthNet {
  const size = opts.size;
  const c1 = opts.baseChannels ?? 8;
  const c2 = c1 * 2;

  const model = new Sequential([
    // --- encoder ---
    new Conv2D(1, c1, 3, 3, 1, 1),
    new ReLU(),
    new MaxPool2D(2),
    new Conv2D(c1, c2, 3, 3, 1, 1),
    new ReLU(),
    new MaxPool2D(2),
    new Conv2D(c2, c2, 3, 3, 1, 1),
    new ReLU(),
    // --- decoder ---
    new Upsample2D(2),
    new Conv2D(c2, c1, 3, 3, 1, 1),
    new ReLU(),
    new Upsample2D(2),
    new Conv2D(c1, 1, 3, 3, 1, 1),
    new Sigmoid(),
  ]);

  return {
    model,
    size,
    describe: () =>
      `DepthNet(${size}x${size}) — ${model.numParams().toLocaleString()} parameters, ` +
      `${model.layers.length} layers`,
  };
}

/** image [size*size] -> depth tensor [1, size, size] */
export function predictDepth(net: DepthNet, image: Float32Array): Tensor {
  const input = new Tensor(Float64Array.from(image), [1, net.size, net.size]);
  return net.model.forward(input);
}

/** Convenience: turn a training Sample into (input, target) tensors. */
export function sampleToTensors(sample: Sample): { input: Tensor; target: Tensor } {
  const input = new Tensor(Float64Array.from(sample.image), [1, sample.size, sample.size]);
  const target = new Tensor(Float64Array.from(sample.depth), [1, sample.size, sample.size]);
  return { input, target };
}
