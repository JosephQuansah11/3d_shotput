/**
 * model.ts — a sequential container that chains layers.
 *
 * Each layer caches whatever it needs during forward, so backward is simply
 * the chain rule applied in reverse order.
 */

import { Tensor } from './tensor';
import { Layer, Param } from './layers';

export class Sequential {
  layers: Layer[];

  constructor(layers: Layer[] = []) {
    this.layers = layers;
  }

  add(layer: Layer): this {
    this.layers.push(layer);
    return this;
  }

  forward(x: Tensor): Tensor {
    let out = x;
    for (const layer of this.layers) out = layer.forward(out);
    return out;
  }

  /** Backprop: dy is dLoss/dOutput. Returns dLoss/dInput. */
  backward(dy: Tensor): Tensor {
    let grad = dy;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad);
    }
    return grad;
  }

  params(): Param[] {
    const out: Param[] = [];
    for (const layer of this.layers) out.push(...layer.params());
    return out;
  }

  zeroGrad(): void {
    for (const p of this.params()) p.grad.data.fill(0);
  }

  /** Total number of trainable scalars. */
  numParams(): number {
    return this.params().reduce((a, p) => a + p.value.size, 0);
  }
  /** Flatten all parameters into plain arrays (order is stable per layer). */
  exportWeights(): number[][] {
    return this.params().map((p) => Array.from(p.value.data));
  }

  /** Load parameters previously produced by exportWeights(). */
  importWeights(weights: number[][]): void {
    const params = this.params();
    if (weights.length !== params.length) {
      throw new Error(`importWeights: expected ${params.length} tensors, got ${weights.length}`);
    }
    for (let i = 0; i < params.length; i++) {
      const w = weights[i];
      if (w.length !== params[i].value.size) {
        throw new Error(`importWeights: tensor ${i} size mismatch`);
      }
      params[i].value.data.set(w);
    }
  }

}
