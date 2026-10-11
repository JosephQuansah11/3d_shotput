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
}
