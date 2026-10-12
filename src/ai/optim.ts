/**
 * optim.ts — gradient descent optimizers.
 *
 * Both SGD (with momentum) and Adam are implemented directly from their
 * update equations so the maths is visible.
 */

import { Param } from './layers';

export interface Optimizer {
  readonly name: string;
  step(params: Param[], lr: number): void;
  reset(): void;
}

/** Stochastic gradient descent with momentum and optional weight decay. */
export class SGD implements Optimizer {
  readonly name = 'SGD';
  momentum: number;
  weightDecay: number;
  private velocity = new Map<Param, Float64Array>();

  constructor(momentum = 0.9, weightDecay = 0) {
    this.momentum = momentum;
    this.weightDecay = weightDecay;
  }

  step(params: Param[], lr: number): void {
    for (const p of params) {
      let v = this.velocity.get(p);
      if (!v || v.length !== p.value.size) {
        v = new Float64Array(p.value.size);
        this.velocity.set(p, v);
      }
      for (let i = 0; i < p.value.size; i++) {
        let g = p.grad.data[i];
        if (this.weightDecay) g += this.weightDecay * p.value.data[i];
        v[i] = this.momentum * v[i] - lr * g;
        p.value.data[i] += v[i];
      }
    }
  }

  reset(): void {
    this.velocity.clear();
  }
}

/** Adam: adaptive moment estimation (Kingma & Ba, 2014). */
export class Adam implements Optimizer {
  readonly name = 'Adam';
  beta1: number;
  beta2: number;
  eps: number;
  private m = new Map<Param, Float64Array>();
  private v = new Map<Param, Float64Array>();
  private t = 0;

  constructor(beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
  }

  step(params: Param[], lr: number): void {
    this.t += 1;
    const b1 = this.beta1;
    const b2 = this.beta2;
    const bias1 = 1 - Math.pow(b1, this.t);
    const bias2 = 1 - Math.pow(b2, this.t);

    for (const p of params) {
      let m = this.m.get(p);
      let v = this.v.get(p);
      if (!m || m.length !== p.value.size) {
        m = new Float64Array(p.value.size);
        v = new Float64Array(p.value.size);
        this.m.set(p, m);
        this.v.set(p, v);
      }
      const vv = v!;
      for (let i = 0; i < p.value.size; i++) {
        const g = p.grad.data[i];
        m[i] = b1 * m[i] + (1 - b1) * g;
        vv[i] = b2 * vv[i] + (1 - b2) * g * g;
        const mHat = m[i] / bias1;
        const vHat = vv[i] / bias2;
        p.value.data[i] -= (lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  reset(): void {
    this.m.clear();
    this.v.clear();
    this.t = 0;
  }
}
