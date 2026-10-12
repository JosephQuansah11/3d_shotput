/**
 * loss.ts — loss functions and their derivatives.
 *
 * The derivative returned is dLoss/dPrediction, ready for
 * Sequential.backward(). Every loss accepts an optional `mask` tensor: where
 * mask is 0 the pixel is ignored. That is what lets the model learn from
 * *sparse* human labels — a few brush strokes instead of a full depth map.
 */

import { Tensor } from './tensor';

export interface Loss {
  readonly name: string;
  /** Returns [scalarLoss, gradientWrtPrediction]. */
  compute(pred: Tensor, target: Tensor, mask?: Tensor | null): { loss: number; grad: Tensor };
}

/* ------------------------------------------------------------------ */
/* Mean squared error (optionally masked)                              */
/* ------------------------------------------------------------------ */

export class MSE implements Loss {
  readonly name = 'MSE';

  compute(pred: Tensor, target: Tensor, mask?: Tensor | null): { loss: number; grad: Tensor } {
    const n = pred.data.length;
    const grad = new Float64Array(n);
    let loss = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      const m = mask ? mask.data[i] : 1;
      if (m <= 0) continue;
      const diff = pred.data[i] - target.data[i];
      loss += m * diff * diff;
      grad[i] = 2 * m * diff;
      count += m;
    }
    const inv = 1 / (count || 1);
    for (let i = 0; i < n; i++) grad[i] *= inv;
    return { loss: loss * inv, grad: new Tensor(grad, pred.shape.slice()) };
  }
}

/* ------------------------------------------------------------------ */
/* Mean absolute error (optionally masked)                             */
/* ------------------------------------------------------------------ */

export class MAE implements Loss {
  readonly name = 'MAE';

  compute(pred: Tensor, target: Tensor, mask?: Tensor | null): { loss: number; grad: Tensor } {
    const n = pred.data.length;
    const grad = new Float64Array(n);
    let loss = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      const m = mask ? mask.data[i] : 1;
      if (m <= 0) continue;
      const diff = pred.data[i] - target.data[i];
      loss += m * Math.abs(diff);
      grad[i] = m * Math.sign(diff);
      count += m;
    }
    const inv = 1 / (count || 1);
    for (let i = 0; i < n; i++) grad[i] *= inv;
    return { loss: loss * inv, grad: new Tensor(grad, pred.shape.slice()) };
  }
}

/* ------------------------------------------------------------------ */
/* Masked MSE + total-variation smoothness                             */
/* ------------------------------------------------------------------ */

/**
 * The workhorse for sparse labels. It combines:
 *
 *   L = MSE over labelled pixels  +  lambda * TV(prediction)
 *
 * Total variation penalises differences between neighbouring depths, so the
 * network interpolates smoothly between brush strokes instead of fitting
 * noise. TV is |dP/dx| + |dP/dy| summed over the image.
 */
export class MaskedMSEWithTV implements Loss {
  readonly name: string;
  lambda: number;

  constructor(lambda = 0.05) {
    this.lambda = lambda;
    this.name = `MaskedMSE+TV(λ=${lambda})`;
  }

  compute(pred: Tensor, target: Tensor, mask?: Tensor | null): { loss: number; grad: Tensor } {
    const [, h, w] = pred.shape;
    const n = pred.data.length;
    const grad = new Float64Array(n);

    // --- masked MSE term ---
    let dataLoss = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      const m = mask ? mask.data[i] : 1;
      if (m <= 0) continue;
      const diff = pred.data[i] - target.data[i];
      dataLoss += m * diff * diff;
      grad[i] += 2 * m * diff;
      count += m;
    }
    const inv = 1 / (count || 1);
    for (let i = 0; i < n; i++) grad[i] *= inv;
    dataLoss *= inv;

    // --- total-variation term ---
    const P = pred.data;
    let tv = 0;
    const tvScale = this.lambda / (h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (x + 1 < w) {
          const d = P[i + 1] - P[i];
          tv += Math.abs(d);
          const s = Math.sign(d) * tvScale;
          grad[i] -= s;
          grad[i + 1] += s;
        }
        if (y + 1 < h) {
          const d = P[i + w] - P[i];
          tv += Math.abs(d);
          const s = Math.sign(d) * tvScale;
          grad[i] -= s;
          grad[i + w] += s;
        }
      }
    }

    return {
      loss: dataLoss + this.lambda * (tv / (h * w)),
      grad: new Tensor(grad, pred.shape.slice()),
    };
  }
}
