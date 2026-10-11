/**
 * loss.ts — loss functions and their derivatives.
 *
 * The derivative returned here is dLoss/dPrediction, ready to feed into
 * Sequential.backward().
 */

import { Tensor } from './tensor';

export interface Loss {
  readonly name: string;
  /** Returns [scalarLoss, gradientWrtPrediction]. */
  compute(pred: Tensor, target: Tensor): { loss: number; grad: Tensor };
}

export class MSE implements Loss {
  readonly name = 'MSE';

  compute(pred: Tensor, target: Tensor): { loss: number; grad: Tensor } {
    const n = pred.data.length;
    const grad = new Float64Array(n);
    let loss = 0;
    for (let i = 0; i < n; i++) {
      const diff = pred.data[i] - target.data[i];
      loss += diff * diff;
      grad[i] = (2 * diff) / n;
    }
    return { loss: loss / n, grad: new Tensor(grad, pred.shape.slice()) };
  }
}

export class MAE implements Loss {
  readonly name = 'MAE';

  compute(pred: Tensor, target: Tensor): { loss: number; grad: Tensor } {
    const n = pred.data.length;
    const grad = new Float64Array(n);
    let loss = 0;
    for (let i = 0; i < n; i++) {
      const diff = pred.data[i] - target.data[i];
      loss += Math.abs(diff);
      grad[i] = (Math.sign(diff) || 0) / n;
    }
    return { loss: loss / n, grad: new Tensor(grad, pred.shape.slice()) };
  }
}
