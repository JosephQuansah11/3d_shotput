import { describe, it, expect } from 'vitest';
import { Tensor } from './tensor';
import { MSE, MaskedMSEWithTV } from './loss';

/** Numerically check dLoss/dPrediction against the analytic gradient. */
function checkLossGrad(loss: MSE | MaskedMSEWithTV, mask: Tensor | null, eps = 1e-6, tol = 1e-5): void {
  const pred = Tensor.randn([1, 5, 5], 1);
  const target = Tensor.randn([1, 5, 5], 1);

  const { grad } = loss.compute(pred, target, mask);
  const base = pred.data.slice();

  for (let i = 0; i < base.length; i++) {
    pred.data.set(base);
    pred.data[i] = base[i] + eps;
    const lp = loss.compute(pred, target, mask).loss;
    pred.data[i] = base[i] - eps;
    const lm = loss.compute(pred, target, mask).loss;
    pred.data[i] = base[i];
    const numeric = (lp - lm) / (2 * eps);
    expect(Math.abs(numeric - grad.data[i])).toBeLessThan(tol);
  }
}

describe('masked losses', () => {
  it('MSE gradient is correct with a mask', () => {
    const mask = Tensor.fill([1, 5, 5], 0);
    // label only a few pixels
    for (const i of [0, 3, 7, 12, 24]) mask.data[i] = 1;
    checkLossGrad(new MSE(), mask);
  });

  it('MaskedMSEWithTV gradient (data + smoothness) is correct', () => {
    const mask = Tensor.fill([1, 5, 5], 0);
    for (const i of [6, 8, 16, 18]) mask.data[i] = 1;
    checkLossGrad(new MaskedMSEWithTV(0.1), mask, 1e-6, 2e-5);
  });

  it('ignores unlabelled pixels entirely', () => {
    const pred = Tensor.fill([1, 3, 3], 0.5);
    const target = Tensor.fill([1, 3, 3], 0.5);
    // change a target value where mask is 0 -> loss must not move
    const mask = Tensor.fill([1, 3, 3], 0);
    mask.data[4] = 1;
    target.data[0] = 0.99;
    const l = new MSE().compute(pred, target, mask).loss;
    expect(l).toBeCloseTo(0, 10);
  });
});
