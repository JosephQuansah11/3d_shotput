import { describe, it, expect } from 'vitest';
import { Tensor, matmul, im2col, col2im, Im2ColResult } from './tensor';
import { Dense, Conv2D, ReLU, Layer } from './layers';
import { Sequential } from './model';
import { MSE } from './loss';
import { Adam } from './optim';

/* ------------------------------------------------------------------ */
/* Numerical gradient checking                                         */
/* ------------------------------------------------------------------ */

/**
 * Verify analytic gradients against central finite differences. This is the
 * strongest possible test that our hand-derived backprop is correct.
 */
function checkGradients(layers: Layer[], inputShape: number[], eps = 1e-6, tol = 1e-5): void {
  const x = Tensor.randn(inputShape, 1);
  const forward = () => {
    let out = x;
    for (const l of layers) out = l.forward(out);
    return out;
  };
  const backward = (dy: Tensor) => {
    let g = dy;
    for (let i = layers.length - 1; i >= 0; i--) g = layers[i].backward(g);
    return g;
  };

  const out = forward();
  const dy = Tensor.randn(out.shape, 1);

  const scalar = () => {
    const y = forward();
    let s = 0;
    for (let i = 0; i < y.data.length; i++) s += y.data[i] * dy.data[i];
    return s;
  };

  const params = layers.flatMap((l) => l.params());
  for (const p of params) p.grad.data.fill(0);
  forward();
  backward(dy); // populates analytic grads

  for (const p of params) {
    const data = p.value.data;
    for (let i = 0; i < data.length; i++) {
      const orig = data[i];
      data[i] = orig + eps;
      const lp = scalar();
      data[i] = orig - eps;
      const lm = scalar();
      data[i] = orig;
      const numeric = (lp - lm) / (2 * eps);
      expect(Math.abs(numeric - p.grad.data[i])).toBeLessThan(tol);
    }
  }
}

describe('tensor primitives', () => {
  it('multiplies matrices correctly', () => {
    const A = new Float64Array([1, 2, 3, 4, 5, 6]); // 2x3
    const B = new Float64Array([7, 8, 9, 10, 11, 12]); // 3x2
    const C = matmul(A, 2, 3, B, 2);
    expect(Array.from(C)).toEqual([58, 64, 139, 154]);
  });

  it('round-trips im2col with col2im', () => {
    const C = 1;
    const H = 5;
    const W = 5;
    const img = new Float64Array(C * H * W);
    for (let i = 0; i < img.length; i++) img[i] = i;
    const res: Im2ColResult = im2col(img, C, H, W, 3, 3, 1, 1);
    const back = col2im(res.col, res);
    // an interior pixel is covered by 9 patches
    expect(back[2 * W + 2]).toBeCloseTo(img[2 * W + 2] * 9, 5);
  });
});

describe('layer gradients', () => {
  it('matches finite differences for Dense', () => {
    checkGradients([new Dense(6, 4)], [6]);
  });

  it('matches finite differences for Conv2D', () => {
    checkGradients([new Conv2D(1, 2, 3, 3, 1, 1)], [1, 4, 4], 1e-6, 2e-5);
  });

  it('matches finite differences for Conv2D + ReLU', () => {
    checkGradients([new Conv2D(1, 2, 3, 3, 1, 1), new ReLU()], [1, 4, 4], 1e-6, 2e-5);
  });
});

describe('training reduces loss', () => {
  it('Adam makes the loss go down on a tiny regression task', () => {
    const model = new Sequential([new Dense(3, 8), new ReLU(), new Dense(8, 1)]);
    const loss = new MSE();
    const opt = new Adam();
    const x = new Tensor(new Float64Array([0.2, 0.5, 0.9]), [3]);
    const target = new Tensor(new Float64Array([1.0]), [1]);

    const first = loss.compute(model.forward(x), target).loss;
    for (let i = 0; i < 200; i++) {
      const pred = model.forward(x);
      const { grad } = loss.compute(pred, target);
      model.zeroGrad();
      model.backward(grad);
      opt.step(model.params(), 0.05);
    }
    const last = loss.compute(model.forward(x), target).loss;
    expect(last).toBeLessThan(first);
    expect(last).toBeLessThan(0.05);
  });
});
