/**
 * tensor.ts — a tiny n-dimensional array library, written by hand.
 *
 * A Tensor is just a flat Float64Array plus a shape. We implement only the
 * primitives the neural network actually needs: elementwise ops, matrix
 * multiply, and `im2col` (the classic trick that turns a convolution into a
 * single matrix multiply). Everything is explicit — there is no autograd
 * framework hiding the calculus.
 */

export type Shape = number[];

export class Tensor {
  data: Float64Array;
  shape: Shape;

  constructor(data: Float64Array, shape: Shape) {
    this.data = data;
    this.shape = shape;
  }

  static zeros(shape: Shape): Tensor {
    return new Tensor(new Float64Array(numel(shape)), shape.slice());
  }

  static fill(shape: Shape, value: number): Tensor {
    const d = new Float64Array(numel(shape));
    d.fill(value);
    return new Tensor(d, shape.slice());
  }

  /** Standard normal (Box–Muller). Used to initialise weights. */
  static randn(shape: Shape, std = 1): Tensor {
    const n = numel(shape);
    const d = new Float64Array(n);
    for (let i = 0; i < n; i += 2) {
      const u1 = Math.random() || 1e-12;
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1)) * std;
      d[i] = r * Math.cos(2 * Math.PI * u2);
      if (i + 1 < n) d[i + 1] = r * Math.sin(2 * Math.PI * u2);
    }
    return new Tensor(d, shape.slice());
  }

  get size(): number {
    return this.data.length;
  }

  clone(): Tensor {
    return new Tensor(this.data.slice(), this.shape.slice());
  }

  /** Reshape without copying (reuses the same buffer). */
  reshape(shape: Shape): Tensor {
    const n = numel(shape);
    if (n !== this.data.length) {
      throw new Error(`reshape: cannot reshape ${this.data.length} into [${shape}]`);
    }
    return new Tensor(this.data, shape.slice());
  }

  /** Index into a 3D tensor [C, H, W]. */
  at3(c: number, y: number, x: number): number {
    const [, H, W] = this.shape;
    return this.data[(c * H + y) * W + x];
  }
}

export function numel(shape: Shape): number {
  return shape.reduce((a, b) => a * b, 1);
}

/* ------------------------------------------------------------------ */
/* Elementwise operations (allocate new tensors)                       */
/* ------------------------------------------------------------------ */

export function add(a: Tensor, b: Tensor): Tensor {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] + b.data[i];
  return new Tensor(out, a.shape.slice());
}

export function sub(a: Tensor, b: Tensor): Tensor {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] - b.data[i];
  return new Tensor(out, a.shape.slice());
}

export function mulScalar(a: Tensor, s: number): Tensor {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] * s;
  return new Tensor(out, a.shape.slice());
}

export function hadamard(a: Tensor, b: Tensor): Tensor {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] * b.data[i];
  return new Tensor(out, a.shape.slice());
}

export function map_(a: Tensor, f: (x: number) => number): Tensor {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = f(a.data[i]);
  return new Tensor(out, a.shape.slice());
}

export function sum(a: Tensor): number {
  let s = 0;
  for (let i = 0; i < a.data.length; i++) s += a.data[i];
  return s;
}

/* ------------------------------------------------------------------ */
/* Matrix multiply                                                     */
/* ------------------------------------------------------------------ */

export function matmul(
  A: Float64Array, M: number, K: number,
  B: Float64Array, N: number,
): Float64Array {
  const C = new Float64Array(M * N);
  for (let i = 0; i < M; i++) {
    const aRow = i * K;
    const cRow = i * N;
    for (let k = 0; k < K; k++) {
      const a = A[aRow + k];
      if (a === 0) continue;
      const bRow = k * N;
      for (let j = 0; j < N; j++) C[cRow + j] += a * B[bRow + j];
    }
  }
  return C;
}

/** C[M,N] = A[M,K] * B[N,K]^T (B given as N rows of length K). */
export function matmulNT(
  A: Float64Array, M: number, K: number,
  B: Float64Array, N: number,
): Float64Array {
  const C = new Float64Array(M * N);
  for (let i = 0; i < M; i++) {
    const aRow = i * K;
    for (let j = 0; j < N; j++) {
      const bRow = j * K;
      let s = 0;
      for (let k = 0; k < K; k++) s += A[aRow + k] * B[bRow + k];
      C[i * N + j] = s;
    }
  }
  return C;
}

/** C[M,N] = A[K,M]^T * B[K,N] (A and B share the reduction dim as rows). */
export function matmulTN(
  A: Float64Array, M: number, K: number,
  B: Float64Array, N: number,
): Float64Array {
  const C = new Float64Array(M * N);
  for (let k = 0; k < K; k++) {
    const aRow = k * M;
    const bRow = k * N;
    for (let i = 0; i < M; i++) {
      const a = A[aRow + i];
      if (a === 0) continue;
      const cRow = i * N;
      for (let j = 0; j < N; j++) C[cRow + j] += a * B[bRow + j];
    }
  }
  return C;
}

/* ------------------------------------------------------------------ */
/* im2col — convolution expressed as matrix multiplication             */
/* ------------------------------------------------------------------ */

export interface Im2ColInfo {
  cols: number; // C*KH*KW
  rows: number; // OH*OW
  outH: number;
  outW: number;
  inH: number;
  inW: number;
  channels: number;
  kH: number;
  kW: number;
  stride: number;
  pad: number;
}

export interface Im2ColResult extends Im2ColInfo {
  /** [C*KH*KW, OH*OW] */
  col: Float64Array;
}

/** Extract sliding patches from a [C, H, W] input into a matrix. */
export function im2col(
  input: Float64Array, C: number, H: number, W: number,
  kH: number, kW: number, stride: number, pad: number,
): Im2ColResult {
  const outH = Math.floor((H + 2 * pad - kH) / stride) + 1;
  const outW = Math.floor((W + 2 * pad - kW) / stride) + 1;
  const cols = C * kH * kW;
  const rows = outH * outW;
  const col = new Float64Array(cols * rows);

  let idx = 0;
  for (let c = 0; c < C; c++) {
    for (let ky = 0; ky < kH; ky++) {
      for (let kx = 0; kx < kW; kx++) {
        const patchRow = idx++;
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            const iy = oy * stride - pad + ky;
            const ix = ox * stride - pad + kx;
            let v = 0;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
              v = input[(c * H + iy) * W + ix];
            }
            col[patchRow * rows + oy * outW + ox] = v;
          }
        }
      }
    }
  }

  return { col, cols, rows, outH, outW, inH: H, inW: W, channels: C, kH, kW, stride, pad };
}

/** Gradient of im2col: scatter patch gradients back into an image buffer. */
export function col2im(dcol: Float64Array, info: Im2ColInfo): Float64Array {
  const { channels: C, inH: H, inW: W, kH, kW, stride, pad, outH, outW, rows } = info;
  const dimg = new Float64Array(C * H * W);
  let idx = 0;
  for (let c = 0; c < C; c++) {
    for (let ky = 0; ky < kH; ky++) {
      for (let kx = 0; kx < kW; kx++) {
        const patchRow = idx++;
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            const iy = oy * stride - pad + ky;
            const ix = ox * stride - pad + kx;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
              dimg[(c * H + iy) * W + ix] += dcol[patchRow * rows + oy * outW + ox];
            }
          }
        }
      }
    }
  }
  return dimg;
}

/* ------------------------------------------------------------------ */
/* Serialization                                                       */
/* ------------------------------------------------------------------ */

export interface SerializedTensor {
  shape: Shape;
  data: number[];
}

export const serialize = (t: Tensor): SerializedTensor => ({
  shape: t.shape.slice(),
  data: Array.from(t.data),
});

export const deserialize = (s: SerializedTensor): Tensor =>
  new Tensor(Float64Array.from(s.data), s.shape.slice());

