/**
 * mat.ts — hand-written matrix math (column-major, 4x4 and 3x3).
 *
 * We use column-major storage to match the GPU convention so the same numbers
 * can be handed to WebGL/WebGPU later without reshuffling. Index helper:
 *
 *   m[col * 4 + row]
 *
 * A 4x4 matrix multiplies a column vector v as  m * v.
 */

import { Vec3, Vec4, v3, v4 } from './vec';

export type Mat4 = Float64Array; // length 16, column-major
export type Mat3 = Float64Array; // length 9,  column-major

/* ------------------------------------------------------------------ */
/* Constructors                                                        */
/* ------------------------------------------------------------------ */

export function mat4Identity(): Mat4 {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function mat4(
  m00: number, m01: number, m02: number, m03: number,
  m10: number, m11: number, m12: number, m13: number,
  m20: number, m21: number, m22: number, m23: number,
  m30: number, m31: number, m32: number, m33: number,
): Mat4 {
  const m = new Float64Array(16);
  // arguments are given row-major for readability, stored column-major
  m[0] = m00; m[4] = m01; m[8] = m02; m[12] = m03;
  m[1] = m10; m[5] = m11; m[9] = m12; m[13] = m13;
  m[2] = m20; m[6] = m21; m[10] = m22; m[14] = m23;
  m[3] = m30; m[7] = m31; m[11] = m32; m[15] = m33;
  return m;
}

export function mat3Identity(): Mat3 {
  const m = new Float64Array(9);
  m[0] = m[4] = m[8] = 1;
  return m;
}

/* ------------------------------------------------------------------ */
/* Accessors (row, col are 0-based)                                    */
/* ------------------------------------------------------------------ */

export const get4 = (m: Mat4, row: number, col: number): number => m[col * 4 + row];
export const set4 = (m: Mat4, row: number, col: number, val: number): void => {
  m[col * 4 + row] = val;
};

/* ------------------------------------------------------------------ */
/* Multiplication                                                      */
/* ------------------------------------------------------------------ */

/** out = a * b  (both 4x4). Writes into a fresh matrix. */
export function mul4(a: Mat4, b: Mat4): Mat4 {
  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + r] * b[c * 4 + k];
      }
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

export function mulMat3(a: Mat3, b: Mat3): Mat3 {
  const out = new Float64Array(9);
  for (let c = 0; c < 3; c++) {
    for (let r = 0; r < 3; r++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) sum += a[k * 3 + r] * b[c * 3 + k];
      out[c * 3 + r] = sum;
    }
  }
  return out;
}

/** Transform a 4-component column vector: out = m * v. */
export function mulVec4(m: Mat4, v: Vec4): Vec4 {
  return v4(
    m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w,
    m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
    m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
    m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w,
  );
}

/** Transform a point (w=1) and divide by w — full perspective divide. */
export function transformPoint(m: Mat4, p: Vec3): Vec4 {
  const out = mulVec4(m, v4(p.x, p.y, p.z, 1));
  if (Math.abs(out.w) > 1e-12) {
    out.x /= out.w;
    out.y /= out.w;
    out.z /= out.w;
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Builders used by the camera                                         */
/* ------------------------------------------------------------------ */

export const translation = (x: number, y: number, z: number): Mat4 =>
  mat4(
    1, 0, 0, x,
    0, 1, 0, y,
    0, 0, 1, z,
    0, 0, 0, 1,
  );

export const scaling = (x: number, y: number, z: number): Mat4 =>
  mat4(
    x, 0, 0, 0,
    0, y, 0, 0,
    0, 0, z, 0,
    0, 0, 0, 1,
  );

export const rotationX = (rad: number): Mat4 => {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return mat4(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1);
};

export const rotationY = (rad: number): Mat4 => {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return mat4(c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1);
};

export const rotationZ = (rad: number): Mat4 => {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return mat4(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
};

/** Right-handed perspective projection. fovY in radians. */
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return mat4(
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  );
}

/** Standard orthographic projection box. */
export function orthographic(
  left: number, right: number, bottom: number, top: number, near: number, far: number,
): Mat4 {
  const lr = 1 / (left - right);
  const bt = 1 / (bottom - top);
  const nf = 1 / (near - far);
  return mat4(
    -2 * lr, 0, 0, (left + right) * lr,
    0, -2 * bt, 0, (top + bottom) * bt,
    0, 0, 2 * nf, (far + near) * nf,
    0, 0, 0, 1,
  );
}

/** Orthonormal "look at" view matrix. */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const zAxis = norm(sub(eye, target)); // camera looks down -z
  const xAxis = norm(cross(up, zAxis));
  const yAxis = cross(zAxis, xAxis);

  // column-major -> basis vectors sit in the first three columns
  return mat4(
    xAxis.x, xAxis.y, xAxis.z, -dot(xAxis, eye),
    yAxis.x, yAxis.y, yAxis.z, -dot(yAxis, eye),
    zAxis.x, zAxis.y, zAxis.z, -dot(zAxis, eye),
    0, 0, 0, 1,
  );
}

/* ------------------------------------------------------------------ */
/* Inversion & transpose (inverse camera / normal matrices)            */
/* ------------------------------------------------------------------ */

export function transpose4(m: Mat4): Mat4 {
  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) out[c * 4 + r] = m[r * 4 + c];
  return out;
}

/** General 4x4 inverse (cofactor expansion). Returns identity if singular. */
export function invert4(m: Mat4): Mat4 {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) return mat4Identity();
  det = 1 / det;

  // result is written in the same flat column-major layout as the input
  const out = new Float64Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

/** Inverse-transpose of the upper-left 3x3 — correct matrix for normals. */
export function normalMatrix(model: Mat4): Mat3 {
  const inv = invert4(model);
  return new Float64Array([
    inv[0], inv[4], inv[8],
    inv[1], inv[5], inv[9],
    inv[2], inv[6], inv[10],
  ]);
}

export function applyMat3(m: Mat3, v: Vec3): Vec3 {
  return v3(
    m[0] * v.x + m[3] * v.y + m[6] * v.z,
    m[1] * v.x + m[4] * v.y + m[7] * v.z,
    m[2] * v.x + m[5] * v.y + m[8] * v.z,
  );
}

const sub = (a: Vec3, b: Vec3): Vec3 => v3(a.x - b.x, a.y - b.y, a.z - b.z);
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: Vec3, b: Vec3): Vec3 =>
  v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a.x, a.y, a.z);
  return l > 1e-12 ? v3(a.x / l, a.y / l, a.z / l) : v3(0, 0, 0);
};

