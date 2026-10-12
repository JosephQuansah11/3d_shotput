import { describe, it, expect } from 'vitest';
import { v3, cross3, dot3, normalize3, len3 } from './vec';
import { mat4Identity, mul4, invert4, mulVec4, perspective, lookAt } from './mat';
import { quatFromAxisAngle, quatRotate, quatToMat3 } from './quat';

const close = (a: number, b: number, eps = 1e-9) => Math.abs(a - b) < eps;

describe('vec3', () => {
  it('computes cross products with the right-hand rule', () => {
    const c = cross3(v3(1, 0, 0), v3(0, 1, 0));
    expect(close(c.x, 0)).toBe(true);
    expect(close(c.y, 0)).toBe(true);
    expect(close(c.z, 1)).toBe(true);
  });

  it('normalizes to unit length', () => {
    expect(close(len3(normalize3(v3(3, 4, 12))), 1)).toBe(true);
  });

  it('dot of perpendicular vectors is zero', () => {
    expect(close(dot3(v3(1, 2, 3), cross3(v3(1, 2, 3), v3(4, 5, 6))), 0)).toBe(true);
  });
});

describe('mat4', () => {
  it('multiplies by the identity without change', () => {
    const m = perspective(1, 1.5, 0.1, 100);
    const r = mul4(m, mat4Identity());
    for (let i = 0; i < 16; i++) expect(close(r[i], m[i], 1e-12)).toBe(true);
  });

  it('inverts a matrix: A * A^-1 = I', () => {
    const m = lookAt(v3(3, 2, 4), v3(0, 0, 0), v3(0, 1, 0));
    const inv = invert4(m);
    const prod = mul4(m, inv);
    const id = mat4Identity();
    for (let i = 0; i < 16; i++) expect(close(prod[i], id[i], 1e-9)).toBe(true);
  });

  it('applies a translation', () => {
    const m = lookAt(v3(0, 0, 5), v3(0, 0, 0), v3(0, 1, 0));
    // a point at the origin should sit on the camera axis
    const p = mulVec4(m, { x: 0, y: 0, z: 0, w: 1 });
    expect(close(p.x, 0, 1e-9)).toBe(true);
    expect(close(p.y, 0, 1e-9)).toBe(true);
    expect(close(p.z, -5, 1e-9)).toBe(true);
  });
});

describe('quaternion', () => {
  it('rotates a vector 90° about Y', () => {
    const q = quatFromAxisAngle(v3(0, 1, 0), Math.PI / 2);
    const r = quatRotate(q, v3(1, 0, 0));
    expect(close(r.x, 0, 1e-9)).toBe(true);
    expect(close(r.z, -1, 1e-9)).toBe(true);
  });

  it('agrees with its rotation matrix form', () => {
    const q = quatFromAxisAngle(normalize3(v3(1, 2, 3)), 0.7);
    const m = quatToMat3(q);
    const v = v3(0.3, -0.5, 0.9);
    const byQuat = quatRotate(q, v);
    const byMat = {
      x: m[0] * v.x + m[3] * v.y + m[6] * v.z,
      y: m[1] * v.x + m[4] * v.y + m[7] * v.z,
      z: m[2] * v.x + m[5] * v.y + m[8] * v.z,
    };
    expect(close(byQuat.x, byMat.x, 1e-9)).toBe(true);
    expect(close(byQuat.y, byMat.y, 1e-9)).toBe(true);
    expect(close(byQuat.z, byMat.z, 1e-9)).toBe(true);
  });
});
