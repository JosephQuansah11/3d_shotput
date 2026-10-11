/**
 * quat.ts — hand-written quaternion math.
 *
 * Quaternions give us smooth, gimbal-lock-free rotation for orbit controls.
 * Convention: q = (x, y, z, w) with w the scalar part, and a unit quaternion
 * represents a rotation by angle t about unit axis a as
 *   q = (a * sin(t/2), cos(t/2)).
 */

import { Vec3, v3 } from './vec';

export interface Quat {
  x: number;
  y: number;
  z: number;
  w: number;
}

export const quat = (x = 0, y = 0, z = 0, w = 1): Quat => ({ x, y, z, w });
export const QUAT_IDENTITY: Quat = quat();

export function quatFromAxisAngle(axis: Vec3, rad: number): Quat {
  const l = Math.hypot(axis.x, axis.y, axis.z);
  if (l < 1e-12) return quat();
  const half = rad / 2;
  const s = Math.sin(half) / l;
  return quat(axis.x * s, axis.y * s, axis.z * s, Math.cos(half));
}

export function quatMul(a: Quat, b: Quat): Quat {
  return quat(
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  );
}

export function quatConjugate(q: Quat): Quat {
  return quat(-q.x, -q.y, -q.z, q.w);
}

export function quatNormalize(q: Quat): Quat {
  const l = Math.hypot(q.x, q.y, q.z, q.w);
  return l > 1e-12 ? quat(q.x / l, q.y / l, q.z / l, q.w / l) : quat();
}

/** Rotate a vector by a unit quaternion: v' = q v q*. */
export function quatRotate(q: Quat, v: Vec3): Vec3 {
  // t = 2 * (q_vec x v)
  const tx = 2 * (q.y * v.z - q.z * v.y);
  const ty = 2 * (q.z * v.x - q.x * v.z);
  const tz = 2 * (q.x * v.y - q.y * v.x);
  return v3(
    v.x + q.w * tx + (q.y * tz - q.z * ty),
    v.y + q.w * ty + (q.z * tx - q.x * tz),
    v.z + q.w * tz + (q.x * ty - q.y * tx),
  );
}

/** Convert to a column-major 3x3 rotation matrix (length 9). */
export function quatToMat3(q: Quat): Float64Array {
  const { x, y, z, w } = quatNormalize(q);
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return new Float64Array([
    1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy),
    2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx),
    2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy),
  ]);
}

/** Spherical linear interpolation between two unit quaternions. */
export function quatSlerp(a: Quat, b: Quat, t: number): Quat {
  let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  let bx = b.x, by = b.y, bz = b.z, bw = b.w;
  if (dot < 0) {
    dot = -dot;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }
  if (dot > 0.9995) {
    return quatNormalize(quat(
      a.x + (bx - a.x) * t,
      a.y + (by - a.y) * t,
      a.z + (bz - a.z) * t,
      a.w + (bw - a.w) * t,
    ));
  }
  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sinTheta;
  const wb = Math.sin(t * theta) / sinTheta;
  return quat(
    a.x * wa + bx * wb,
    a.y * wa + by * wb,
    a.z * wa + bz * wb,
    a.w * wa + bw * wb,
  );
}
