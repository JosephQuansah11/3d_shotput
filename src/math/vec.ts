/**
 * vec.ts — hand-written vector math.
 *
 * Everything here is plain arithmetic on plain objects. No libraries, no
 * typed-array magic beyond what we write ourselves. This is the foundation
 * layer of the whole engine: geometry, the renderer, the CNN and the camera
 * all speak in these types.
 */

export interface Vec2 {
  x: number;
  y: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Vec4 {
  x: number;
  y: number;
  z: number;
  w: number;
}

/* ------------------------------------------------------------------ */
/* Constructors                                                        */
/* ------------------------------------------------------------------ */

export const v2 = (x = 0, y = 0): Vec2 => ({ x, y });
export const v3 = (x = 0, y = 0, z = 0): Vec3 => ({ x, y, z });
export const v4 = (x = 0, y = 0, z = 0, w = 0): Vec4 => ({ x, y, z, w });

export const ZERO3: Vec3 = v3(0, 0, 0);
export const ONE3: Vec3 = v3(1, 1, 1);

/* ------------------------------------------------------------------ */
/* Vec3 algebra                                                        */
/* ------------------------------------------------------------------ */

export const add3 = (a: Vec3, b: Vec3): Vec3 => v3(a.x + b.x, a.y + b.y, a.z + b.z);
export const sub3 = (a: Vec3, b: Vec3): Vec3 => v3(a.x - b.x, a.y - b.y, a.z - b.z);
export const mul3 = (a: Vec3, b: Vec3): Vec3 => v3(a.x * b.x, a.y * b.y, a.z * b.z);
export const scale3 = (a: Vec3, s: number): Vec3 => v3(a.x * s, a.y * s, a.z * s);
export const neg3 = (a: Vec3): Vec3 => v3(-a.x, -a.y, -a.z);

export const dot3 = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;

export const cross3 = (a: Vec3, b: Vec3): Vec3 =>
  v3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x,
  );

export const len3 = (a: Vec3): number => Math.sqrt(dot3(a, a));
export const lenSq3 = (a: Vec3): number => dot3(a, a);

export function normalize3(a: Vec3): Vec3 {
  const l = len3(a);
  return l > 1e-12 ? scale3(a, 1 / l) : v3(0, 0, 0);
}

export const dist3 = (a: Vec3, b: Vec3): number => len3(sub3(a, b));

/** Linear interpolation: t=0 -> a, t=1 -> b. */
export const lerp3 = (a: Vec3, b: Vec3, t: number): Vec3 =>
  v3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);

/* ------------------------------------------------------------------ */
/* Vec2 algebra                                                        */
/* ------------------------------------------------------------------ */

export const add2 = (a: Vec2, b: Vec2): Vec2 => v2(a.x + b.x, a.y + b.y);
export const sub2 = (a: Vec2, b: Vec2): Vec2 => v2(a.x - b.x, a.y - b.y);
export const scale2 = (a: Vec2, s: number): Vec2 => v2(a.x * s, a.y * s);
export const dot2 = (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y;
export const len2 = (a: Vec2): number => Math.hypot(a.x, a.y);
export const perp2 = (a: Vec2): Vec2 => v2(-a.y, a.x);

export function normalize2(a: Vec2): Vec2 {
  const l = len2(a);
  return l > 1e-12 ? scale2(a, 1 / l) : v2(0, 0);
}

/* ------------------------------------------------------------------ */
/* Clamp / misc helpers used across the renderer and CV code           */
/* ------------------------------------------------------------------ */

export const clamp = (x: number, lo: number, hi: number): number =>
  x < lo ? lo : x > hi ? hi : x;

export const clamp01 = (x: number): number => clamp(x, 0, 1);

export const smoothstep = (edge0: number, edge1: number, x: number): number => {
  const t = clamp01((x - edge0) / (edge1 - edge0 || 1e-12));
  return t * t * (3 - 2 * t);
};
