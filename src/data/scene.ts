/**
 * data/scene.ts — a tiny ray tracer, written from scratch.
 *
 * Its job is to manufacture *supervised* training data: for every random 3D
 * scene we can produce (a) the shaded image a camera would see, and (b) the
 * exact depth of every pixel. The network then learns the mapping
 * image -> depth that we could never label by hand.
 */

import { Vec3, v3, add3, sub3, scale3, dot3, normalize3 } from '../math/vec';

export interface Material {
  albedo: number; // grayscale reflectance in [0,1]
}

export interface Sphere {
  kind: 'sphere';
  center: Vec3;
  radius: number;
  material: Material;
}

export interface Plane {
  kind: 'plane';
  point: Vec3;
  normal: Vec3;
  material: Material;
}

export type Primitive = Sphere | Plane;

export interface Light {
  direction: Vec3; // pointing *towards* the light
  intensity: number;
}

export interface Scene {
  primitives: Primitive[];
  light: Light;
  ambient: number;
  /** background intensity */
  background: number;
}

export interface Hit {
  t: number; // distance along the ray
  point: Vec3;
  normal: Vec3;
  material: Material;
}

/* ------------------------------------------------------------------ */
/* Intersections                                                       */
/* ------------------------------------------------------------------ */

function intersectSphere(origin: Vec3, dir: Vec3, s: Sphere): number | null {
  const oc = sub3(origin, s.center);
  const b = dot3(oc, dir);
  const c = dot3(oc, oc) - s.radius * s.radius;
  const disc = b * b - c;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  const t0 = -b - sq;
  const t1 = -b + sq;
  if (t0 > 1e-4) return t0;
  if (t1 > 1e-4) return t1;
  return null;
}

function intersectPlane(origin: Vec3, dir: Vec3, p: Plane): number | null {
  const denom = dot3(p.normal, dir);
  if (Math.abs(denom) < 1e-6) return null;
  const t = dot3(sub3(p.point, origin), p.normal) / denom;
  return t > 1e-4 ? t : null;
}

/** Nearest intersection of a ray with the scene, or null for a miss. */
export function trace(scene: Scene, origin: Vec3, dir: Vec3): Hit | null {
  let best: Hit | null = null;
  for (const prim of scene.primitives) {
    let t: number | null = null;
    let point: Vec3 = v3();
    let normal: Vec3 = v3();

    if (prim.kind === 'sphere') {
      t = intersectSphere(origin, dir, prim);
      if (t !== null) {
        point = add3(origin, scale3(dir, t));
        normal = normalize3(sub3(point, prim.center));
      }
    } else {
      t = intersectPlane(origin, dir, prim);
      if (t !== null) {
        point = add3(origin, scale3(dir, t));
        normal = prim.normal;
      }
    }

    if (t !== null && (!best || t < best.t)) {
      best = { t, point, normal, material: prim.material };
    }
  }
  return best;
}

/** Lambert shading: ambient + diffuse against the single scene light. */
export function shade(scene: Scene, hit: Hit): number {
  const ndl = Math.max(0, dot3(hit.normal, scene.light.direction));
  const intensity = scene.ambient + scene.light.intensity * ndl;
  return Math.min(1, hit.material.albedo * intensity);
}

/* ------------------------------------------------------------------ */
/* Scene generation                                                    */
/* ------------------------------------------------------------------ */

const rand = (a: number, b: number): number => a + Math.random() * (b - a);

/**
 * Build a random scene: a ground plane plus a handful of spheres of varying
 * size and position. These simple primitives give the network a rich variety
 * of silhouettes, occlusions and depth gradients to learn from.
 */
export function randomScene(): Scene {
  const primitives: Primitive[] = [];

  primitives.push({
    kind: 'plane',
    point: v3(0, -0.6, 0),
    normal: v3(0, 1, 0),
    material: { albedo: rand(0.25, 0.55) },
  });

  const count = Math.floor(rand(2, 5));
  for (let i = 0; i < count; i++) {
    primitives.push({
      kind: 'sphere',
      center: v3(rand(-1.1, 1.1), rand(-0.4, 0.5), rand(-2.4, -0.6)),
      radius: rand(0.18, 0.5),
      material: { albedo: rand(0.5, 1.0) },
    });
  }

  const lightDir = normalize3(v3(rand(-0.6, 0.6), rand(0.4, 1), rand(-0.6, 0.3)));
  return {
    primitives,
    light: { direction: lightDir, intensity: rand(0.7, 1.0) },
    ambient: rand(0.15, 0.3),
    background: rand(0.05, 0.2),
  };
}
