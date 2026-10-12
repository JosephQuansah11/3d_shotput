/**
 * camera.ts — an orbit camera built from the primitives in mat.ts.
 *
 * The camera owns three pieces of state: a target point it looks at, an
 * azimuth/elevation orientation around that target, and a distance. From those
 * we derive eye position, the view matrix, and the projection matrix.
 */

import { Vec3, v3 } from './vec';
import { Mat4, lookAt, mul4, perspective } from './mat';

export interface CameraState {
  target: Vec3;
  /** azimuth in radians (around Y) */
  azimuth: number;
  /** elevation in radians (up/down) */
  elevation: number;
  distance: number;
  fovY: number;
  near: number;
  far: number;
}

export function defaultCamera(): CameraState {
  return {
    target: v3(0, 0, 0),
    azimuth: Math.PI * 0.25,
    elevation: Math.PI * 0.15,
    distance: 3.2,
    fovY: (50 * Math.PI) / 180,
    near: 0.05,
    far: 100,
  };
}

/** World-space eye position derived from the orbit parameters. */
export function cameraEye(c: CameraState): Vec3 {
  const ce = Math.cos(c.elevation);
  return v3(
    c.target.x + c.distance * ce * Math.sin(c.azimuth),
    c.target.y + c.distance * Math.sin(c.elevation),
    c.target.z + c.distance * ce * Math.cos(c.azimuth),
  );
}

export function viewMatrix(c: CameraState): Mat4 {
  return lookAt(cameraEye(c), c.target, v3(0, 1, 0));
}

export function projectionMatrix(c: CameraState, aspect: number): Mat4 {
  return perspective(c.fovY, aspect, c.near, c.far);
}

export function viewProjection(c: CameraState, aspect: number): Mat4 {
  return mul4(projectionMatrix(c, aspect), viewMatrix(c));
}
