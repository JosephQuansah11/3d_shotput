/**
 * viewer/renderer.ts — a software 3D renderer drawing to a 2D canvas.
 *
 * No WebGL, no three.js. Every vertex is transformed by our own matrices,
 * projected with a perspective divide, sorted back-to-front (painter's
 * algorithm) and filled with flat Lambert shading. This is the "3d image
 * geometric viewer" at its core.
 */

import { Vec3, v3, dot3, normalize3 } from '../math/vec';
import { Mat4, mulVec4 } from '../math/mat';
import { CameraState, cameraEye } from '../math/camera';
import { PointCloud, Mesh } from '../cv/geometry';

export type SceneGeometry =
  | { kind: 'points'; cloud: PointCloud }
  | { kind: 'mesh'; mesh: Mesh }
  | { kind: 'wireframe'; mesh: Mesh };

export interface RenderStyle {
  background: string;
  pointSize: number;
  light: Vec3;
  ambient: number;
  /** multiply colors by shading */
  shading: boolean;
  /** draw depth-fade so far points dim out */
  depthFade: boolean;
}

export const DEFAULT_STYLE: RenderStyle = {
  background: '#0b0f17',
  pointSize: 2,
  light: normalize3(v3(0.4, 0.8, 0.6)),
  ambient: 0.25,
  shading: true,
  depthFade: true,
};

export interface Frame {
  viewProj: Mat4;
  eye: Vec3;
  width: number;
  height: number;
}

/** Build a frame (matrices + viewport) from the camera and canvas size. */
export function makeFrame(camera: CameraState, viewProj: Mat4, width: number, height: number): Frame {
  return { viewProj, eye: cameraEye(camera), width, height };
}

interface Projected {
  x: number;
  y: number;
  w: number; // clip-space w ~ distance, used for sorting
  visible: boolean;
}

function project(frame: Frame, p: Vec3): Projected {
  const clip = mulVec4(frame.viewProj, { x: p.x, y: p.y, z: p.z, w: 1 });
  const visible = clip.w > 1e-4;
  const iw = visible ? 1 / clip.w : 0;
  const ndcX = clip.x * iw;
  const ndcY = clip.y * iw;
  return {
    x: (ndcX * 0.5 + 0.5) * frame.width,
    y: (1 - (ndcY * 0.5 + 0.5)) * frame.height,
    w: clip.w,
    visible,
  };
}

const shadeFactor = (style: RenderStyle, normal: Vec3): number => {
  if (!style.shading) return 1;
  const ndl = Math.max(0, dot3(normal, style.light));
  return Math.min(1, style.ambient + (1 - style.ambient) * ndl);
};

const rgb = (r: number, g: number, b: number): string =>
  `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;

const readVec = (arr: Float32Array, i: number): Vec3 =>
  v3(arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]);

/* ------------------------------------------------------------------ */
/* Point cloud                                                         */
/* ------------------------------------------------------------------ */

export function drawPointCloud(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  cloud: PointCloud,
  style: RenderStyle,
): void {
  interface P { x: number; y: number; w: number; r: number; g: number; b: number; s: number }
  const buf: P[] = [];

  for (let i = 0; i < cloud.count; i++) {
    const pr = project(frame, readVec(cloud.positions, i));
    if (!pr.visible) continue;
    const n = readVec(cloud.normals, i);
    let sh = shadeFactor(style, n);
    if (style.depthFade) {
      // fade points beyond ~6 units
      sh *= Math.max(0.15, 1 - pr.w / 8);
    }
    const c = readVec(cloud.colors, i);
    buf.push({ x: pr.x, y: pr.y, w: pr.w, r: c.x * sh, g: c.y * sh, b: c.z * sh, s: style.pointSize });
  }

  // painter's algorithm: far points first
  buf.sort((a, b) => b.w - a.w);

  const half = style.pointSize / 2;
  for (const p of buf) {
    ctx.fillStyle = rgb(p.r, p.g, p.b);
    ctx.fillRect(p.x - half, p.y - half, p.s, p.s);
  }
}

/* ------------------------------------------------------------------ */
/* Triangle mesh (flat shaded, back-face culled)                       */
/* ------------------------------------------------------------------ */

export function drawMesh(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  mesh: Mesh,
  style: RenderStyle,
): void {
  interface Tri { a: Projected; b: Projected; c: Projected; depth: number; color: string }
  const tris: Tri[] = [];
  const idx = mesh.indices;

  for (let i = 0; i < idx.length; i += 3) {
    const i0 = idx[i];
    const i1 = idx[i + 1];
    const i2 = idx[i + 2];
    const p0 = project(frame, readVec(mesh.positions, i0));
    const p1 = project(frame, readVec(mesh.positions, i1));
    const p2 = project(frame, readVec(mesh.positions, i2));
    if (!p0.visible || !p1.visible || !p2.visible) continue;

    // screen-space signed area -> back-face culling
    const area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    if (area <= 0) continue;

    // flat normal = average of the three vertex normals
    const n0 = readVec(mesh.normals, i0);
    const n1 = readVec(mesh.normals, i1);
    const n2 = readVec(mesh.normals, i2);
    const n = normalize3(v3(n0.x + n1.x + n2.x, n0.y + n1.y + n2.y, n0.z + n1.z + n2.z));
    const sh = shadeFactor(style, n);

    const c0 = readVec(mesh.colors, i0);
    const c1 = readVec(mesh.colors, i1);
    const c2 = readVec(mesh.colors, i2);
    const cr = ((c0.x + c1.x + c2.x) / 3) * sh;
    const cg = ((c0.y + c1.y + c2.y) / 3) * sh;
    const cb = ((c0.z + c1.z + c2.z) / 3) * sh;

    tris.push({ a: p0, b: p1, c: p2, depth: (p0.w + p1.w + p2.w) / 3, color: rgb(cr, cg, cb) });
  }

  tris.sort((a, b) => b.depth - a.depth);
  for (const t of tris) {
    ctx.fillStyle = t.color;
    ctx.strokeStyle = t.color;
    ctx.beginPath();
    ctx.moveTo(t.a.x, t.a.y);
    ctx.lineTo(t.b.x, t.b.y);
    ctx.lineTo(t.c.x, t.c.y);
    ctx.closePath();
    ctx.fill();
  }
}

/* ------------------------------------------------------------------ */
/* Wireframe                                                           */
/* ------------------------------------------------------------------ */

export function drawWireframe(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  mesh: Mesh,
  style: RenderStyle,
): void {
  const idx = mesh.indices;
  ctx.strokeStyle = 'rgba(120, 220, 255, 0.6)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < idx.length; i += 3) {
    const i0 = idx[i];
    const i1 = idx[i + 1];
    const i2 = idx[i + 2];
    const p0 = project(frame, readVec(mesh.positions, i0));
    const p1 = project(frame, readVec(mesh.positions, i1));
    const p2 = project(frame, readVec(mesh.positions, i2));
    if (!p0.visible || !p1.visible || !p2.visible) continue;
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.closePath();
  }
  ctx.stroke();
  void style;
}

/* ------------------------------------------------------------------ */
/* Frame orchestration                                                 */
/* ------------------------------------------------------------------ */

export function renderScene(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  geometry: SceneGeometry,
  style: RenderStyle,
): void {
  ctx.fillStyle = style.background;
  ctx.fillRect(0, 0, frame.width, frame.height);

  if (geometry.kind === 'points') drawPointCloud(ctx, frame, geometry.cloud, style);
  else if (geometry.kind === 'mesh') drawMesh(ctx, frame, geometry.mesh, style);
  else drawWireframe(ctx, frame, geometry.mesh, style);
}

/** Draw a ground grid for spatial context. */
export function drawGrid(ctx: CanvasRenderingContext2D, frame: Frame, size = 3, step = 0.5): void {
  ctx.strokeStyle = 'rgba(90, 120, 160, 0.35)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = -size; i <= size + 1e-9; i += step) {
    const a = project(frame, v3(i, -0.6, -size));
    const b = project(frame, v3(i, -0.6, size));
    const c = project(frame, v3(-size, -0.6, i));
    const d = project(frame, v3(size, -0.6, i));
    if (a.visible && b.visible) { ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); }
    if (c.visible && d.visible) { ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); }
  }
  ctx.stroke();
}

