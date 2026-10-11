/**
 * cv/geometry.ts — turning a depth map into 3D geometry.
 *
 * This is where computer vision meets the math layer: a per-pixel depth is
 * back-projected through a pinhole camera model to produce a point cloud, then
 * neighbouring points are stitched into triangles and their normals computed
 * from cross products.
 */

import { Vec3, v3, cross3, normalize3, sub3, add3, scale3 } from '../math/vec';

export interface PointCloud {
  positions: Float32Array; // xyz triples
  colors: Float32Array; // rgb triples in [0,1]
  normals: Float32Array; // xyz triples
  count: number;
}

export interface Mesh {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  indices: Uint32Array;
}

export interface DepthToCloudOptions {
  /** depth 0 -> far, depth 1 -> near, in world units */
  near: number;
  far: number;
  /** vertical field of view in radians */
  fovY: number;
  /** stride over pixels (2 = every other pixel, faster & lighter) */
  stride: number;
  /** optional luminance buffer for vertex colouring */
  image?: Float32Array;
}

/**
 * Back-project a depth map (values in [0,1], 1 = nearest) into 3D.
 * Uses the standard pinhole equations:
 *   Z = near + (1 - d) * (far - near)
 *   X = (u - cx) / f * Z
 *   Y = (cy - v) / f * Z
 */
export function depthToPointCloud(
  depth: Float32Array,
  width: number,
  height: number,
  opts: DepthToCloudOptions,
): PointCloud {
  const { near, far, fovY, stride, image } = opts;
  const f = 1 / Math.tan(fovY / 2); // focal length (normalised pixels)
  const cx = width / 2;
  const cy = height / 2;

  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const cols: number[] = [];

  for (let y = 0; y < height; y += stride) {
    for (let x = 0; x < width; x += stride) {
      const d = depth[y * width + x];
      const z = near + (1 - d) * (far - near);
      const px = ((x - cx) / (0.5 * width) / f) * z;
      const py = ((cy - y) / (0.5 * width) / f) * z;
      xs.push(px);
      ys.push(py);
      zs.push(-z);
      const lum = image ? image[y * width + x] : d;
      cols.push(lum, lum, lum);
    }
  }

  const count = xs.length;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[i * 3] = xs[i];
    positions[i * 3 + 1] = ys[i];
    positions[i * 3 + 2] = zs[i];
    colors[i * 3] = cols[i * 3];
    colors[i * 3 + 1] = cols[i * 3 + 1];
    colors[i * 3 + 2] = cols[i * 3 + 2];
  }

  const normals = estimatePointNormals(positions, count);
  return { positions, colors, normals, count };
}

/** Estimate normals by chaining neighbouring points along the raster order. */
function estimatePointNormals(positions: Float32Array, count: number): Float32Array {
  const normals = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) normals[i * 3 + 2] = -1; // default +Z
  for (let i = 1; i < count; i++) {
    const a = readVec(positions, i - 1);
    const b = readVec(positions, i);
    const n = normalize3(cross3(sub3(b, a), v3(0, 0, 1)));
    if (Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z) > 0.1) {
      normals[i * 3] = n.x;
      normals[i * 3 + 1] = n.y;
      normals[i * 3 + 2] = n.z;
    }
  }
  return normals;
}

const readVec = (arr: Float32Array, i: number): Vec3 =>
  v3(arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]);

/**
 * Build a triangle mesh from a dense depth grid. Quads whose depth jumps
 * sharply (silhouette edges) are skipped so geometry doesn't smear across
 * object boundaries.
 */
export function depthToMesh(
  depth: Float32Array,
  width: number,
  height: number,
  opts: DepthToCloudOptions & { discontinuity?: number },
): Mesh {
  const { near, far, fovY, image } = opts;
  const discontinuity = opts.discontinuity ?? 0.08;
  const f = 1 / Math.tan(fovY / 2);
  const cx = width / 2;
  const cy = height / 2;

  const project = (x: number, y: number): Vec3 => {
    const d = depth[y * width + x];
    const z = near + (1 - d) * (far - near);
    const px = ((x - cx) / (0.5 * width) / f) * z;
    const py = ((cy - y) / (0.5 * width) / f) * z;
    return v3(px, py, -z);
  };

  const positions: number[] = [];
  const normals: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];

  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const d00 = depth[y * width + x];
      const d10 = depth[y * width + x + 1];
      const d01 = depth[(y + 1) * width + x];
      const d11 = depth[(y + 1) * width + x + 1];
      if (Math.max(d00, d10, d01, d11) - Math.min(d00, d10, d01, d11) > discontinuity) {
        continue; // silhouette
      }

      const p00 = project(x, y);
      const p10 = project(x + 1, y);
      const p01 = project(x, y + 1);
      const p11 = project(x + 1, y + 1);

      const n = normalize3(cross3(sub3(p10, p00), sub3(p01, p00)));
      const nn = Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z) > 0.1 ? n : v3(0, 0, -1);

      const base = positions.length / 3;
      const quad: Array<[Vec3, number]> = [
        [p00, y * width + x],
        [p10, y * width + x + 1],
        [p01, (y + 1) * width + x],
        [p11, (y + 1) * width + x + 1],
      ];
      for (const [p, idx] of quad) {
        positions.push(p.x, p.y, p.z);
        normals.push(nn.x, nn.y, nn.z);
        const lum = image ? image[idx] : 0.7;
        colors.push(lum, lum, lum);
      }

      indices.push(base, base + 1, base + 2);
      indices.push(base + 1, base + 3, base + 2);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    colors: new Float32Array(colors),
    indices: new Uint32Array(indices),
  };
}

/** Bounding box, centroid and radius of a point cloud, used to frame the camera. */
export function bounds(positions: Float32Array): { min: Vec3; max: Vec3; center: Vec3; radius: number } {
  const min = v3(Infinity, Infinity, Infinity);
  const max = v3(-Infinity, -Infinity, -Infinity);
  for (let i = 0; i < positions.length; i += 3) {
    min.x = Math.min(min.x, positions[i]);
    min.y = Math.min(min.y, positions[i + 1]);
    min.z = Math.min(min.z, positions[i + 2]);
    max.x = Math.max(max.x, positions[i]);
    max.y = Math.max(max.y, positions[i + 1]);
    max.z = Math.max(max.z, positions[i + 2]);
  }
  const center = scale3(add3(min, max), 0.5);
  let radius = 0;
  for (let i = 0; i < positions.length; i += 3) {
    const p = v3(positions[i], positions[i + 1], positions[i + 2]);
    radius = Math.max(radius, Math.hypot(p.x - center.x, p.y - center.y, p.z - center.z));
  }
  return { min, max, center, radius: radius || 1 };
}

