import { describe, it, expect } from 'vitest';
import { randomScene, trace, shade } from '../data/scene';
import { DEFAULT_RENDER, randomSample, renderScene } from '../data/dataset';
import { depthToPointCloud, depthToMesh, bounds } from './geometry';
import { sobel, gaussianBlur, normalizeToUnit } from './filters';
import { GrayImage } from './image';

describe('ray tracer', () => {
  it('renders an image and depth buffer of the right size', () => {
    const s = renderScene(randomScene(), { ...DEFAULT_RENDER, size: 16 });
    expect(s.image.length).toBe(256);
    expect(s.depth.length).toBe(256);
    for (const d of s.depth) {
      expect(d).toBeGreaterThanOrEqual(0);
      expect(d).toBeLessThanOrEqual(1);
    }
  });

  it('hits a sphere placed on the view axis', () => {
    const scene = {
      primitives: [{ kind: 'sphere' as const, center: { x: 0, y: 0, z: -2 }, radius: 0.5, material: { albedo: 1 } }],
      light: { direction: { x: 0, y: 1, z: 0 }, intensity: 1 },
      ambient: 0.2,
      background: 0,
    };
    const hit = trace(scene, { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: -1 });
    expect(hit).not.toBeNull();
    expect(hit!.t).toBeCloseTo(1.5, 5);
    expect(shade(scene, hit!)).toBeGreaterThan(0);
  });
});

describe('geometry reconstruction', () => {
  it('back-projects a flat depth map to a consistent point count', () => {
    const size = 8;
    const depth = new Float32Array(size * size).fill(0.5);
    const cloud = depthToPointCloud(depth, size, size, {
      near: 0.5, far: 4, fovY: Math.PI / 3, stride: 1,
    });
    expect(cloud.count).toBe(size * size);
    expect(cloud.positions.length).toBe(size * size * 3);
  });

  it('builds a mesh with valid indices', () => {
    const size = 8;
    const depth = new Float32Array(size * size).fill(0.5);
    const mesh = depthToMesh(depth, size, size, { near: 0.5, far: 4, fovY: Math.PI / 3, stride: 1 });
    expect(mesh.indices.length % 3).toBe(0);
    const maxIdx = Math.max(...mesh.indices);
    expect(maxIdx).toBeLessThan(mesh.positions.length / 3);
  });

  it('computes sane bounds', () => {
    const positions = new Float32Array([-1, -1, -1, 1, 1, 1]);
    const b = bounds(positions);
    expect(b.center.x).toBeCloseTo(0);
    expect(b.radius).toBeCloseTo(Math.sqrt(3), 4);
  });
});

describe('cv filters', () => {
  it('produces unit-range output from normalizeToUnit', () => {
    const buf = new Float32Array([-3, 0, 5]);
    const n = normalizeToUnit(buf);
    expect(n[0]).toBeCloseTo(0);
    expect(n[2]).toBeCloseTo(1);
  });

  it('detects a vertical edge with Sobel', () => {
    const size = 8;
    const img: GrayImage = { data: new Float32Array(size * size), width: size, height: size };
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) img.data[y * size + x] = x < size / 2 ? 0 : 1;
    }
    const g = sobel(img);
    const center = g.magnitude[4 * size + 4];
    expect(center).toBeGreaterThan(0);
  });

  it('blurs without changing dimensions', () => {
    const s = randomSample({ ...DEFAULT_RENDER, size: 8 });
    const img: GrayImage = { data: s.image, width: 8, height: 8 };
    const blurred = gaussianBlur(img);
    expect(blurred.width).toBe(8);
    expect(blurred.data.length).toBe(64);
  });
});
