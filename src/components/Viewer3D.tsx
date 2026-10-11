import { useEffect, useRef } from 'react';
import { CameraState, defaultCamera, viewProjection } from '../math/camera';
import { SceneGeometry, RenderStyle, renderScene, drawGrid, makeFrame } from '../viewer/renderer';
import { bounds } from '../cv/geometry';

export interface Viewer3DProps {
  geometry: SceneGeometry | null;
  style: RenderStyle;
  showGrid: boolean;
  /** incrementing this value re-frames the camera on the geometry */
  frameKey: number;
  autoRotate: boolean;
}

/**
 * Viewer3D — a canvas that renders our geometry with the software renderer and
 * lets the user orbit with the mouse. All the camera maths is ours.
 */
export function Viewer3D({ geometry, style, showGrid, frameKey, autoRotate }: Viewer3DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<CameraState>(defaultCamera());
  const geometryRef = useRef<SceneGeometry | null>(geometry);
  const styleRef = useRef(style);
  const gridRef = useRef(showGrid);
  const autoRef = useRef(autoRotate);
  const dragRef = useRef<{ x: number; y: number; button: number } | null>(null);

  geometryRef.current = geometry;
  styleRef.current = style;
  gridRef.current = showGrid;
  autoRef.current = autoRotate;

  // Re-frame the camera whenever the caller bumps frameKey.
  useEffect(() => {
    const g = geometryRef.current;
    if (!g) return;
    const positions = g.kind === 'points' ? g.cloud.positions : g.mesh.positions;
    const b = bounds(positions);
    const cam = cameraRef.current;
    cam.target = b.center;
    cam.distance = b.radius * 2.6;
    cam.near = Math.max(0.01, b.radius * 0.02);
    cam.far = b.radius * 20;
  }, [frameKey, geometry]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const loop = () => {
      const cam = cameraRef.current;
      if (autoRef.current && !dragRef.current) cam.azimuth += 0.005;

      const w = canvas.width;
      const h = canvas.height;
      const aspect = w / h;
      const vp = viewProjection(cam, aspect);
      const frame = makeFrame(cam, vp, w, h);

      const g = geometryRef.current;
      if (g) {
        if (gridRef.current) drawGrid(ctx, frame);
        renderScene(ctx, frame, g, styleRef.current);
      } else {
        ctx.fillStyle = styleRef.current.background;
        ctx.fillRect(0, 0, w, h);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    dragRef.current = { x: e.clientX, y: e.clientY, button: e.button };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    const cam = cameraRef.current;

    if (drag.button === 2) {
      // pan: move target along the camera's right/up in the horizontal plane
      const scale = cam.distance * 0.0015;
      cam.target = {
        x: cam.target.x - Math.cos(cam.azimuth) * dx * scale,
        y: cam.target.y + dy * scale,
        z: cam.target.z + Math.sin(cam.azimuth) * dx * scale,
      };
    } else {
      cam.azimuth -= dx * 0.01;
      cam.elevation = Math.max(-1.4, Math.min(1.4, cam.elevation + dy * 0.01));
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    dragRef.current = null;
    (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
  };

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const cam = cameraRef.current;
    cam.distance = Math.max(0.05, cam.distance * (1 + Math.sign(e.deltaY) * 0.1));
  };

  return (
    <canvas
      ref={canvasRef}
      className="viewer-canvas"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onWheel={onWheel}
      onContextMenu={(e) => e.preventDefault()}
    />
  );
}
