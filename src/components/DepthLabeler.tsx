import { useEffect, useMemo, useRef, useState } from 'react';
import { GrayImage } from '../cv/image';

export interface DepthLabelerProps {
  image: GrayImage;
  onSave: (depth: Float32Array, mask: Float32Array) => void;
  onCancel: () => void;
}

type Brush = 'near' | 'far' | 'erase';

const DISPLAY = 384;

/** Colour ramp for the depth overlay (blue = far, red = near). */
function ramp(d: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, d));
  return [
    Math.max(0, Math.min(1, 1.5 * t - 0.4)) * 255,
    Math.max(0, Math.min(1, 1 - Math.abs(2 * t - 1))) * 255,
    Math.max(0, Math.min(1, 1 - 2 * t)) * 255,
  ];
}

/**
 * DepthLabeler — paint sparse depth supervision directly onto a photo.
 *
 * Red strokes mark "near", blue strokes mark "far", and the eraser removes
 * labels. Only painted pixels are supervised; the trainer fills the rest using
 * the smoothness prior. This is how you get real-world depth labels without a
 * depth sensor.
 */
export function DepthLabeler({ image, onSave, onCancel }: DepthLabelerProps) {
  const size = image.width;
  const depthRef = useRef<Float32Array>(new Float32Array(size * size));
  const maskRef = useRef<Float32Array>(new Float32Array(size * size));
  const drawing = useRef(false);
  const displayRef = useRef<HTMLCanvasElement>(null);
  const offscreen = useMemo(() => {
    const c = document.createElement('canvas');
    c.width = size;
    c.height = size;
    return c;
  }, [size]);

  const [brush, setBrush] = useState<Brush>('near');
  const [radius, setRadius] = useState(3);
  const [version, setVersion] = useState(0);

  const paint = (cx: number, cy: number) => {
    const depth = depthRef.current;
    const mask = maskRef.current;
    const r = radius;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy > r * r) continue;
        const x = cx + dx;
        const y = cy + dy;
        if (x < 0 || y < 0 || x >= size || y >= size) continue;
        const i = y * size + x;
        if (brush === 'erase') {
          mask[i] = 0;
          depth[i] = 0;
        } else {
          mask[i] = 1;
          depth[i] = brush === 'near' ? 0.95 : 0.05;
        }
      }
    }
    setVersion((v) => v + 1);
  };

  // Composite the photo + overlay into the visible canvas whenever it changes.
  useEffect(() => {
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(size, size);
    const depth = depthRef.current;
    const mask = maskRef.current;
    for (let i = 0; i < size * size; i++) {
      const g = Math.round(255 * Math.max(0, Math.min(1, image.data[i])));
      let r = g;
      let gg = g;
      let b = g;
      if (mask[i] > 0) {
        const [rr, gg2, bb] = ramp(depth[i]);
        r = Math.round(0.35 * g + 0.65 * rr);
        gg = Math.round(0.35 * g + 0.65 * gg2);
        b = Math.round(0.35 * g + 0.65 * bb);
      }
      const p = i * 4;
      img.data[p] = r;
      img.data[p + 1] = gg;
      img.data[p + 2] = b;
      img.data[p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);

    const display = displayRef.current;
    if (!display) return;
    const dctx = display.getContext('2d');
    if (!dctx) return;
    dctx.imageSmoothingEnabled = false;
    dctx.clearRect(0, 0, DISPLAY, DISPLAY);
    dctx.drawImage(offscreen, 0, 0, size, size, 0, 0, DISPLAY, DISPLAY);
  }, [image, offscreen, size, version]);

  const toData = (e: React.PointerEvent): { x: number; y: number } => {
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    return {
      x: Math.floor(((e.clientX - rect.left) / rect.width) * size),
      y: Math.floor(((e.clientY - rect.top) / rect.height) * size),
    };
  };

  const labelled = useMemo(() => {
    let n = 0;
    for (let i = 0; i < maskRef.current.length; i++) if (maskRef.current[i] > 0) n++;
    return Math.round((n / maskRef.current.length) * 100);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version]);

  return (
    <div className="labeler">
      <canvas
        ref={displayRef}
        width={DISPLAY}
        height={DISPLAY}
        className="labeler-canvas"
        onPointerDown={(e) => {
          drawing.current = true;
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          const d = toData(e);
          paint(d.x, d.y);
        }}
        onPointerMove={(e) => {
          if (!drawing.current) return;
          const d = toData(e);
          paint(d.x, d.y);
        }}
        onPointerUp={() => {
          drawing.current = false;
        }}
      />

      <div className="labeler-controls">
        <div className="segmented">
          <button className={brush === 'near' ? 'active' : ''} onClick={() => setBrush('near')}>Near</button>
          <button className={brush === 'far' ? 'active' : ''} onClick={() => setBrush('far')}>Far</button>
          <button className={brush === 'erase' ? 'active' : ''} onClick={() => setBrush('erase')}>Erase</button>
        </div>
        <label>
          Brush {radius}px
          <input type="range" min={1} max={10} value={radius} onChange={(e) => setRadius(+e.target.value)} />
        </label>
        <span className="hint">{labelled}% labelled</span>
        <button className="primary" onClick={() => onSave(depthRef.current.slice(), maskRef.current.slice())}>
          Save label
        </button>
        <button onClick={onCancel}>Cancel</button>
      </div>
      <p className="hint">
        Paint red where objects are <strong>near</strong> and blue where they are <strong>far</strong>.
        The model learns from the painted pixels and smooths everything in between.
      </p>
    </div>
  );
}
