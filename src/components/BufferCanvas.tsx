import { useEffect, useRef } from 'react';

export interface BufferCanvasProps {
  data: Float32Array | null;
  size: number;
  colormap: 'gray' | 'depth';
  label: string;
}

/** Map a scalar in [0,1] to an RGB triple (blue -> cyan -> green -> red). */
function ramp(d: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, d));
  const r = Math.max(0, Math.min(1, 1.5 * t - 0.4));
  const g = Math.max(0, Math.min(1, 1 - Math.abs(2 * t - 1)));
  const b = Math.max(0, Math.min(1, 1 - 2 * t));
  return [r, g, b];
}

/** Draws a square float buffer to a canvas, optionally with a depth ramp. */
export function BufferCanvas({ data, size, colormap, label }: BufferCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!data) {
      ctx.clearRect(0, 0, size, size);
      return;
    }

    const img = ctx.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const v = data[i] ?? 0;
      let r: number;
      let g: number;
      let b: number;
      if (colormap === 'gray') {
        r = g = b = Math.max(0, Math.min(1, v));
      } else {
        [r, g, b] = ramp(v);
      }
      const p = i * 4;
      img.data[p] = r * 255;
      img.data[p + 1] = g * 255;
      img.data[p + 2] = b * 255;
      img.data[p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, size, colormap]);

  return (
    <figure className="buffer-canvas">
      <canvas ref={ref} width={size} height={size} />
      <figcaption>{label}</figcaption>
    </figure>
  );
}
