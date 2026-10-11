import { useEffect, useMemo, useRef, useState } from 'react';
import { Sample } from '../data/dataset';
import { randomSample, DEFAULT_RENDER } from '../data/dataset';
import { AugmentOptions, augmentSample, mulberry32 } from '../data/augment';

export interface AugmentationPreviewProps {
  options: AugmentOptions;
  size: number;
  /** optional real sample to preview; falls back to a fresh synthetic scene */
  source?: Sample | null;
  count?: number;
}

/** Depth colour ramp (blue = far, red = near). */
function ramp(d: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, d));
  return [
    Math.max(0, Math.min(1, 1.5 * t - 0.4)) * 255,
    Math.max(0, Math.min(1, 1 - Math.abs(2 * t - 1))) * 255,
    Math.max(0, Math.min(1, 1 - 2 * t)) * 255,
  ];
}

/** Each tile shows the augmented image (left) beside its depth target (right). */
function drawSample(canvas: HTMLCanvasElement, sample: Sample): void {
  const { size, image, depth, mask } = sample;
  const w = size * 2;
  canvas.width = w;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const img = ctx.createImageData(w, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      if (x < size) {
        const v = Math.round(255 * Math.max(0, Math.min(1, image[y * size + x])));
        img.data[p] = v;
        img.data[p + 1] = v;
        img.data[p + 2] = v;
      } else {
        const i = y * size + (x - size);
        const [r, g, b] = ramp(depth[i]);
        // dim pixels that are not supervised
        const k = mask && mask[i] <= 0 ? 0.25 : 1;
        img.data[p] = r * k;
        img.data[p + 1] = g * k;
        img.data[p + 2] = b * k;
      }
      img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * AugmentationPreview — a live grid of augmented samples.
 *
 * The base scene is held fixed while you drag the sliders so the effect of each
 * transformation is directly comparable; press *Shuffle* to draw a new base.
 * Left half of a tile is the augmented input, right half the depth target
 * (unlabelled pixels are dimmed).
 */
export function AugmentationPreview({ options, size, source, count = 6 }: AugmentationPreviewProps) {
  const [nonce, setNonce] = useState(0);
  const canvases = useRef<Array<HTMLCanvasElement | null>>([]);

  const base = useMemo<Sample>(
    () => source ?? randomSample({ ...DEFAULT_RENDER, size }),
    [source, size, nonce],
  );

  const variants = useMemo(() => {
    const rng = mulberry32(1000 + nonce);
    const out: Sample[] = [];
    for (let i = 0; i < count; i++) out.push(augmentSample(base, options, rng));
    return out;
  }, [base, options, count, nonce]);

  useEffect(() => {
    variants.forEach((s, i) => {
      const c = canvases.current[i];
      if (c) drawSample(c, s);
    });
  }, [variants]);

  return (
    <div className="augment-preview">
      <div className="augment-preview-head">
        <span className="hint">input · depth target (dimmed = unlabelled)</span>
        <button onClick={() => setNonce((n) => n + 1)}>Shuffle</button>
      </div>
      <div className="augment-preview-grid">
        {variants.map((_, i) => (
          <canvas
            key={i}
            ref={(el) => {
              canvases.current[i] = el;
            }}
          />
        ))}
      </div>
    </div>
  );
}
