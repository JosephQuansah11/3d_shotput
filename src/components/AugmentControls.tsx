import { AugmentOptions, DEFAULT_AUGMENT, NO_AUGMENT, isAugmentEnabled } from '../data/augment';

export interface AugmentControlsProps {
  value: AugmentOptions;
  onChange: (value: AugmentOptions) => void;
  disabled?: boolean;
}

/**
 * AugmentControls — a compact editor for the augmentation policy.
 *
 * A master switch turns the whole policy on/off; the sliders tune the strength
 * of the transformations that preserve the image ↔ depth correspondence.
 */
export function AugmentControls({ value, onChange, disabled }: AugmentControlsProps) {
  const enabled = isAugmentEnabled(value);
  const set = (patch: Partial<AugmentOptions>) => onChange({ ...value, ...patch });

  return (
    <div className="augment">
      <label className="check">
        <input
          type="checkbox"
          checked={enabled}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked ? { ...DEFAULT_AUGMENT } : { ...NO_AUGMENT })}
        />
        <strong>Data augmentation</strong>
      </label>

      {enabled && (
        <div className="grid2">
          <label>
            Flip probability {value.hflip.toFixed(2)}
            <input type="range" min={0} max={1} step={0.05} value={value.hflip} disabled={disabled}
              onChange={(e) => set({ hflip: +e.target.value })} />
          </label>
          <label>
            Brightness {value.brightness.toFixed(2)}
            <input type="range" min={0} max={0.5} step={0.01} value={value.brightness} disabled={disabled}
              onChange={(e) => set({ brightness: +e.target.value })} />
          </label>
          <label>
            Contrast {value.contrast.toFixed(2)}
            <input type="range" min={0} max={0.6} step={0.01} value={value.contrast} disabled={disabled}
              onChange={(e) => set({ contrast: +e.target.value })} />
          </label>
          <label>
            Noise {value.noise.toFixed(3)}
            <input type="range" min={0} max={0.15} step={0.005} value={value.noise} disabled={disabled}
              onChange={(e) => set({ noise: +e.target.value })} />
          </label>
          <label>
            Translate {value.translate.toFixed(2)}
            <input type="range" min={0} max={0.2} step={0.01} value={value.translate} disabled={disabled}
              onChange={(e) => set({ translate: +e.target.value })} />
          </label>
          <label>
            Rotate {value.rotate.toFixed(2)}
            <input type="range" min={0} max={0.4} step={0.01} value={value.rotate} disabled={disabled}
              onChange={(e) => set({ rotate: +e.target.value })} />
          </label>
          <label>
            Cutout {value.cutout.toFixed(2)}
            <input type="range" min={0} max={0.6} step={0.05} value={value.cutout} disabled={disabled}
              onChange={(e) => set({ cutout: +e.target.value })} />
          </label>
        </div>
      )}
      <p className="hint">
        Flip and geometric jitter warp the image, depth and mask together; brightness,
        contrast and noise touch the image only. Applied fresh every epoch.
      </p>
    </div>
  );
}
