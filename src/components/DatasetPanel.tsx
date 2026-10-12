import { useRef, useState } from 'react';
import { DepthLabeler } from './DepthLabeler';
import { GrayImage } from '../cv/image';
import { ingestImageFile, ingestDepthFile, denseMask } from '../data/ingest';
import {
  RealSample, newId, makeThumbnail, labelledFraction, toTrainingSample, STORAGE_SIZE,
} from '../data/realDataset';
import { TrainProgress } from '../ai/trainer';
import { AugmentControls } from './AugmentControls';
import { AugmentationPreview } from './AugmentationPreview';
import { AugmentOptions, DEFAULT_AUGMENT, isAugmentEnabled } from '../data/augment';

export interface FinetuneOptions {
  epochs: number;
  learningRate: number;
  includeSynthetic: boolean;
  lambda: number;
  augment: AugmentOptions;
}

export interface DatasetPanelProps {
  samples: RealSample[];
  onAdd: (sample: RealSample) => void;
  onDelete: (id: string) => void;
  onClear: () => void;
  onFinetune: (opts: FinetuneOptions) => void;
  training: boolean;
  hasModel: boolean;
  progress: TrainProgress | null;
  modelSize: number;
}

type Pending = { gray: GrayImage; name: string };

/**
 * DatasetPanel — the real-world data flow.
 *
 * Add a photo, then attach depth supervision either by uploading a depth-map
 * image or by painting strokes in the labeler. Saved samples persist in
 * IndexedDB and can fine-tune the model via transfer learning.
 */
export function DatasetPanel(props: DatasetPanelProps) {
  const { samples, onAdd, onDelete, onClear, onFinetune, training, hasModel, progress, modelSize } = props;

  const photoInput = useRef<HTMLInputElement>(null);
  const depthInput = useRef<HTMLInputElement>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [labeling, setLabeling] = useState(false);
  const [invert, setInvert] = useState(false);
  const [busy, setBusy] = useState(false);
  const [opts, setOpts] = useState<FinetuneOptions>({
    epochs: 12, learningRate: 0.005, includeSynthetic: true, lambda: 0.05,
    augment: { ...DEFAULT_AUGMENT },
  });

  const pickPhoto = async (file: File) => {
    setBusy(true);
    try {
      const gray = await ingestImageFile(file);
      setPending({ gray, name: file.name });
    } finally {
      setBusy(false);
    }
  };

  const saveWithDepthMap = async (file: File) => {
    if (!pending) return;
    setBusy(true);
    try {
      const depth = await ingestDepthFile(file, invert);
      onAdd({
        id: newId(),
        name: pending.name,
        createdAt: Date.now(),
        size: STORAGE_SIZE,
        image: pending.gray.data,
        depth,
        mask: denseMask(STORAGE_SIZE),
        thumbnail: makeThumbnail(pending.gray.data, STORAGE_SIZE),
        source: 'depth-map',
      });
      setPending(null);
    } finally {
      setBusy(false);
    }
  };

  const saveWithBrush = (depth: Float32Array, mask: Float32Array) => {
    if (!pending) return;
    onAdd({
      id: newId(),
      name: pending.name,
      createdAt: Date.now(),
      size: STORAGE_SIZE,
      image: pending.gray.data,
      depth,
      mask,
      thumbnail: makeThumbnail(pending.gray.data, STORAGE_SIZE),
      source: 'brush',
    });
    setPending(null);
    setLabeling(false);
  };

  if (labeling && pending) {
    return (
      <section className="panel">
        <h2>Label depth — {pending.name}</h2>
        <DepthLabeler image={pending.gray} onSave={saveWithBrush} onCancel={() => setLabeling(false)} />
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>Real-world data</h2>
      <p className="hint">
        Teach the model from your own photos. Attach a depth map or paint rough
        near/far strokes — sparse labels are enough thanks to the smoothness prior.
      </p>

      <input ref={photoInput} type="file" accept="image/*" hidden
        onChange={(e) => { const f = e.target.files?.[0]; if (f) pickPhoto(f); e.target.value = ''; }} />
      <input ref={depthInput} type="file" accept="image/*" hidden
        onChange={(e) => { const f = e.target.files?.[0]; if (f) saveWithDepthMap(f); e.target.value = ''; }} />

      {!pending ? (
        <div className="row">
          <button className="primary" disabled={busy || training} onClick={() => photoInput.current?.click()}>
            Add photo
          </button>
          <button disabled={samples.length === 0 || training} onClick={onClear}>Clear dataset</button>
        </div>
      ) : (
        <div className="pending">
          <p className="hint">Photo: <strong>{pending.name}</strong> — choose how to label it.</p>
          <div className="row">
            <button className="primary" onClick={() => setLabeling(true)}>Paint labels</button>
            <button disabled={busy} onClick={() => depthInput.current?.click()}>Upload depth map</button>
            <button onClick={() => setPending(null)}>Cancel</button>
          </div>
          <label className="check">
            <input type="checkbox" checked={invert} onChange={(e) => setInvert(e.target.checked)} />
            invert depth map (white = far)
          </label>
        </div>
      )}

      {samples.length > 0 && (
        <>
          <div className="sample-grid">
            {samples.map((s) => (
              <figure key={s.id} className="sample-card" title={s.name}>
                {s.thumbnail ? <img src={s.thumbnail} alt={s.name} /> : <div className="thumb-fallback" />}
                <figcaption>
                  <span>{Math.round(labelledFraction(s) * 100)}%</span>
                  <button onClick={() => onDelete(s.id)} disabled={training}>×</button>
                </figcaption>
              </figure>
            ))}
          </div>

          <div className="finetune">
            <h3>Fine-tune on real data</h3>
            <div className="grid2">
              <label>
                Epochs
                <input type="number" min={1} max={200} value={opts.epochs} disabled={training}
                  onChange={(e) => setOpts({ ...opts, epochs: +e.target.value })} />
              </label>
              <label>
                Learning rate
                <input type="number" step={0.001} min={0.0001} max={1} value={opts.learningRate} disabled={training}
                  onChange={(e) => setOpts({ ...opts, learningRate: +e.target.value })} />
              </label>
              <label>
                Smoothness λ
                <input type="number" step={0.01} min={0} max={1} value={opts.lambda} disabled={training}
                  onChange={(e) => setOpts({ ...opts, lambda: +e.target.value })} />
              </label>
              <label className="check">
                <input type="checkbox" checked={opts.includeSynthetic} disabled={training}
                  onChange={(e) => setOpts({ ...opts, includeSynthetic: e.target.checked })} />
                mix synthetic scenes
              </label>
            </div>
            <AugmentControls
              value={opts.augment}
              onChange={(a) => setOpts({ ...opts, augment: a })}
              disabled={training}
            />
            {isAugmentEnabled(opts.augment) && (
              <AugmentationPreview
                options={opts.augment}
                size={modelSize}
                source={samples[0] ? toTrainingSample(samples[0], modelSize) : null}
              />
            )}
            <div className="row">
              <button className="primary" disabled={training} onClick={() => onFinetune(opts)}>
                {training ? 'Training…' : 'Fine-tune model'}
              </button>
            </div>
            {!hasModel && <p className="hint">Tip: train the base model first, then fine-tune here.</p>}
            {progress && (
              <p className="hint">
                epoch {progress.epoch}/{progress.totalEpochs} · scene {progress.sample}/{progress.totalSamples} ·
                loss {progress.loss.toFixed(5)}
              </p>
            )}
          </div>
        </>
      )}
    </section>
  );
}

