import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Viewer3D } from './components/Viewer3D';
import { BufferCanvas } from './components/BufferCanvas';
import { TrainingPanel } from './components/TrainingPanel';
import { useTrainer, TrainConfig } from './hooks/useTrainer';
import { DatasetPanel, FinetuneOptions } from './components/DatasetPanel';
import {
  RealSample, getAllRealSamples, putRealSample, deleteRealSample, clearRealSamples, toTrainingSample,
} from './data/realDataset';
import { createDepthNet, predictDepth, DepthNet } from './ai/depthnet';
import { fileToImageData, imageDataToGray, resizeGray, GrayImage } from './cv/image';
import { sobel, normalizeToUnit, shadingPseudoDepth } from './cv/filters';
import { depthToPointCloud, depthToMesh } from './cv/geometry';
import { SceneGeometry, RenderStyle } from './viewer/renderer';
import { v3, normalize3 } from './math/vec';

const MODEL_KEY = 'shotput.model.v1';

interface SavedModel {
  size: number;
  baseChannels: number;
  weights: number[][];
  modelInfo: string;
}

const DEFAULT_CONFIG: TrainConfig = {
  size: 24,
  baseChannels: 4,
  epochs: 10,
  learningRate: 0.01,
  datasetSize: 80,
};

type DisplayMode = 'points' | 'mesh' | 'wireframe';

export default function App() {
  const trainer = useTrainer();
  const [config, setConfig] = useState<TrainConfig>(DEFAULT_CONFIG);
  const [net, setNet] = useState<DepthNet | null>(null);
  const [modelInfo, setModelInfo] = useState('');
  const [sourceGray, setSourceGray] = useState<GrayImage | null>(null);
  const [imageName, setImageName] = useState('');
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<DisplayMode>('points');
  const [frameKey, setFrameKey] = useState(0);
  const [samples, setSamples] = useState<RealSample[]>([]);

  // render style controls
  const [pointSize, setPointSize] = useState(2);
  const [shading, setShading] = useState(true);
  const [depthFade, setDepthFade] = useState(true);
  const [showGrid, setShowGrid] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);

  const fileInput = useRef<HTMLInputElement>(null);

  /* -------------------------------------------------------------- */
  /* Derived buffers                                                 */
  /* -------------------------------------------------------------- */

  const gray = useMemo(
    () => (sourceGray ? resizeGray(sourceGray, config.size) : null),
    [sourceGray, config.size],
  );

  const edges = useMemo(() => {
    if (!gray) return null;
    return normalizeToUnit(sobel(gray).magnitude);
  }, [gray]);

  const usedModel = !!net && net.size === config.size;

  const depth = useMemo<Float32Array | null>(() => {
    if (!gray) return null;
    if (usedModel && net) {
      const t = predictDepth(net, gray.data);
      return Float32Array.from(t.data);
    }
    return shadingPseudoDepth(gray);
  }, [gray, net, usedModel]);

  const geometry = useMemo<SceneGeometry | null>(() => {
    if (!depth || !gray) return null;
    const opts = {
      near: 0.8,
      far: 3.2,
      fovY: (50 * Math.PI) / 180,
      stride: config.size <= 24 ? 1 : 2,
      image: gray.data,
    };
    if (displayMode === 'points') {
      return { kind: 'points', cloud: depthToPointCloud(depth, config.size, config.size, opts) };
    }
    return {
      kind: displayMode === 'mesh' ? 'mesh' : 'wireframe',
      mesh: depthToMesh(depth, config.size, config.size, opts),
    };
  }, [depth, gray, displayMode, config.size]);

  const style: RenderStyle = useMemo(
    () => ({
      background: '#0b0f17',
      pointSize,
      light: normalize3(v3(0.4, 0.85, 0.5)),
      ambient: 0.25,
      shading,
      depthFade,
    }),
    [pointSize, shading, depthFade],
  );

  // Re-frame the camera when new geometry appears.
  useEffect(() => {
    if (geometry) setFrameKey((k) => k + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [usedModel, sourceGray, config.size]);

  // Load any real-photo samples persisted in IndexedDB.
  useEffect(() => {
    getAllRealSamples()
      .then((list) => setSamples(list.sort((a, b) => b.createdAt - a.createdAt)))
      .catch(() => undefined);
  }, []);

  // Restore a previously trained model from local storage on first mount.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(MODEL_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as SavedModel;
      const restored = createDepthNet({ size: saved.size, baseChannels: saved.baseChannels });
      restored.model.importWeights(saved.weights);
      setNet(restored);
      setModelInfo(saved.modelInfo);
      setConfig((c) => ({ ...c, size: saved.size, baseChannels: saved.baseChannels }));
    } catch {
      /* ignore corrupt storage */
    }
  }, []);

  /* -------------------------------------------------------------- */
  /* Handlers                                                        */
  /* -------------------------------------------------------------- */

  const onPickFile = useCallback(async (file: File) => {
    try {
      const imgData = await fileToImageData(file, 512);
      setSourceGray(imageDataToGray(imgData));
      setImageName(file.name);
      setImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
    } catch (err) {
      console.error(err);
    }
  }, []);

  const onTrain = useCallback(async () => {
    try {
      const outcome = await trainer.start(config);
      const trained = createDepthNet({ size: outcome.size, baseChannels: outcome.baseChannels });
      trained.model.importWeights(outcome.weights);
      setNet(trained);
      const info = `${trained.describe()} — best loss ${outcome.bestLoss.toFixed(5)} over ${outcome.history.length} epochs`;
      setModelInfo(info);
      try {
        const payload: SavedModel = {
          size: outcome.size,
          baseChannels: outcome.baseChannels,
          weights: outcome.weights,
          modelInfo: info,
        };
        localStorage.setItem(MODEL_KEY, JSON.stringify(payload));
      } catch {
        /* storage may be full or unavailable */
      }
    } catch {
      /* surfaced through trainer.error */
    }
  }, [config, trainer]);

  const onAddSample = useCallback(async (sample: RealSample) => {
    try {
      await putRealSample(sample);
    } catch (err) {
      console.error('failed to persist sample', err);
    }
    setSamples((prev) => [sample, ...prev]);
  }, []);

  const onDeleteSample = useCallback(async (id: string) => {
    try {
      await deleteRealSample(id);
    } catch {
      /* ignore */
    }
    setSamples((prev) => prev.filter((s) => s.id !== id));
  }, []);

  const onClearSamples = useCallback(async () => {
    try {
      await clearRealSamples();
    } catch {
      /* ignore */
    }
    setSamples([]);
  }, []);

  // Transfer learning: continue from the current weights on real photos.
  const onFinetune = useCallback(
    async (opts: FinetuneOptions) => {
      if (samples.length === 0) return;
      const size = net && net.size === config.size ? net.size : config.size;
      const baseChannels = net && net.size === size ? net.baseChannels : config.baseChannels;
      const realSamples = samples.map((s) => toTrainingSample(s, size));
      const weights =
        net && net.size === size && net.baseChannels === baseChannels
          ? net.model.exportWeights()
          : null;

      try {
        const outcome = await trainer.finetune({
          size,
          baseChannels,
          epochs: opts.epochs,
          learningRate: opts.learningRate,
          lambda: opts.lambda,
          weights,
          samples: realSamples,
          syntheticCount: opts.includeSynthetic ? Math.max(8, realSamples.length) : 0,
        });

        const trained = createDepthNet({ size: outcome.size, baseChannels: outcome.baseChannels });
        trained.model.importWeights(outcome.weights);
        setNet(trained);
        const info = `${trained.describe()} — fine-tuned on ${samples.length} real photo(s), best loss ${outcome.bestLoss.toFixed(5)}`;
        setModelInfo(info);
        try {
          const payload: SavedModel = {
            size: outcome.size,
            baseChannels: outcome.baseChannels,
            weights: outcome.weights,
            modelInfo: info,
          };
          localStorage.setItem(MODEL_KEY, JSON.stringify(payload));
        } catch {
          /* ignore */
        }
      } catch {
        /* surfaced through trainer.error */
      }
    },
    [samples, config, net, trainer],
  );

  const onForget = useCallback(() => {
    setNet(null);
    setModelInfo('');
    try {
      localStorage.removeItem(MODEL_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  /* -------------------------------------------------------------- */
  /* Render                                                          */
  /* -------------------------------------------------------------- */

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>3D Shotput</h1>
          <p className="tagline">
            2D image → 3D geometry, using hand-written maths, computer vision and
            your own from-scratch neural network.
          </p>
        </div>
        <div className="header-actions">
          <div className="badge">{usedModel ? 'your trained model' : 'baseline pseudo-depth'}</div>
          {net && (
            <button onClick={onForget}>Forget model</button>
          )}
        </div>
      </header>

      <main className="layout">
        <div className="column left">
          <section className="panel">
            <h2>1 · Load an image</h2>
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onPickFile(f);
              }}
            />
            <div className="row">
              <button className="primary" onClick={() => fileInput.current?.click()}>
                Choose image
              </button>
              <button
                onClick={() => {
                  setSourceGray(null);
                  setImageName('');
                  setImageUrl((prev) => {
                    if (prev) URL.revokeObjectURL(prev);
                    return null;
                  });
                }}
                disabled={!sourceGray}
              >
                Clear
              </button>
            </div>
            {imageName && <p className="hint">Loaded: {imageName}</p>}

            {imageUrl && (
              <img className="preview" src={imageUrl} alt="uploaded" />
            )}

            {gray && (
              <div className="buffer-grid">
                <BufferCanvas data={gray.data} size={config.size} colormap="gray" label="input" />
                <BufferCanvas data={depth} size={config.size} colormap="depth" label={usedModel ? 'predicted depth' : 'pseudo depth'} />
                <BufferCanvas data={edges} size={config.size} colormap="gray" label="sobel edges" />
              </div>
            )}
            {!gray && <p className="hint">No image yet — pick one to build geometry.</p>}
          </section>

          <TrainingPanel
            config={config}
            setConfig={setConfig}
            status={trainer.status}
            progress={trainer.progress}
            history={trainer.history}
            error={trainer.error}
            modelInfo={modelInfo}
            onTrain={onTrain}
            onStop={trainer.stop}
          />

          <DatasetPanel
            samples={samples}
            onAdd={onAddSample}
            onDelete={onDeleteSample}
            onClear={onClearSamples}
            onFinetune={onFinetune}
            training={trainer.status === 'training'}
            hasModel={!!net}
            progress={trainer.progress}
          />
        </div>

        <div className="column right">
          <section className="panel viewer-panel">
            <div className="viewer-head">
              <h2>3 · Explore in 3D</h2>
              <div className="segmented">
                {(['points', 'mesh', 'wireframe'] as DisplayMode[]).map((m) => (
                  <button
                    key={m}
                    className={displayMode === m ? 'active' : ''}
                    onClick={() => setDisplayMode(m)}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            <div className="viewer-wrap">
              <Viewer3D
                geometry={geometry}
                style={style}
                showGrid={showGrid}
                frameKey={frameKey}
                autoRotate={autoRotate}
              />
              {!geometry && <div className="viewer-empty">Load an image to see geometry</div>}
            </div>

            <div className="viewer-controls">
              <label>
                Point size {pointSize}
                <input type="range" min={1} max={6} value={pointSize} onChange={(e) => setPointSize(+e.target.value)} />
              </label>
              <label className="check">
                <input type="checkbox" checked={shading} onChange={(e) => setShading(e.target.checked)} /> shading
              </label>
              <label className="check">
                <input type="checkbox" checked={depthFade} onChange={(e) => setDepthFade(e.target.checked)} /> depth fade
              </label>
              <label className="check">
                <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} /> grid
              </label>
              <label className="check">
                <input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} /> auto-rotate
              </label>
              <button onClick={() => setFrameKey((k) => k + 1)}>Recenter</button>
            </div>
            <p className="hint">Drag to orbit · scroll to zoom · right-drag to pan</p>
          </section>

          <section className="panel">
            <h2>How it works</h2>
            <ol className="steps">
              <li><strong>Math core</strong> — vectors, matrices, quaternions and a pinhole camera, all written by hand.</li>
              <li><strong>Computer vision</strong> — images become float tensors; Sobel, Gaussian and Laplacian operators run alongside the network.</li>
              <li><strong>Your model</strong> — a conv encoder/decoder trained from scratch (forward + backprop + Adam) to predict depth from shaded ray-traced scenes.</li>
              <li><strong>Reconstruction</strong> — the depth map is back-projected to a point cloud / mesh and drawn on canvas with a software renderer.</li>
            </ol>
          </section>
        </div>
      </main>

      <footer className="app-footer">
        <span>Pure TypeScript · no 3D or ML libraries</span>
      </footer>
    </div>
  );
}

