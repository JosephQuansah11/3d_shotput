import { TrainProgress } from '../ai/trainer';
import { TrainConfig, TrainStatus } from '../hooks/useTrainer';

export interface TrainingPanelProps {
  config: TrainConfig;
  setConfig: (c: TrainConfig) => void;
  status: TrainStatus;
  progress: TrainProgress | null;
  history: number[];
  error: string | null;
  modelInfo: string;
  onTrain: () => void;
  onStop: () => void;
}

const pct = (n: number, d: number) => (d ? Math.round((n / d) * 100) : 0);

/** Controls for training the from-scratch depth model, plus live loss readout. */
export function TrainingPanel(props: TrainingPanelProps) {
  const { config, setConfig, status, progress, history, error, modelInfo, onTrain, onStop } = props;
  const training = status === 'training';

  const update = (patch: Partial<TrainConfig>) => setConfig({ ...config, ...patch });

  const epochPct = progress ? pct(progress.epoch - 1, progress.totalEpochs) : 0;
  const samplePct = progress ? pct(progress.sample, progress.totalSamples) : 0;
  const overall = progress ? Math.round((epochPct + samplePct / 100) / progress.totalEpochs * 100) : 0;

  return (
    <section className="panel">
      <h2>Train your model</h2>
      <p className="hint">
        A convolutional network implemented from scratch learns image&nbsp;→&nbsp;depth from
        ray-traced scenes. No pretrained weights — you own every parameter.
      </p>

      <div className="grid2">
        <label>
          Image size
          <select value={config.size} disabled={training} onChange={(e) => update({ size: +e.target.value })}>
            {[16, 20, 24, 32, 40, 48].map((s) => (
              <option key={s} value={s}>{s}×{s}</option>
            ))}
          </select>
        </label>
        <label>
          Base channels
          <select value={config.baseChannels} disabled={training} onChange={(e) => update({ baseChannels: +e.target.value })}>
            {[4, 6, 8, 12].map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
        <label>
          Epochs
          <input type="number" min={1} max={200} value={config.epochs} disabled={training}
            onChange={(e) => update({ epochs: +e.target.value })} />
        </label>
        <label>
          Learning rate
          <input type="number" step={0.001} min={0.0001} max={1} value={config.learningRate} disabled={training}
            onChange={(e) => update({ learningRate: +e.target.value })} />
        </label>
        <label>
          Scenes
          <input type="number" min={10} max={2000} value={config.datasetSize} disabled={training}
            onChange={(e) => update({ datasetSize: +e.target.value })} />
        </label>
        <label>
          Optimizer
          <input value="Adam" disabled />
        </label>
      </div>

      <div className="row">
        <button className="primary" onClick={onTrain} disabled={training}>
          {training ? 'Training…' : 'Train model'}
        </button>
        <button onClick={onStop} disabled={!training}>Stop</button>
      </div>

      {modelInfo && <p className="model-info">{modelInfo}</p>}

      {progress && (
        <div className="progress-block">
          <div className="progress-bar"><div style={{ width: `${overall}%` }} /></div>
          <div className="progress-meta">
            <span>epoch {progress.epoch}/{progress.totalEpochs}</span>
            <span>scene {progress.sample}/{progress.totalSamples}</span>
            <span>loss {progress.loss.toFixed(5)}</span>
            <span>best {progress.bestLoss.toFixed(5)}</span>
          </div>
        </div>
      )}

      {history.length > 0 && <LossChart history={history} />}
      {error && <p className="error">{error}</p>}
    </section>
  );
}

/** Tiny inline SVG loss curve. */
function LossChart({ history }: { history: number[] }) {
  const w = 320;
  const h = 90;
  const max = Math.max(...history, 1e-6);
  const min = Math.min(...history);
  const range = max - min || 1;
  const points = history
    .map((v, i) => {
      const x = (i / Math.max(1, history.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 8) - 4;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg className="loss-chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <polyline points={points} fill="none" stroke="#5ad1ff" strokeWidth="2" />
    </svg>
  );
}
