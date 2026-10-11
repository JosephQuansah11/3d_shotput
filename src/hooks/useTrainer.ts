import { useCallback, useRef, useState } from 'react';
import { TrainProgress } from '../ai/trainer';
import type { WorkerInMessage, WorkerOutMessage } from '../workers/train.worker';

export interface TrainConfig {
  size: number;
  baseChannels: number;
  epochs: number;
  learningRate: number;
  datasetSize: number;
}

export interface TrainOutcome {
  weights: number[][];
  history: number[];
  bestLoss: number;
  size: number;
  baseChannels: number;
}

export type TrainStatus = 'idle' | 'training' | 'done' | 'error';

/** Manages the training Web Worker and exposes its lifecycle to React. */
export function useTrainer() {
  const [status, setStatus] = useState<TrainStatus>('idle');
  const [progress, setProgress] = useState<TrainProgress | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  const stop = useCallback(() => {
    workerRef.current?.postMessage({ type: 'stop' } satisfies WorkerInMessage);
  }, []);

  const start = useCallback(
    (config: TrainConfig): Promise<TrainOutcome> => {
      workerRef.current?.terminate();
      setStatus('training');
      setError(null);
      setProgress(null);
      setHistory([]);

      return new Promise<TrainOutcome>((resolve, reject) => {
        const worker = new Worker(new URL('../workers/train.worker.ts', import.meta.url), {
          type: 'module',
        });
        workerRef.current = worker;

        worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
          const msg = event.data;
          if (msg.type === 'progress') {
            setProgress(msg.progress);
          } else if (msg.type === 'done') {
            setStatus('done');
            setHistory(msg.history);
            worker.terminate();
            workerRef.current = null;
            resolve(msg);
          } else if (msg.type === 'error') {
            setStatus('error');
            setError(msg.message);
            worker.terminate();
            workerRef.current = null;
            reject(new Error(msg.message));
          }
        };
        worker.onerror = (e) => {
          setStatus('error');
          setError(e.message);
          reject(new Error(e.message));
        };

        worker.postMessage({ type: 'start', ...config } satisfies WorkerInMessage);
      });
    },
    [],
  );

  return { status, progress, history, error, start, stop };
}
