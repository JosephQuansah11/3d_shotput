/// <reference lib="webworker" />
/**
 * workers/train.worker.ts — runs training off the main thread.
 *
 * The worker builds a synthetic dataset with the ray tracer, constructs a
 * fresh DepthNet and optimises it with Adam. Weights are posted back to the
 * UI when training finishes so the main thread can run inference.
 */

import { createDepthNet } from '../ai/depthnet';
import { train, TrainProgress } from '../ai/trainer';
import { MSE } from '../ai/loss';
import { Adam } from '../ai/optim';
import { generateDataset, DEFAULT_RENDER } from '../data/dataset';

export interface TrainStartMessage {
  type: 'start';
  size: number;
  baseChannels: number;
  epochs: number;
  learningRate: number;
  datasetSize: number;
}

export type WorkerInMessage = TrainStartMessage | { type: 'stop' };

export type WorkerOutMessage =
  | { type: 'progress'; progress: TrainProgress }
  | {
      type: 'done';
      weights: number[][];
      history: number[];
      bestLoss: number;
      size: number;
      baseChannels: number;
    }
  | { type: 'error'; message: string };

let stopRequested = false;

self.onmessage = (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;

  if (msg.type === 'stop') {
    stopRequested = true;
    return;
  }

  if (msg.type !== 'start') return;

  try {
    stopRequested = false;

    const renderOpts = { ...DEFAULT_RENDER, size: msg.size };
    const dataset = generateDataset(msg.datasetSize, renderOpts);

    const net = createDepthNet({ size: msg.size, baseChannels: msg.baseChannels });
    const optimizer = new Adam();

    const result = train(
      net.model,
      new MSE(),
      optimizer,
      dataset,
      { epochs: msg.epochs, learningRate: msg.learningRate, reportEvery: 10 },
      (progress) => {
        (self as unknown as Worker).postMessage({ type: 'progress', progress });
      },
      () => stopRequested,
    );

    (self as unknown as Worker).postMessage({
      type: 'done',
      weights: net.model.exportWeights(),
      history: result.history,
      bestLoss: result.bestLoss,
      size: msg.size,
      baseChannels: msg.baseChannels,
    });
  } catch (err) {
    (self as unknown as Worker).postMessage({
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
