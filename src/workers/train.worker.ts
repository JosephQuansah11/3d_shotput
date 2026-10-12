/// <reference lib="webworker" />
/**
 * workers/train.worker.ts — runs training off the main thread.
 *
 * Two entry points:
 *  - 'start'    : train a fresh model on ray-traced synthetic scenes.
 *  - 'finetune' : continue from existing weights (transfer learning) on a mix
 *                 of real, human-labelled photos and optional synthetic scenes,
 *                 using a masked loss with a smoothness prior.
 */

import { createDepthNet } from '../ai/depthnet';
import { train, TrainProgress } from '../ai/trainer';
import { MSE, MaskedMSEWithTV } from '../ai/loss';
import { Adam } from '../ai/optim';
import { generateDataset, DEFAULT_RENDER, Sample } from '../data/dataset';
import { AugmentOptions } from '../data/augment';

export interface TrainStartMessage {
  type: 'start';
  size: number;
  baseChannels: number;
  epochs: number;
  learningRate: number;
  datasetSize: number;
  augment: AugmentOptions | null;
}

export interface FinetuneMessage {
  type: 'finetune';
  size: number;
  baseChannels: number;
  epochs: number;
  learningRate: number;
  lambda: number;
  /** starting weights for transfer learning, or null for a fresh model */
  weights: number[][] | null;
  /** real, human-labelled samples already resampled to `size` */
  samples: Sample[];
  /** extra ray-traced scenes to mix in and prevent forgetting */
  syntheticCount: number;
  augment: AugmentOptions | null;
}

export type WorkerInMessage = TrainStartMessage | FinetuneMessage | { type: 'stop' };

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

const post = (msg: WorkerOutMessage) => (self as unknown as Worker).postMessage(msg);

function run(
  msg: TrainStartMessage | FinetuneMessage,
  dataset: Sample[],
  loss: MSE | MaskedMSEWithTV,
): void {
  const net = createDepthNet({ size: msg.size, baseChannels: msg.baseChannels });
  if (msg.type === 'finetune' && msg.weights) net.model.importWeights(msg.weights);

  const optimizer = new Adam();
  const result = train(
    net.model,
    loss,
    optimizer,
    dataset,
    { epochs: msg.epochs, learningRate: msg.learningRate, reportEvery: 10, augment: msg.augment },
    (progress) => post({ type: 'progress', progress }),
    () => stopRequested,
  );

  post({
    type: 'done',
    weights: net.model.exportWeights(),
    history: result.history,
    bestLoss: result.bestLoss,
    size: msg.size,
    baseChannels: msg.baseChannels,
  });
}

self.onmessage = (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;
  if (msg.type === 'stop') {
    stopRequested = true;
    return;
  }

  try {
    stopRequested = false;
    if (msg.type === 'start') {
      const dataset = generateDataset(msg.datasetSize, { ...DEFAULT_RENDER, size: msg.size });
      run(msg, dataset, new MSE());
    } else if (msg.type === 'finetune') {
      const synthetic = msg.syntheticCount > 0
        ? generateDataset(msg.syntheticCount, { ...DEFAULT_RENDER, size: msg.size })
        : [];
      run(msg, [...msg.samples, ...synthetic], new MaskedMSEWithTV(msg.lambda));
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
