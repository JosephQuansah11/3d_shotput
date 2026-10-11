/**
 * ai/trainer.ts — the training loop.
 *
 * Plain supervised learning: forward pass, loss, backprop, optimizer step.
 * Nothing here is library magic — it's the same four lines you'd write on a
 * whiteboard, running over the synthetic dataset.
 */

import { Sequential } from './model';
import { Optimizer } from './optim';
import { Loss } from './loss';
import { sampleToTensors } from './depthnet';
import { Sample } from '../data/dataset';

export interface TrainConfig {
  epochs: number;
  learningRate: number;
  /** how often (in samples) to report progress */
  reportEvery?: number;
}

export interface TrainProgress {
  epoch: number;
  totalEpochs: number;
  sample: number;
  totalSamples: number;
  loss: number; // running average this epoch
  bestLoss: number;
}

export interface TrainResult {
  history: number[]; // average loss per epoch
  bestLoss: number;
}

/**
 * Train `model` on `samples`. `onProgress` is called periodically so a Web
 * Worker can stream updates back to the UI without blocking it.
 */
export function train(
  model: Sequential,
  loss: Loss,
  optimizer: Optimizer,
  samples: Sample[],
  config: TrainConfig,
  onProgress?: (p: TrainProgress) => void,
  shouldStop?: () => boolean,
): TrainResult {
  const history: number[] = [];
  let bestLoss = Infinity;
  const reportEvery = config.reportEvery ?? 25;

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    // Fisher–Yates shuffle so each epoch sees a different order
    const order = samples.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }

    let running = 0;
    let seen = 0;

    for (let s = 0; s < order.length; s++) {
      if (shouldStop && shouldStop()) return { history, bestLoss };

      const sample = samples[order[s]];
      const { input, target } = sampleToTensors(sample);

      const pred = model.forward(input);
      const { loss: l, grad } = loss.compute(pred, target);

      model.zeroGrad();
      model.backward(grad);
      optimizer.step(model.params(), config.learningRate);

      running += l;
      seen += 1;

      if (seen % reportEvery === 0 || s === order.length - 1) {
        const avg = running / seen;
        onProgress?.({
          epoch: epoch + 1,
          totalEpochs: config.epochs,
          sample: seen,
          totalSamples: order.length,
          loss: avg,
          bestLoss: Math.min(bestLoss, avg),
        });
      }
    }

    const epochLoss = running / seen;
    history.push(epochLoss);
    bestLoss = Math.min(bestLoss, epochLoss);
  }

  return { history, bestLoss };
}
