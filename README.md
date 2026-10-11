# 3D Shotput

**Turn a 2D image into explorable 3D geometry — with pure maths, computer vision, and a neural network you build and train from scratch.**

3D Shotput is a React + TypeScript platform that reconstructs 3D shape from a
single photograph and renders it live on a canvas. Every layer is hand-written:
no three.js, no TensorFlow, no pretrained weights. The only runtime dependency is React.

```
2D image ──▶ CV preprocessing ──▶ your CNN (image → depth) ──▶ back-projection ──▶ 3D canvas
             (Sobel, Gaussian)     (conv encoder/decoder)      (pinhole model)     (software renderer)
```

## Why it exists

The goal is to *show how* a 2D image becomes 3D — the maths and the learning,
not a black box. The code is organised as a stack you can read top to bottom:

| Layer | Folder | What's inside |
| --- | --- | --- |
| **Math core** | `src/math` | Vectors, 4×4 matrices (column-major), quaternions, an orbit camera, `lookAt`, perspective/orthographic projections, general 4×4 inversion, normal matrices. |
| **Computer vision** | `src/cv` | Image → float tensor with bilinear resampling; Sobel, Gaussian, Laplacian convolution; depth→point-cloud and depth→mesh back-projection. |
| **Neural network** | `src/ai` | A tiny tensor library with `im2col`; `Dense`, `Conv2D`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `MaxPool2D`, `Upsample2D`, `Flatten`; a `Sequential` model; MSE/MAE losses; SGD-with-momentum and Adam; a training loop. |
| **Data** | `src/data` | A from-scratch ray tracer that renders random 3D scenes into matched *(image, ground-truth depth)* pairs — the supervision the model learns from. |
| **Renderer** | `src/viewer` | A software rasteriser: matrix transform → perspective divide → painter's algorithm → flat Lambert shading, all onto a 2D canvas. |
| **UI** | `src/components`, `src/App.tsx` | React controls, the canvas viewer with orbit/zoom/pan, live buffers (input, depth, edges) and the training panel. |
| **Worker** | `src/workers` | Training runs in a Web Worker so the interface never blocks. |

## The model

`DepthNet` is a small convolutional encoder/decoder:

```
conv(1→c) → relu → pool
conv(c→2c) → relu → pool
conv(2c→2c) → relu
upsample → conv(2c→c) → relu
upsample → conv(c→1) → sigmoid        →  depth map in [0,1]
```

It is trained with Adam to minimise MSE against the ray-traced depth maps. You
choose the resolution, channel width, epochs, learning rate and dataset size in
the UI and watch the loss curve fall in real time.

## Running it

```bash
npm install
npm run dev        # start the dev server
npm run build      # production build
npm test           # unit tests + gradient checks + an end-to-end training test
```

Then in the browser:

1. **Load an image** (or leave it empty to see the baseline).
2. **Train model** — pick a size (16–48), channel width, epochs and scene count,
   then press *Train*. A worker builds the synthetic dataset and optimises the
   network.
3. **Explore in 3D** — switch between *points*, *mesh* and *wireframe*, orbit
   with the mouse, scroll to zoom, right-drag to pan.

Until you train a model, the viewer uses a non-learned "shape-from-shading"
pseudo-depth so the pipeline is visible end to end immediately.

## Correctness

The maths is not just plausible — it's tested. `npm test` runs:

- matrix inverse identity (`A·A⁻¹ = I`), quaternion↔matrix agreement, vector identities;
- `im2col`/`col2im` round-trip;
- **numerical gradient checks** for `Dense`, `Conv2D` and `Conv2D+ReLU`, comparing
  the hand-derived backprop against central finite differences;
- an end-to-end training test proving the loss decreases on ray-traced data.

## Extending it

- Add a residual block or batch-norm layer in `src/ai/layers.ts`.
- Swap the ray-traced dataset for real photographs once you have depth labels.
- Feed the depth map into `depthToMesh` with a different `discontinuity`
  threshold to trade silhouette fidelity for connectivity.
- Replace the software renderer with WebGL/WebGPU — the matrices in `src/math`
  are already column-major and GPU-ready.
