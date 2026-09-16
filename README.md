# edgeforge

Lower a PyTorch network to a compact integer graph and run it in a
from-scratch **C++17 edge runtime** — the train → compress → deploy path for a
model that has to run on a compute-constrained device.

```
 PyTorch (train.py)      lower.py                              runtime.cpp
 ┌──────────────┐   ┌────────────────────────────┐   ┌──────────────────────────────┐
 │ EdgeNet fp32 │──►│ calibrate activations       │──►│ .efm loader                  │
 │ conv-relu-   │   │ per-channel int8 weights    │   │ int8 conv / linear kernels   │
 │ pool ×2, fc  │   │ int32 biases, fused requant │   │ NEON dot products (arm64)    │
 └──────────────┘   │ + NumPy integer reference   │   │ requant fused with ReLU      │
                    └────────────────────────────┘   │ eval | bench | dump          │
                                                     └──────────────────────────────┘
                                                        cuda/edgeforge_cuda.cu: same graph
                                                        on an NVIDIA GPU, parity-checked
```

- **Lowering**: post-training quantization — symmetric per-output-channel
  int8 weights, per-tensor activation scales from a 256-image calibration
  set, int32 biases (`round(b / (s_w · s_x))`), requantization fused with the
  following ReLU, float logits on the last layer. Exported to a small binary
  format (`.efm`) with no runtime dependencies.
- **Runtime**: one C++ file. int8 × int8 → int32 accumulation, per-channel
  float multiplier, round-to-nearest-even requant (`lrintf`, matches
  `numpy.rint`). NEON kernels on arm64 (`vmull_s8`/`vpadalq_s16` for int8,
  `vfmaq_f32` with two accumulators for fp32); scalar fallback elsewhere.
- **Verification**: `test_parity.py` compares the C++ runtime against
  PyTorch (fp32) and against a NumPy integer reference that implements the
  exact quantized arithmetic (int8). Both agree to float-printing precision.

## Results (MNIST, 10,000 test images, Apple M4 Max, single core)

| path                              | accuracy   | weights   | p50 latency / image | p99      |
|-----------------------------------|-----------:|----------:|--------------------:|---------:|
| PyTorch eager, CPU fp32, 1 thread |    98.00 % | 203.7 KiB |             56.5 µs |  68.9 µs |
| edgeforge fp32, scalar C++        |    98.00 % | 203.7 KiB |             95.8 µs | 166.7 µs |
| edgeforge fp32, NEON              |    98.00 % | 203.7 KiB |             49.4 µs |  66.8 µs |
| **edgeforge int8, NEON**          | **97.93 %**| **51.2 KiB** |          49.5 µs |  66.4 µs |

Parity (25 images): fp32 runtime vs PyTorch max |Δlogit| = 2.0e-6; int8
runtime vs NumPy integer reference = 9.5e-7; int8 vs fp32 argmax agree 25/25.

What the numbers say, honestly:
- int8 costs **0.07 points** of accuracy for a **4× smaller** model. That is
  the win that matters on an edge device: weights fit in cache / on-chip SRAM.
- On this *tiny* network int8 and fp32 NEON tie at ~49 µs. Profiling shows the
  per-pixel receptive-field gather (an im2col-style copy) dominates, not the
  multiply-accumulate; the 3×3 reductions are only 9 and 72 elements long, too
  short for the int8 kernel's throughput advantage to show. A direct
  convolution that reads the input in place, and processing several output
  channels per gathered patch, is the next optimization — and on a real edge
  model with wider channels the int8 path pulls ahead.
- Naive scalar C++ loses to PyTorch's vendor kernels by ~2×; the NEON
  rewrite closes that gap and passes it. Vectorization is not optional.

## CUDA status

`cuda/edgeforge_cuda.cu` implements the int8 graph on an NVIDIA GPU
(one-thread-per-output conv kernel, warp-per-row linear kernel with shuffle
reduction, fused requant) and a harness that runs the CPU runtime and the GPU
side by side and asserts parity. It was written on a machine without an
NVIDIA GPU and is **not exercised in CI**; validate with
`make edgeforge_cuda && ./edgeforge_cuda data/model_int8.efm data/mnist_test.bin`
on any CUDA box (a free Colab T4 works).

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision numpy
.venv/bin/python train.py        # 2 epochs, ~10 s on Apple Silicon → data/model_fp32.pt
.venv/bin/python lower.py        # calibrate + quantize → data/model_{fp32,int8}.efm
make                             # builds ./edgeforge  (macOS w/ broken Xcode shim: see Makefile header)
make test                        # eval both precisions + parity test
make bench                       # per-image latency
.venv/bin/python bench_torch.py  # PyTorch baseline
```

## Files

- `train.py` — EdgeNet + MNIST training, raw test-set export
- `lower.py` — calibration, PTQ int8, `.efm` writer, NumPy fp32 and integer references
- `runtime.cpp` — the edge runtime (loader, kernels, CLI)
- `cuda/edgeforge_cuda.cu` — CUDA kernels + CPU/GPU parity harness
- `test_parity.py`, `bench_torch.py`, `Makefile`, `.github/workflows/ci.yml`

## Deliberate simplifications

- The graph is fixed (conv/relu/pool/linear/flatten); there is no general op
  registry or graph IR. Adding an op is one `case` in the runtime and one
  branch in `lower.py`.
- Per-tensor activation scales, static (calibration-time). Per-channel
  activations or QAT would recover the last 0.07 points.
- Single-threaded by design: it models one edge core. Batch-1 only.
- MNIST is a stand-in dataset; the lowering and runtime are dataset-agnostic.
