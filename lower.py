"""Lower the trained PyTorch model to the edge runtime format (.efm).

Two passes:
  fp32  : straight export of the graph + float weights
  int8  : post-training quantization -- symmetric per-output-channel weights,
          per-tensor activation scales from a calibration set, int32 biases,
          requantization fused with ReLU. This is the "compress to run in low
          precision" step; the C++ runtime executes the integer graph.

Also contains the NumPy integer reference (`ref_int8`) the C++ runtime must
match, and a fake-quant evaluation so accuracy can be reported per precision.

    .venv/bin/python lower.py
Outputs: data/model_fp32.efm, data/model_int8.efm, data/calib.npz
"""
import os, struct
import numpy as np
import torch
from train import EdgeNet, loaders, DATA

MAGIC = b"EFM1"
CONV, RELU, POOL, LINEAR, FLATTEN = 1, 2, 3, 4, 5
f32 = np.float32


def rint(x):
    return np.rint(x).astype(np.int32)


def qsym(x, scale):
    return np.clip(rint(x / scale), -128, 127).astype(np.int8)


# ---------------- graph description (fixed for EdgeNet) ----------------

def graph(sd):
    """Ordered layer list with float weights. Shapes: conv w [oc, ic, k, k]; linear w [of, if]."""
    return [
        ("conv", sd["conv1.weight"].numpy(), sd["conv1.bias"].numpy()), ("relu",), ("pool", 2),
        ("conv", sd["conv2.weight"].numpy(), sd["conv2.bias"].numpy()), ("relu",), ("pool", 2),
        ("flatten",),
        ("linear", sd["fc1.weight"].numpy(), sd["fc1.bias"].numpy()), ("relu",),
        ("linear", sd["fc2.weight"].numpy(), sd["fc2.bias"].numpy()),
    ]


# ---------------- float reference (NumPy, matches PyTorch) ----------------

def conv2d(x, w, b, pad=1):  # x [C,H,W]
    C, H, W = x.shape
    OC, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((OC, H, W), dtype=x.dtype)
    for oc in range(OC):
        acc = np.zeros((H, W), dtype=np.float64)
        for ic in range(C):
            for kh in range(k):
                for kw in range(k):
                    acc += w[oc, ic, kh, kw].astype(np.float64) * xp[ic, kh:kh + H, kw:kw + W]
        out[oc] = acc + b[oc]
    return out


def maxpool(x, k):
    C, H, W = x.shape
    return x.reshape(C, H // k, k, W // k, k).max(axis=(2, 4))


def run_fp32(layers, x):
    for L in layers:
        if L[0] == "conv": x = conv2d(x, L[1], L[2])
        elif L[0] == "relu": x = np.maximum(x, 0)
        elif L[0] == "pool": x = maxpool(x, L[1])
        elif L[0] == "flatten": x = x.reshape(-1)
        elif L[0] == "linear": x = L[1] @ x + L[2]
    return x


# ---------------- calibration + quantized graph ----------------

def calibrate(layers, xs):
    """Per-tensor activation scales: input, and the output of every conv/linear (pre-ReLU)."""
    maxabs_in = max(float(np.abs(x).max()) for x in xs)
    outs = []
    for x in xs:
        cur, i = x, 0
        for L in layers:
            if L[0] == "conv": cur = conv2d(cur, L[1], L[2]); outs.append((i, float(np.abs(cur).max()))); i += 1
            elif L[0] == "linear": cur = L[1] @ cur + L[2]; outs.append((i, float(np.abs(cur).max()))); i += 1
            elif L[0] == "relu": cur = np.maximum(cur, 0)
            elif L[0] == "pool": cur = maxpool(cur, L[1])
            elif L[0] == "flatten": cur = cur.reshape(-1)
    n = max(i for i, _ in outs) + 1
    s_out = [max(m for i, m in outs if i == j) / 127.0 for j in range(n)]
    return f32(maxabs_in / 127.0), [f32(s) for s in s_out]


def quantize(layers, s_in, s_outs):
    """Returns the int8 graph: same layer list, weights replaced by (q_w, s_w, q_b, s_in_layer, s_out)."""
    q, s_cur, j = [], s_in, 0
    n_weighted = len(s_outs)
    for L in layers:
        if L[0] in ("conv", "linear"):
            w, b = L[1].astype(f32), L[2].astype(f32)
            axes = tuple(range(1, w.ndim))
            s_w = (np.abs(w).max(axis=axes) / 127.0).astype(f32)
            s_w[s_w == 0] = f32(1e-8)
            q_w = qsym(w, s_w.reshape((-1,) + (1,) * (w.ndim - 1)))
            q_b = rint(b / (s_w * s_cur))
            last = j == n_weighted - 1
            s_out = f32(0) if last else s_outs[j]          # 0 => emit float logits
            q.append((L[0], q_w, s_w, q_b, s_cur, s_out))
            s_cur = s_out
            j += 1
        else:
            q.append(L)
    return q


def ref_int8(qlayers, x, s_in):
    """Integer reference the C++ runtime must reproduce. x: float input [C,H,W]."""
    cur = qsym(x, s_in)
    for L in qlayers:
        kind = L[0]
        if kind in ("conv", "linear"):
            _, q_w, s_w, q_b, s_x, s_out = L
            if kind == "conv":
                acc = conv2d(cur.astype(np.int32), q_w.astype(np.int32), q_b.astype(np.int32))
                acc = acc.astype(np.int32)
                y = acc.astype(f32) * (s_w * s_x).astype(f32).reshape(-1, 1, 1)
            else:
                acc = (q_w.astype(np.int32) @ cur.astype(np.int32)) + q_b
                y = acc.astype(f32) * (s_w * s_x).astype(f32)
            if s_out == 0:
                return y                                       # float logits
            cur = np.clip(rint(y / s_out), -128, 127).astype(np.int8)
        elif kind == "relu": cur = np.maximum(cur, 0)
        elif kind == "pool": cur = maxpool(cur, L[1])
        elif kind == "flatten": cur = cur.reshape(-1)
    return cur


# ---------------- .efm writer ----------------

def write_efm(path, layers, dtype, s_in, in_shape):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIIIf", dtype, len(layers), *in_shape, float(s_in)))
        for L in layers:
            kind = L[0]
            if kind == "conv":
                w = L[1]
                oc, ic, k, _ = w.shape
                f.write(struct.pack("<IIIIII", CONV, ic, oc, k, 1, k // 2))
                _write_weighted(f, L, dtype, oc)
            elif kind == "linear":
                of, inf = L[1].shape
                f.write(struct.pack("<III", LINEAR, inf, of))
                _write_weighted(f, L, dtype, of)
            elif kind == "relu": f.write(struct.pack("<I", RELU))
            elif kind == "pool": f.write(struct.pack("<II", POOL, L[1]))
            elif kind == "flatten": f.write(struct.pack("<I", FLATTEN))


def _write_weighted(f, L, dtype, n_out):
    if dtype == 0:
        f.write(struct.pack("<f", 0.0)); f.write(np.ones(n_out, f32).tobytes())
        f.write(L[1].astype(f32).tobytes()); f.write(L[2].astype(f32).tobytes())
    else:
        _, q_w, s_w, q_b, s_x, s_out = L
        # the runtime's per-channel multiplier is s_w * s_x, computed here exactly as ref_int8 does
        f.write(struct.pack("<f", float(s_out))); f.write((s_w * s_x).astype(f32).tobytes())
        f.write(q_w.astype(np.int8).tobytes()); f.write(q_b.astype(np.int32).tobytes())


def main():
    sd = torch.load(os.path.join(DATA, "model_fp32.pt"))
    layers = graph(sd)
    _, test = loaders()
    xb0, _ = next(iter(test))
    xs = [xb0[i].numpy() for i in range(256)]             # calibration set: 256 images [1,28,28]
    s_in, s_outs = calibrate(layers, xs)
    q = quantize(layers, s_in, s_outs)

    write_efm(os.path.join(DATA, "model_fp32.efm"), layers, 0, 1.0, (1, 28, 28))
    write_efm(os.path.join(DATA, "model_int8.efm"), q, 1, s_in, (1, 28, 28))

    # accuracy per precision on 1000 test images (NumPy references; the C++ runtime evals all 10k)
    xb, yb = xb0.numpy(), _.numpy()
    ok32 = sum(int(run_fp32(layers, x).argmax() == y) for x, y in zip(xb, yb))
    ok8 = sum(int(ref_int8(q, x, s_in).argmax() == y) for x, y in zip(xb, yb))
    size32 = sum(L[1].size * 4 + L[2].size * 4 for L in layers if L[0] in ("conv", "linear"))
    size8 = sum(L[1].size + L[2].size * 4 for L in q if L[0] in ("conv", "linear"))
    n = len(yb)
    print(f"fp32: acc {ok32/n:.4f}  weights {size32/1024:.1f} KiB")
    print(f"int8: acc {ok8/n:.4f}  weights {size8/1024:.1f} KiB   (s_in={s_in:.5f})")
    np.savez(os.path.join(DATA, "calib.npz"), s_in=s_in, s_outs=np.array(s_outs))
    print("wrote data/model_fp32.efm, data/model_int8.efm")


if __name__ == "__main__":
    main()
