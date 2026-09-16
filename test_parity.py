"""Numerical parity: the C++ runtime vs PyTorch (fp32) and vs the NumPy integer
reference (int8), on 25 test images. Also checks int8 vs fp32 agree on argmax.

    .venv/bin/python test_parity.py
"""
import os, struct, subprocess, sys
import numpy as np
import torch
from train import EdgeNet, DATA
from lower import graph, quantize, calibrate, ref_int8, run_fp32

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "edgeforge")
N = 25


def cpp_logits(model_path, i):
    out = subprocess.check_output([BIN, "dump", model_path, os.path.join(DATA, "mnist_test.bin"), str(i)], text=True)
    return np.array([float(v) for v in out.split()], dtype=np.float32)


def load_test(n):
    with open(os.path.join(DATA, "mnist_test.bin"), "rb") as f:
        total = struct.unpack("<I", f.read(4))[0]
        xs, ys = [], []
        for _ in range(min(n, total)):
            ys.append(f.read(1)[0])
            px = np.frombuffer(f.read(784), dtype=np.uint8).astype(np.float32).reshape(1, 28, 28)
            xs.append((px / 255.0 - 0.1307) / 0.3081)
    return np.array(xs, dtype=np.float32), np.array(ys)


def main():
    assert os.path.exists(BIN), "build first: make"
    xs, ys = load_test(N)
    sd = torch.load(os.path.join(DATA, "model_fp32.pt"))
    net = EdgeNet(); net.load_state_dict(sd); net.eval()
    layers = graph(sd)
    calib = np.load(os.path.join(DATA, "calib.npz"))
    q = quantize(layers, np.float32(calib["s_in"]), [np.float32(s) for s in calib["s_outs"]])

    worst32 = worst8 = 0.0
    argmax_agree = 0
    for i in range(N):
        with torch.no_grad():
            torch_logits = net(torch.from_numpy(xs[i:i + 1])).numpy()[0]
        c32 = cpp_logits(os.path.join(DATA, "model_fp32.efm"), i)
        c8 = cpp_logits(os.path.join(DATA, "model_int8.efm"), i)
        ref8 = ref_int8(q, xs[i], np.float32(calib["s_in"]))
        worst32 = max(worst32, float(np.abs(c32 - torch_logits).max()))
        worst8 = max(worst8, float(np.abs(c8 - ref8).max()))
        argmax_agree += int(c8.argmax() == c32.argmax())

    print(f"fp32 runtime vs PyTorch:      max |diff| = {worst32:.2e}  (tolerance 1e-4)")
    print(f"int8 runtime vs NumPy int ref: max |diff| = {worst8:.2e}  (tolerance 1e-3)")
    print(f"int8 vs fp32 argmax agreement: {argmax_agree}/{N}")
    ok = worst32 < 1e-4 and worst8 < 1e-3 and argmax_agree >= N - 1
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
