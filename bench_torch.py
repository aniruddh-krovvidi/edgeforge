"""Baseline: PyTorch eager, CPU, batch size 1 -- what the same network costs
per image before lowering. (torch.compile is not used: on an edge target you
would not ship a Python interpreter at all, which is the point of the runtime.)

    .venv/bin/python bench_torch.py
"""
import os, time
import numpy as np
import torch
from train import EdgeNet, DATA

torch.set_num_threads(1)   # edge-like: single core, same as the C++ runtime
net = EdgeNet(); net.load_state_dict(torch.load(os.path.join(DATA, "model_fp32.pt"))); net.eval()
x = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    for _ in range(200): net(x)                       # warm-up
    us = []
    for _ in range(5000):
        t0 = time.perf_counter(); net(x); us.append((time.perf_counter() - t0) * 1e6)
us = np.sort(us)
print(f"torch eager cpu fp32 (1 thread): 5000 iters  mean {us.mean():.1f} us  p50 {us[len(us)//2]:.1f} us  p99 {us[int(len(us)*0.99)]:.1f} us")
