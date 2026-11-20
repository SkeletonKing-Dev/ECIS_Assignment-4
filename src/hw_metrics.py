"""
Hardware measurements on the CPU: parameter count, MACs, storage size,
latency, throughput and peak inference memory.

Used by the hardware-aware NAS (constraint checks) and by the benchmark.
"""
import contextlib
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model, input_size):
    """Multiply-accumulates of Conv2d and Linear layers for one image."""
    macs = []

    def conv_hook(m, i, o):
        macs.append(o.numel() * m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups))

    def fc_hook(m, i, o):
        macs.append(m.in_features * m.out_features)

    hooks = [m.register_forward_hook(conv_hook if isinstance(m, nn.Conv2d) else fc_hook)
             for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    model.eval()
    model(torch.zeros(1, *input_size))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return sum(macs)


def storage_size_mb(model):
    """Size of the serialized FP32 state_dict (what is stored on the device)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2 ** 20


@torch.no_grad()
def latency_ms(fn, x, warmup=10, runs=50):
    """Mean and p95 latency of fn(x) in milliseconds."""
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return float(t.mean()), float(np.percentile(t, 95))


@contextlib.contextmanager
def quiet_stderr():
    """The profiler's C++ backend prints start/stop lines to stderr; hide them."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


@torch.no_grad()
def peak_memory_mb(model, x):
    """Peak inference memory = parameters/buffers + peak activation memory.

    Activation memory is measured with the PyTorch profiler: every CPU
    allocation and free during one forward pass is recorded as a '[memory]'
    event; replaying them in time order gives the peak of live tensor memory.
    Returns (peak_mb, method). If the profiler does not expose memory events
    on this PyTorch version, falls back to an estimate from layer output sizes."""
    weights = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
    try:
        from torch.profiler import ProfilerActivity, profile
        model(x)
        with quiet_stderr(), profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            model(x)
        events = sorted((e.start_ns(), e.nbytes()) for e in prof.profiler.kineto_results.events()
                        if e.name() == "[memory]")
        if not events:
            raise RuntimeError("no memory events")
        cur = peak = 0
        for _, nbytes in events:
            cur += nbytes
            peak = max(peak, cur)
        return (weights + peak) / 2 ** 20, "profiler"
    except Exception:
        sizes = []
        hooks = [m.register_forward_hook(lambda m, i, o: sizes.append(o.numel() * o.element_size()))
                 for m in model.modules() if len(list(m.children())) == 0]
        model(x)
        for h in hooks:
            h.remove()
        act = max(a + b for a, b in zip(sizes, sizes[1:])) if len(sizes) > 1 else sum(sizes)
        return (weights + act) / 2 ** 20, "estimate"


def measure_model(model, input_size, threads=1, runs=50):
    """All hardware metrics for one PyTorch model on the CPU (batch size 1)."""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = torch.randn(1, *input_size)
    lat, p95 = latency_ms(model, x, runs=runs)
    mem, method = peak_memory_mb(model, x)
    torch.set_num_threads(old)
    return {"params": count_params(model), "macs_M": count_macs(model, input_size) / 1e6,
            "size_mb": storage_size_mb(model), "latency_ms": lat, "latency_p95_ms": p95,
            "peak_mem_mb": mem, "mem_method": method}
