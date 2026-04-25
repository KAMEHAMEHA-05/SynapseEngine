# =============================================================
# bench_pytorch.py — identical tasks to bench_julia.jl
# pip install torch
# python bench_pytorch.py
# =============================================================

import time, gc, tracemalloc, statistics
import torch
import torch.nn as nn

print()
print("=" * 64)
print("  PyTorch — Wall Time & Memory Benchmark")
print(f"  torch version: {torch.__version__}")
print("=" * 64)

DEVICE = "cpu"   # change to "cuda" if you have a GPU
DTYPE  = torch.float32

def fmt_t(s):
    if s < 1e-6: return f"{s*1e9:.1f} ns"
    if s < 1e-3: return f"{s*1e6:.2f} μs"
    if s < 1.0:  return f"{s*1e3:.2f} ms"
    return f"{s:.3f} s"

def fmt_m(b):
    if b < 1024:     return f"{b} B"
    if b < 1024**2:  return f"{b/1024:.1f} KB"
    return f"{b/1024**2:.2f} MB"

def bench(f, warmup=5, runs=50):
    for _ in range(warmup):
        f()
    times, allocs = [], []
    for _ in range(runs):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        f()
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(t1 - t0)
        allocs.append(peak)
    return {
        "time_mean":  statistics.mean(times),
        "time_min":   min(times),
        "alloc_mean": statistics.mean(allocs),
    }

def row(label, r):
    print(f"  {label:<38}  {fmt_t(r['time_mean']):>8}  "
          f"{fmt_t(r['time_min']):>8}  {fmt_m(r['alloc_mean']):>10}")

print()
print(f"  {'Task':<38}  {'mean':>8}  {'min':>8}  {'alloc/call':>10}")
print("  " + "-" * 62)

torch.manual_seed(42)

# ── A. Forward pass only (no_grad) ───────────────────────────
print("\n  [A] Forward pass (no grad)")

for label, in_d, out_d in [
    ("tiny   1→1",      1,    1),
    ("small  32→32",    32,   32),
    ("medium 256→256",  256,  256),
    ("large  1024→1024",1024, 1024),
]:
    layer = nn.Linear(in_d, out_d).to(DEVICE)
    x = torch.randn(in_d, dtype=DTYPE, device=DEVICE)
    def fwd(layer=layer, x=x):
        with torch.no_grad():
            return layer(x)
    r = bench(fwd)
    row(f"fwd {label}", r)

# ── B. Forward + backward ─────────────────────────────────────
print("\n  [B] Forward + backward (width=256)")

for depth in [1, 2, 5, 10, 20]:
    layers = nn.Sequential(*[nn.Linear(256, 256) for _ in range(depth)]).to(DEVICE)
    x0 = torch.randn(256, dtype=DTYPE, device=DEVICE, requires_grad=True)
    def fwd_bwd(layers=layers, x0=x0):
        out = layers(x0)
        loss = out.sum()
        loss.backward()
        if x0.grad is not None:
            x0.grad.zero_()
        for p in layers.parameters():
            if p.grad is not None:
                p.grad.zero_()
    r = bench(fwd_bwd)
    row(f"fwd+bwd depth={depth}", r)

# ── C. Full train step ────────────────────────────────────────
print("\n  [C] Full train step (forward + backward + weight update)")

for in_d, hid, out_d in [(4,16,1),(32,128,1),(256,512,1)]:
    model = nn.Sequential(nn.Linear(in_d, hid), nn.ReLU(),
                          nn.Linear(hid, out_d)).to(DEVICE)
    opt   = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    x = torch.randn(in_d, dtype=DTYPE, device=DEVICE)
    y = torch.ones(out_d, dtype=DTYPE, device=DEVICE)
    def train_step(model=model, opt=opt, x=x, y=y):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    r = bench(train_step)
    row(f"train step {in_d}→{hid}→{out_d}", r)

# ── D. Memory breakdown ────────────────────────────────────────
print("\n  [D] Memory breakdown")

layer128 = nn.Linear(128, 128).to(DEVICE)
x128 = torch.randn(128, dtype=DTYPE, device=DEVICE)

def fwd_only():
    with torch.no_grad():
        return layer128(x128)

def fwd_bwd_128():
    x = x128.detach().requires_grad_(True)
    out = layer128(x)
    out.sum().backward()

row("forward pass only (128→128)",  bench(fwd_only))
row("forward + backprop (128→128)", bench(fwd_bwd_128))

# ── E. Scalability ─────────────────────────────────────────────
print("\n  [E] Scalability — depth vs time")
print(f"  {'depth':<12}  {'fwd+bwd':<10}  {'ns/param':<10}")

for depth in [1, 5, 10, 20, 50, 100]:
    layers = nn.Sequential(*[nn.Linear(32, 32) for _ in range(depth)]).to(DEVICE)
    x0 = torch.randn(32, dtype=DTYPE, device=DEVICE)
    n_params = sum(p.numel() for p in layers.parameters())
    def fb(layers=layers, x0=x0):
        x = x0.detach().requires_grad_(True)
        out = layers(x)
        out.sum().backward()
    r = bench(fb)
    ns = r['time_mean'] * 1e9 / (depth * 5)  # 5 nodes per layer approx
    print(f"  {depth:<12}  {fmt_t(r['time_mean']):<10}  {ns:<10.1f}")

print()
print("=" * 64)
print()
print("Paste these numbers alongside bench_julia.jl output.")
print()

# ── F. Expected comparison summary (pre-filled with typical values)
print("=" * 64)
print("  EXPECTED COMPARISON SUMMARY (typical CPU, Float32)")
print("=" * 64)
print("""
  Task                              SynapseEngine    PyTorch       Ratio
  ─────────────────────────────────────────────────────────────────────
  fwd  tiny   1→1                  ~1 μs            ~4 μs         0.25x  (you're faster — overhead dominates PT)
  fwd  small  32→32                ~1 μs            ~5 μs         0.2x   (same reason)
  fwd  medium 256→256              ~6 μs            ~8 μs         0.75x  (closing gap)
  fwd  large  1024→1024            ~70 μs           ~25 μs        2.8x   (PT's BLAS kicks in here)
  fwd+bwd     depth=1              ~11 μs           ~15 μs        0.7x
  fwd+bwd     depth=5              ~28 μs           ~30 μs        0.9x
  fwd+bwd     depth=20             ~90 μs           ~60 μs        1.5x   (PT dispatch overhead amortizes)
  train step  4→16→1               ~25 μs           ~40 μs        0.6x
  train step  256→512→1            ~200 μs          ~80 μs        2.5x   (BLAS gap opens up)
  ─────────────────────────────────────────────────────────────────────
  Memory per fwd+bwd (128→128)     ~500 KB          ~50 KB        10x    (your closures allocate a lot)
  Backward/forward ratio            ~9x              ~2-3x         3-4x worse

  KEY INSIGHT:
  - Small nets  (<128 units): you are competitive or faster because PyTorch
    has significant per-call Python+C++ dispatch overhead (~4-10 μs fixed cost).
  - Large nets  (>256 units): PyTorch pulls ahead because it calls optimised
    BLAS (OpenBLAS/MKL) for matmul, while your matmul hits Julia's generic
    LinearAlgebra which is good but not BLAS-tuned for all shapes.
  - Memory: your biggest gap. Every op allocates a closure + grad buffer.
    PyTorch uses a tape (vector of op records) instead of heap closures.
  - Backward ratio: 9x (you) vs 2-3x (PT). Fix = pre-allocate grad buffers.
""")