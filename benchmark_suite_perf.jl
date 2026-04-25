# =============================================================
# bench_julia.jl — wall time + memory for your autograd engine
# Run after including your framework:
#   julia -e 'include("Tensor.jl"); include("bench_julia.jl")'
# =============================================================

using Printf

println("\n", "="^64)
println("  SynapseEngine — Wall Time & Memory Benchmark")
println("="^64)

# ── helpers ──────────────────────────────────────────────────
function bench(f; warmup=50, runs=50)
    for _ in 1:warmup; f(); end
    stats = [@timed(f()) for _ in 1:runs]
    times  = [s.time   for s in stats]
    allocs = [s.bytes  for s in stats]
    return (
        time_mean  = sum(times)  / runs,
        time_min   = minimum(times),
        alloc_mean = sum(allocs) / runs,
        alloc_min  = minimum(allocs),
    )
end

fmt_t(s) = s < 1e-6 ? @sprintf("%.1f ns", s*1e9) :
           s < 1e-3 ? @sprintf("%.2f μs", s*1e6) :
           s < 1.0  ? @sprintf("%.2f ms", s*1e3) :
                      @sprintf("%.3f s",  s)
fmt_m(b) = b < 1024       ? @sprintf("%d B",    b) :
           b < 1024^2     ? @sprintf("%.1f KB", b/1024) :
                            @sprintf("%.2f MB", b/1024^2)

function row(label, r)
    @printf("  %-38s  %8s  %8s  %10s\n",
        label, fmt_t(r.time_mean), fmt_t(r.time_min), fmt_m(r.alloc_mean))
end

println()
@printf("  %-38s  %8s  %8s  %10s\n", "Task", "mean", "min", "alloc/call")
println("  ", "-"^62)

T = Float32
Random.seed!(42)

# ── 1. Forward pass only ─────────────────────────────────────
println("\n  [A] Forward pass (no grad)")

for (label, in_d, out_d) in [
        ("tiny   1→1",    1,    1),
        ("small  32→32",  32,   32),
        ("medium 256→256",256,  256),
        ("large  1024→1024",1024,1024),
    ]
    l = Layer(T, in_d, out_d, xavier_init, ReLU)
    x = Tensor(randn(T, in_d))
    r = bench(() -> l(x))
    row("fwd $label", r)
end

# ── 2. Forward + backward ────────────────────────────────────
println("\n  [B] Forward + backward (width=256)")

for depth in [1, 2, 5, 10, 20]
    layers = [Layer(T, 256, 256, xavier_init, ReLU) for _ in 1:depth]
    x0 = Tensor(randn(T, 256))
    function fwd_bwd()
        h = x0
        for l in layers; h = l(h); end
        loss = sum(h)
        backprop(loss)
    end
    r = bench(fwd_bwd)
    row("fwd+bwd depth=$depth", r)
end

# ── 3. Full train step (forward + backward + SGD update) ─────
println("\n  [C] Full train step (forward + backward + weight update)")

for (in_d, hid, out_d) in [(4,16,1),(32,128,1),(256,512,1)]
    l1 = Layer(T, in_d, hid,  xavier_init, ReLU)
    l2 = Layer(T, hid,  out_d, xavier_init, Linear)
    x  = Tensor(randn(T, in_d))
    y  = Tensor(T[1.0])
    params = [l1.w, l1.b, l2.w, l2.b]
    function train_step()
        pred = l2(l1(x))
        loss = mse_loss(pred, y)
        backprop(loss)
        for p in params
            p.grad !== nothing && (p.data .-= T(0.01) .* p.grad)
        end
    end
    r = bench(train_step)
    row("train step $(in_d)→$(hid)→$(out_d)", r)
end

# ── 4. Memory breakdown: what allocates most? ────────────────
println("\n  [D] Memory breakdown — where does allocation go?")

l = Layer(T, 128, 128, xavier_init, ReLU)
x = Tensor(randn(T, 128))
h = l(x)
loss = sum(h)

r_fwd   = bench(() -> l(x))
r_bwd   = bench(() -> backprop(sum(l(x))))
r_topo  = bench(() -> build_topo(loss))

row("forward pass only (128→128)",   r_fwd)
row("forward + backprop (128→128)",  r_bwd)
row("build_topo only",               r_topo)

# ── 5. Scalability table ─────────────────────────────────────
println("\n  [E] Scalability — nodes in graph vs time")
@printf("  %-12s  %-8s  %-10s  %-10s\n", "depth", "nodes", "fwd+bwd", "ns/node")
for depth in [1, 5, 10, 20, 50, 100]
    layers = [Layer(T, 32, 32, xavier_init, ReLU) for _ in 1:depth]
    x0 = Tensor(randn(T, 32))
    out = foldl((h,l)->l(h), layers; init=x0)
    nodes = length(build_topo(out))
    r = bench(() -> backprop(foldl((h,l)->l(h), layers; init=x0)))
    ns_per_node = r.time_mean * 1e9 / nodes
    @printf("  %-12d  %-8d  %-10s  %-10.1f\n",
        depth, nodes, fmt_t(r.time_mean), ns_per_node)
end

println("\n", "="^64, "\n")
println("Copy the numbers above into the comparison table.")
println("Run bench_pytorch.py next and paste both sets together.\n")