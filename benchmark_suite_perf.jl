# =============================================================
# bench_julia.jl — wall time + memory for SynapseEngine
# Usage:
#   julia -e 'include("Tensor.jl"); include("bench_julia.jl")'
# =============================================================

using Printf

println("\n", "="^64)
println("  SynapseEngine — Wall Time & Memory Benchmark")
println("="^64)

# ── helpers ──────────────────────────────────────────────────
function bench(f; warmup=50, runs=50)
    for _ in 1:warmup; f(); end
    GC.gc()
    stats = [@timed(f()) for _ in 1:runs]
    times  = [s.time  for s in stats]
    allocs = [s.bytes for s in stats]
    return (
        time_mean  = sum(times)  / runs,
        time_min   = minimum(times),
        alloc_mean = sum(allocs) / runs,
    )
end

fmt_t(s) = s < 1e-6 ? @sprintf("%.1f ns", s*1e9)  :
           s < 1e-3 ? @sprintf("%.2f μs", s*1e6)  :
           s < 1.0  ? @sprintf("%.2f ms", s*1e3)  :
                      @sprintf("%.3f s",  s)
fmt_m(b) = b < 1024   ? @sprintf("%d B",    b)        :
           b < 1024^2 ? @sprintf("%.1f KB", b/1024)   :
                        @sprintf("%.2f MB", b/1024^2)

function row(label, r)
    @printf("  %-42s  %8s  %8s  %10s\n",
        label, fmt_t(r.time_mean), fmt_t(r.time_min), fmt_m(r.alloc_mean))
end

println()
@printf("  %-42s  %8s  %8s  %10s\n", "Task", "mean", "min", "alloc/call")
println("  ", "-"^66)

T = Float32
Random.seed!(42)

# ── A. Forward pass only — Model inference, no backward ──────
println("\n  [A] Forward pass — Model inference (no grad)")

for (label, in_d, out_d) in [
        ("tiny   1→1",       1,    1),
        ("small  32→32",     32,   32),
        ("medium 256→256",   256,  256),
        ("large  1024→1024", 1024, 1024),
    ]
    local l = Layer(T, in_d, out_d, xavier_init, ReLU)
    local m = Model([l], inp -> l(inp))
    local x = randn(T, in_d)
    r = bench(() -> m(x))
    row("fwd $label", r)
end

# ── B. Forward + backward — Model training step ───────────────
println("\n  [B] Forward + backward — Model (width=256)")

for depth in [1, 2, 5, 10, 20]
    local layers = [Layer(T, 256, 256, xavier_init, ReLU) for _ in 1:depth]
    local m = Model(layers, function(inp)
        h = inp
        for l in layers; h = l(h); end
        return h
    end)
    local x = randn(T, 256)
    local y = Tensor(randn(T, 256))
    function fwd_bwd_model()
        pred = m(x)
        loss = mse_loss(pred, y, layers[1]._cache)
        backprop!(m, loss)
    end
    r = bench(fwd_bwd_model)
    row("fwd+bwd depth=$depth", r)
end

# ── C. Full train step — Model + weight update ────────────────
println("\n  [C] Full train step (forward + backward + weight update)")

for (in_d, hid, out_d) in [(4,16,1),(32,128,1),(256,512,1)]
    local l1 = Layer(T, in_d,  hid,  xavier_init, ReLU)
    local l2 = Layer(T, hid,   out_d, xavier_init, Linear)
    local m  = Model([l1, l2], inp -> l2(l1(inp)))
    local x  = randn(T, in_d)
    local y  = Tensor(T[1.0])
    function train_step_model()
        pred = m(x)
        loss = mse_loss(pred, y, l1._cache)
        backprop!(m, loss)
        for layer in m.layers
            layer.w.grad !== nothing && (layer.w.data .-= T(0.01) .* layer.w.grad)
            layer.b.grad !== nothing && (layer.b.data .-= T(0.01) .* layer.b.grad)
        end
    end
    r = bench(train_step_model)
    row("train step $(in_d)→$(hid)→$(out_d)", r)
end

# ── D. Memory breakdown ───────────────────────────────────────
println("\n  [D] Memory breakdown — where does allocation go?")

let l   = Layer(T, 128, 128, xavier_init, ReLU),
    m   = Model([l], inp -> l(inp)),
    x   = randn(T, 128),
    y   = Tensor(randn(T, 128))

    r_fwd  = bench(() -> m(x))
    r_bwd  = bench(() -> begin
        pred = m(x)
        loss = mse_loss(pred, y, l._cache)
        backprop!(m, loss)
    end)
    r_topo = bench(() -> begin
        pred = m(x)
        loss = mse_loss(pred, y, l._cache)
        build_topo(loss)
    end)

    row("forward pass only (128→128)",  r_fwd)
    row("forward + backprop (128→128)", r_bwd)
    row("build_topo only",              r_topo)
end

# ── E. Scalability — Model depth vs time ─────────────────────
println("\n  [E] Scalability — Model depth vs time")
@printf("  %-12s  %-8s  %-10s  %-10s\n", "depth", "nodes", "fwd+bwd", "ns/node")

for depth in [1, 5, 10, 20, 50, 100]
    local layers = [Layer(T, 32, 32, xavier_init, ReLU) for _ in 1:depth]
    local m = Model(layers, function(inp)
        h = inp
        for l in layers; h = l(h); end
        return h
    end)
    local x = randn(T, 32)
    local y = Tensor(randn(T, 32))

    # count nodes from one forward pass
    pred_sample = m(x)
    loss_sample = mse_loss(pred_sample, y, layers[1]._cache)
    nodes = length(build_topo(loss_sample))

    r = bench(() -> begin
        pred = m(x)
        loss = mse_loss(pred, y, layers[1]._cache)
        backprop!(m, loss)
    end)
    ns_per_node = r.time_mean * 1e9 / nodes
    @printf("  %-12d  %-8d  %-10s  %-10.1f\n",
        depth, nodes, fmt_t(r.time_mean), ns_per_node)
end

# ── F. Experimental path — raw layers, no cache ──────────────
# This is intentionally uncached — reflects experimentation use case
println("\n  [F] Raw layer calls (experimental, no Model, no cache)")

for (label, in_d, out_d) in [
        ("tiny   1→1",     1,   1),
        ("small  32→32",   32,  32),
        ("medium 256→256", 256, 256),
    ]
    local l = Layer(T, in_d, out_d, xavier_init, ReLU)
    local x = Tensor(randn(T, in_d))
    r = bench(() -> l(x))
    row("raw fwd $label", r)
end

println("\n", "="^64, "\n")
println("Copy the numbers above into the comparison table.")
println("Run bench_pytorch.py next and paste both sets together.\n")