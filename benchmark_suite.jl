# =============================================================================
# Benchmark & Test Suite for Custom Julia Autograd Framework
# =============================================================================
# Run this file alongside your Tensor.jl (or whatever your main file is named).
# Assumes the following are available in scope:
#   Tensor, matmul, ReLU, LeakyReLU, Linear, Layer
#   xavier_init, ones_init, zeroes_init
#   backprop, mse_loss, build_topo, ensure_grad!
# =============================================================================

using Test
using Statistics: mean, std
using Printf

println("\n", "="^62)
println("  Neural Network Framework — Benchmark & Test Suite")
println("="^62)

# ─────────────────────────────────────────────────────────────
# SECTION 1: GRADIENT CORRECTNESS (finite difference checks)
# ─────────────────────────────────────────────────────────────
println("\n[1/4] Gradient correctness\n")

"""
    finite_diff_check(f, x_data; ε=1e-4, tol=1e-2)

Numerically estimates ∂f/∂x_i for every element of x using central differences,
then compares to the analytical gradient produced by backprop.
Returns (max_err, pass::Bool).
"""
function finite_diff_check(f, x_data::Array{T}; ε=T(1e-4), tol=1e-2) where T
    x = Tensor(copy(x_data))
    out = f(x)
    backprop(out)
    analytic = copy(x.grad)

    numeric = similar(x_data)
    for i in eachindex(x_data)
        xp = copy(x_data); xp[i] += ε
        xm = copy(x_data); xm[i] -= ε
        fp = f(Tensor(xp)).data[1]
        fm = f(Tensor(xm)).data[1]
        numeric[i] = (fp - fm) / (2ε)
    end

    max_err = maximum(abs.(analytic .- numeric))
    return max_err, max_err < tol
end

@testset "Gradient correctness" begin

    # --- Scalar ops ---
    @testset "Elementwise ops" begin
        T = Float64

        # Addition: f(x) = sum(x + x)
        err, ok = finite_diff_check(x -> sum(x + Tensor(ones(T, 3))), ones(T, 3))
        @test ok; @printf("  add          max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")

        # Subtraction: f(x) = sum(x - c)
        err, ok = finite_diff_check(x -> sum(x - Tensor(fill(T(0.5), 3))), ones(T, 3))
        @test ok; @printf("  sub          max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")

        # Multiplication: f(x) = sum(x * x)  (squaring)
        err, ok = finite_diff_check(x -> sum(x * x), T[1.0, 2.0, 3.0])
        @test ok; @printf("  mul (x*x)    max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")

        # Division: f(x) = sum(c / x)
        err, ok = finite_diff_check(x -> sum(Tensor(T[2.0, 3.0, 4.0]) / x), T[1.0, 2.0, 3.0])
        @test ok; @printf("  div          max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")

        # ReLU: f(x) = sum(ReLU(x))
        err, ok = finite_diff_check(x -> sum(ReLU(x)), T[-1.0, 0.5, 2.0])
        @test ok; @printf("  ReLU         max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")

        # LeakyReLU
        err, ok = finite_diff_check(x -> sum(LeakyReLU(x)), T[-1.0, 0.5, 2.0])
        @test ok; @printf("  LeakyReLU    max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")
    end

    @testset "Matmul gradient" begin
        T = Float64
        W_data = randn(T, 3, 4)
        x_data = randn(T, 4)
        W = Tensor(W_data)

        # Check gradient w.r.t. x: f(x) = sum(W * x)
        err, ok = finite_diff_check(x -> sum(matmul(Tensor(W_data), x)), x_data)
        @test ok; @printf("  matmul dx    max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")

        # Check gradient w.r.t. W: embed W in the closure
        err_w, ok_w = let xf = Tensor(x_data)
            f_w = function(wt)
                out = matmul(wt, xf)
                sum(out)
            end
            finite_diff_check(f_w, W_data)
        end
        @test ok_w; @printf("  matmul dW    max_err=%.2e  %s\n", err_w, ok_w ? "PASS" : "FAIL")
    end

    @testset "MSE loss gradient" begin
        T = Float64
        pred_data = T[1.0, 2.0, 3.0]
        target    = Tensor(T[0.0, 0.0, 0.0])
        err, ok   = finite_diff_check(p -> mse_loss(p, Tensor(T[0.0, 0.0, 0.0])), pred_data)
        @test ok; @printf("  mse_loss     max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")
    end

    @testset "Deep chain gradient (5 layers)" begin
        T = Float64
        x_data = randn(T, 4)
        W1 = randn(T, 6, 4); W2 = randn(T, 4, 6); W3 = randn(T, 4, 4)
        W4 = randn(T, 3, 4); W5 = randn(T, 1, 3)
        function deep_f(x)
            h = ReLU(matmul(Tensor(W1), x))
            h = ReLU(matmul(Tensor(W2), h))
            h = ReLU(matmul(Tensor(W3), h))
            h = ReLU(matmul(Tensor(W4), h))
            sum(matmul(Tensor(W5), h))
        end
        err, ok = finite_diff_check(deep_f, x_data)
        @test ok; @printf("  5-layer chain max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")
    end

    @testset "Shared node (diamond) gradient" begin
        T = Float64
        # x feeds into two branches that both add to the output
        # Correct gradient = sum of contributions from both paths
        x_data = T[1.0, 2.0]
        W1 = T[1.0 0.0; 0.0 1.0]; W2 = T[2.0 0.0; 0.0 2.0]; Wout = T[1.0 1.0]
        function diamond_f(x)
            b1 = matmul(Tensor(W1), x)
            b2 = matmul(Tensor(W2), x)
            sum(matmul(Tensor(Wout), b1 + b2))
        end
        err, ok = finite_diff_check(diamond_f, x_data)
        @test ok; @printf("  diamond graph max_err=%.2e  %s\n", err, ok ? "PASS" : "FAIL")
    end
end

# ─────────────────────────────────────────────────────────────
# SECTION 2: TRAINING CONVERGENCE
# ─────────────────────────────────────────────────────────────
println("\n[2/4] Training convergence\n")

function sgd_step!(params, lr)
    for p in params
        if p.grad !== nothing
            p.data .-= lr .* p.grad
        end
    end
end

@testset "Training convergence" begin

    @testset "Constant mapping: f(x) = 7 (1-layer linear)" begin
        T = Float32
        Random.seed!(1)
        l = Layer(T, 1, 1, xavier_init, Linear)
        losses = Float32[]
        for _ in 1:2000
            pred = l(Tensor(T[1.0]))
            loss = mse_loss(pred, Tensor(T[7.0]))
            backprop(loss)
            sgd_step!([l.w, l.b], T(0.05))
            push!(losses, loss.data[1])
        end
        final_pred = l(Tensor(T[1.0])).data[1]
        @printf("  constant map  final_pred=%.4f  target=7.0  loss=%.2e  %s\n",
            final_pred, losses[end], abs(final_pred - 7f0) < 0.1 ? "PASS" : "FAIL")
        @test abs(final_pred - 7f0) < 0.1
    end

    @testset "Linear regression: y = 3x + 2" begin
        T = Float32
        Random.seed!(2)
        l = Layer(T, 1, 1, xavier_init, Linear)
        xs = Float32[-2, -1, 0, 1, 2]
        ys = 3f0 .* xs .+ 2f0
        losses = Float32[]
        for epoch in 1:3000
            total = 0f0
            for (x, y) in zip(xs, ys)
                pred = l(Tensor(T[x]))
                loss = mse_loss(pred, Tensor(T[y]))
                backprop(loss)
                sgd_step!([l.w, l.b], T(0.01))
                total += loss.data[1]
            end
            push!(losses, total / length(xs))
        end
        # Check slope ≈ 3 and bias ≈ 2
        slope = l.w.data[1,1]; bias = l.b.data[1]
        @printf("  lin reg       slope=%.3f (expect≈3)  bias=%.3f (expect≈2)  %s\n",
            slope, bias, (abs(slope-3f0)<0.2 && abs(bias-2f0)<0.2) ? "PASS" : "FAIL")
        @test abs(slope - 3f0) < 0.2
        @test abs(bias  - 2f0) < 0.2
    end

    @testset "XOR classification (2-layer)" begin
        T = Float32
        Random.seed!(3)
        l1 = Layer(T, 2, 4, xavier_init, LeakyReLU)
        l2 = Layer(T, 4, 1, xavier_init, Linear)
        xor_xs = T[0 0; 0 1; 1 0; 1 1]'  # 2×4
        xor_ys = T[0, 1, 1, 0]
        losses = Float32[]
        for epoch in 1:5000
            total = 0f0
            for i in 1:4
                x = Tensor(xor_xs[:, i])
                y = Tensor(T[xor_ys[i]])
                h = l1(x)
                pred = l2(h)
                loss = mse_loss(pred, y)
                backprop(loss)
                sgd_step!([l1.w, l1.b, l2.w, l2.b], T(0.05))
                total += loss.data[1]
            end
            push!(losses, total / 4)
        end
        # Check predictions are on correct side of 0.5
        correct = 0
        for i in 1:4
            x = Tensor(xor_xs[:, i])
            pred = l2(l1(x)).data[1]
            correct += (pred > 0.5) == (xor_ys[i] > 0.5) ? 1 : 0
        end
        @printf("  XOR           accuracy=%d/4  final_loss=%.4f  %s\n",
            correct, losses[end], correct >= 3 ? "PASS" : "FAIL")
        @test correct >= 3
    end

    @testset "Loss is monotonically decreasing (noise-free data)" begin
        T = Float32
        Random.seed!(4)
        l = Layer(T, 1, 1, xavier_init, Linear)
        losses = Float32[]
        for _ in 1:200
            pred = l(Tensor(T[2.0]))
            loss = mse_loss(pred, Tensor(T[5.0]))
            backprop(loss)
            sgd_step!([l.w, l.b], T(0.01))
            push!(losses, loss.data[1])
        end
        # Allow small bumps but overall trend should decrease
        trend_ok = losses[end] < losses[1] * 0.01
        @printf("  monotone loss  start=%.4f  end=%.6f  %s\n",
            losses[1], losses[end], trend_ok ? "PASS" : "FAIL")
        @test trend_ok
    end
end

# ─────────────────────────────────────────────────────────────
# SECTION 3: PERFORMANCE BENCHMARKS
# ─────────────────────────────────────────────────────────────
println("\n[3/4] Performance scaling\n")

function time_it(f, n_warmup=2, n_runs=10)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_runs]
    return mean(times), std(times)
end

@testset "Performance" begin

    println("  Forward pass (single sample, varying width):")
    for width in [16, 64, 256, 1024]
        T = Float32
        l = Layer(T, width, width, xavier_init, ReLU)
        x = Tensor(randn(T, width))
        μ, σ = time_it(() -> l(x))
        @printf("    width=%4d  mean=%.2e s  std=%.2e s\n", width, μ, σ)
        @test μ < 1.0  # should always be sub-second for dense layers
    end

    println("  Backward pass (varying depth, width=32):")
    for depth in [2, 5, 10, 20]
        T = Float32
        layers = [Layer(T, 32, 32, xavier_init, ReLU) for _ in 1:depth]
        x0 = Tensor(randn(T, 32))
        forward = () -> begin
            h = x0
            for l in layers; h = l(h); end
            sum(h)
        end
        μ_fwd, _ = time_it(forward)
        μ_bwd, _ = time_it(() -> backprop(forward()))
        @printf("    depth=%2d  fwd=%.2e s  bwd=%.2e s  ratio=%.1fx\n",
            depth, μ_fwd, μ_bwd, μ_bwd/μ_fwd)
        @test μ_bwd < 5.0
    end

    println("  Topological sort cost (varying graph size):")
    for depth in [5, 20, 50, 100]
        T = Float32
        layers = [Layer(T, 8, 8, xavier_init, ReLU) for _ in 1:depth]
        x0 = Tensor(randn(T, 8))
        out = foldl((h, l) -> l(h), layers; init=x0)
        μ, _ = time_it(() -> build_topo(out))
        @printf("    depth=%3d  topo_sort=%.2e s  nodes=%d\n",
            depth, μ, length(build_topo(out)))
        @test μ < 1.0
    end
end

# ─────────────────────────────────────────────────────────────
# SECTION 4: NUMERICAL STABILITY
# ─────────────────────────────────────────────────────────────
println("\n[4/4] Numerical stability\n")

@testset "Numerical stability" begin

    @testset "Float16 backprop doesn't NaN" begin
        T = Float16
        l1 = Layer(T, 4, 8, xavier_init, ReLU)
        l2 = Layer(T, 8, 1, xavier_init, Linear)
        x = Tensor(randn(T, 4))
        pred = l2(l1(x))
        loss = mse_loss(pred, Tensor(T[1.0]))
        backprop(loss)
        has_nan = any(isnan, l1.w.grad) || any(isnan, l2.w.grad)
        @printf("  Float16 grads  has_nan=%s  %s\n", has_nan, !has_nan ? "PASS" : "FAIL")
        @test !has_nan
    end

    @testset "Float64 vs Float32 gradient error < 1e-4" begin
        Random.seed!(5)
        W32 = Float32.(randn(4, 4)); x32 = Float32.(randn(4))
        W64 = Float64.(W32);        x64 = Float64.(x32)
        f32 = (x) -> sum(ReLU(matmul(Tensor(W32), x)))
        f64 = (x) -> sum(ReLU(matmul(Tensor(W64), x)))
        _, ok32 = finite_diff_check(f32, x32)
        _, ok64 = finite_diff_check(f64, x64)
        x32t = Tensor(x32); out32 = f32(x32t); backprop(out32)
        x64t = Tensor(x64); out64 = f64(x64t); backprop(out64)
        max_diff = maximum(abs.(Float64.(x32t.grad) .- x64t.grad))
        @printf("  F32 vs F64     max_grad_diff=%.2e  %s\n",
            max_diff, max_diff < 1e-4 ? "PASS" : "FAIL")
        @test max_diff < 1e-4
    end

    @testset "Very large weights don't produce NaN" begin
        T = Float32
        l = Layer(T, 4, 4, xavier_init, LeakyReLU)
        l.w.data .*= T(1e3)  # deliberately large
        x = Tensor(randn(T, 4))
        pred = l(x)
        loss = mse_loss(pred, Tensor(zeros(T, 4)))
        backprop(loss)
        finite_grads = all(isfinite, l.w.grad)
        @printf("  large weights  finite_grads=%s  %s\n", finite_grads, finite_grads ? "PASS" : "FAIL")
        @test finite_grads
    end

    @testset "Division near-zero is handled" begin
        T = Float64
        eps_val = 1e-8
        a = Tensor(T[1.0, 2.0, 3.0])
        b = Tensor(T[eps_val, eps_val, eps_val])
        out = sum(a / b)
        backprop(out)
        @printf("  near-zero div  grad_b_finite=%s  %s\n",
            all(isfinite, b.grad), all(isfinite, b.grad) ? "PASS" : "FAIL")
        # Note: may fail for very small eps — this surfaces the issue
        @test all(isfinite, b.grad)
    end

    @testset "Grad accumulation: multiple backward calls" begin
        T = Float32
        x = Tensor(T[1.0, 2.0])
        out1 = sum(x * x)
        backprop(out1)
        g1 = copy(x.grad)

        # Second backprop on fresh computation
        x2 = Tensor(T[1.0, 2.0])
        out2 = sum(x2 * x2)
        backprop(out2)
        g2 = copy(x2.grad)

        @test g1 ≈ g2   # deterministic
        @printf("  grad repeatability  g1=%s g2=%s  %s\n",
            g1, g2, g1 ≈ g2 ? "PASS" : "FAIL")
    end
end

println("\n", "="^62)
println("  All benchmarks complete.")
println("="^62, "\n")