# Optimizers/Adam.jl

using ..Core

abstract type Optimizer end

mutable struct Adam <: Optimizer
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    t::Int
    m::Dict{UInt, AbstractArray}   
    v::Dict{UInt, AbstractArray}  
end

Adam(; lr=1f-3, beta1=0.9f0, beta2=0.999f0, eps=1f-8) =
    Adam(lr, beta1, beta2, eps, 0, Dict(), Dict())

function step!(opt::Adam, params::Vector{Tensor})
    opt.t += 1
    for p in params
        p.grad === nothing && continue
        id = objectid(p)
        if !haskey(opt.m, id)
            opt.m[id] = similar(p.grad); fill!(opt.m[id], 0)
            opt.v[id] = similar(p.grad); fill!(opt.v[id], 0)
        end
        g = p.grad
        opt.m[id] .= opt.beta1 .* opt.m[id] .+ (1 - opt.beta1) .* g
        opt.v[id] .= opt.beta2 .* opt.v[id] .+ (1 - opt.beta2) .* g.^2
        m̂ = opt.m[id] ./ (1 - opt.beta1^opt.t)
        v̂ = opt.v[id] ./ (1 - opt.beta2^opt.t)
        p.data .-= opt.lr .* m̂ ./ (sqrt.(v̂) .+ opt.eps)
    end
end
