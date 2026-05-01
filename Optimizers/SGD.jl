# Optimizers/SGD.jl

using ..Core

abstract type Optimizer end

mutable struct SGD <: Optimizer
    lr::Float32
    momentum::Float32
    velocity::Dict{UInt, AbstractArray}  
end

SGD(; lr=0.01f0, momentum=0.0f0) = SGD(lr, momentum, Dict())

function step!(opt::SGD, params::Vector{Tensor})
    for p in params
        p.grad === nothing && continue
        id = objectid(p)
        if opt.momentum > 0
            if !haskey(opt.velocity, id)
                opt.velocity[id] = similar(p.grad)
                fill!(opt.velocity[id], 0)
            end
            opt.velocity[id] .= opt.momentum .* opt.velocity[id] .+ p.grad
            p.data .-= opt.lr .* opt.velocity[id]
        else
            p.data .-= opt.lr .* p.grad
        end
    end
end

