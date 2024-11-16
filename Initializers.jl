# Initializers.jl
module Initializers

using Random
using LinearAlgebra
include("CoreUtils.jl")  # For Tensor struct and helper functions

export xavier_uniform, xavier_normal, he_normal, he_uniform, uniform_random
export normal_random, zeros_init, ones_init, orthogonal_init, constant_init

function xavier_uniform(shape::Vector{Int})
    n_in = shape[2]
    n_out = shape[3]
    limit = sqrt(6.0 / (n_in + n_out))
    data = (rand(prod(shape)) .- 0.5) .* (2 * limit)
    return Tensor(data, shape)
end

function xavier_normal(shape::Vector{Int})
    n_in = shape[2]
    n_out = shape[3]
    std = sqrt(2.0 / (n_in + n_out))
    data = randn(prod(shape)) .* std
    return Tensor(data, shape)
end

function he_normal(shape::Vector{Int})
    n_in = shape[2]
    std = sqrt(2.0 / n_in)
    data = randn(prod(shape)) .* std
    return Tensor(data, shape)
end

function he_uniform(shape::Vector{Int})
    n_in = shape[2]
    limit = sqrt(6.0 / n_in)
    data = (rand(prod(shape)) .- 0.5) .* (2 * limit)
    return Tensor(data, shape)
end

function uniform_random(shape::Vector{Int}, a::Float64=-0.1, b::Float64=0.1)
    data = (rand(prod(shape)) .* (b - a)) .+ a
    return Tensor(data, shape)
end

function normal_random(shape::Vector{Int}, mean::Float64=0.0, std::Float64=0.1)
    data = randn(prod(shape)) .* std .+ mean
    return Tensor(data, shape)
end

function zeros_init(shape::Vector{Int})
    return zeroTensor(shape)
end

function ones_init(shape::Vector{Int})
    return oneTensor(shape)
end

function orthogonal_init(shape::Vector{Int}, gain::Float64=1.0)
    flat_shape = prod(shape)
    a = randn(flat_shape, flat_shape)
    q, r = qr(a)
    q = Matrix(q)
    data = vec(q[:,1:shape[end]]) .* gain
    return Tensor(data, shape)
end

function constant_init(shape::Vector{Int}, value::Float64=0.0)
    data = fill(value, prod(shape))
    return Tensor(data, shape)
end

end