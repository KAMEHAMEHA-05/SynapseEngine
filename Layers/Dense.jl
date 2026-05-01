# Layers/Dense.jl

using ..Core
using LinearAlgebra

mutable struct Dense{Tw, Tb, F}
    w::Tw
    b::Tb
    activation::F
    trainable::Bool
    _cache::Union{Nothing, Dict{OpCacheKey, Tensor}}
end

function Dense(w::AbstractArray{T,2}, b::AbstractArray{T,1}, activation::F; trainable=true) where {T,F}
    w_tensor = Tensor(w)
    b_tensor = Tensor(b)
    w_tensor.grad = similar(w); fill!(w_tensor.grad, 0)
    b_tensor.grad = similar(b); fill!(b_tensor.grad, 0)
    return Dense(w_tensor, b_tensor, activation, trainable, nothing)
end

function Dense(::Type{T}, in_dim::Int, out_dim::Int, init::Function, activation::F; trainable=true) where {T,F}
    W, b = init(T, out_dim, in_dim)
    w_tensor = Tensor(W)
    b_tensor = Tensor(b)
    w_tensor.grad = similar(W); fill!(w_tensor.grad, 0)
    b_tensor.grad = similar(b); fill!(b_tensor.grad, 0)
    return Dense(w_tensor, b_tensor, activation, trainable, nothing)
end

function (layer::Dense)(x::Tensor)
    c  = layer._cache
    mm = matmul(layer.w, x, c)
    z  = Base.:+(mm, layer.b, c)
    return layer.activation(z; cache=c)  
end

function clip_grad!(layer::Dense, max_norm::Real)
    for param in [layer.w, layer.b]
        if param.grad !== nothing
            norm_ = norm(param.grad)
            if norm_ > max_norm
                param.grad .*= max_norm / norm_
            end
        end
    end
end