# Initializers/Initializers.jl

module Initializers

function zeroes_init(::Type{T}, out_dim, in_dim) where T
    return zeros(T, out_dim, in_dim), zeros(T, out_dim)
end

function ones_init(::Type{T}, out_dim, in_dim) where T
    return ones(T, out_dim, in_dim), ones(T, out_dim)
end

function xavier_init(::Type{T}, out_dim, in_dim) where T
    scale = T(sqrt(2.0 / in_dim))
    W = randn(T, out_dim, in_dim) .* scale
    b = randn(T, out_dim) .* T(0.01)   
    return W, b
end

function identity_init(::Type{T}, out_dim, in_dim) where T
    @assert out_dim == in_dim "identity_init requires square weight matrix"
    return Matrix{T}(I, out_dim, in_dim), zeros(T, out_dim)
end

export zeroes_init, ones_init, xavier_init, identity_init

end