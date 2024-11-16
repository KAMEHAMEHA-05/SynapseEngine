function xavier_uniform(shape::Vector{Int})
    n_in = shape[2]  # Input features
    n_out = shape[3] # Output features
    limit = sqrt(6.0 / (n_in + n_out))
    data = (rand(prod(shape)) .- 0.5) .* (2 * limit)
    return Tensor(data, shape)
end

function he_normal(shape::Vector{Int})
    n_in = shape[2]  # Input features
    std = sqrt(2.0 / n_in)
    data = randn(prod(shape)) .* std
    return Tensor(data, shape)
end

function uniform(shape::Vector{Int}, a::Float64=-0.1, b::Float64=0.1)
    data = (rand(prod(shape)) .* (b - a)) .+ a
    return Tensor(data, shape)
end