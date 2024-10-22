include("exceptions.jl")
using .Exceptions
import Base: +, *, /, -, size, reshape, print

mutable struct Tensor
    ndims::Int
    shape::Vector{Int}
    data::Vector{Float64}

    function Tensor(data::Vector{Float64}, shape::Vector{Int})
        ndims = maximum([length(shape), 0])
        return new(ndims, shape, data)
    end
end

function valueAt(t::Tensor, index::Vector{Int})
    flat_index = 1
    for (ind, factor) in enumerate(index)
        jump=1
        for x in t.shape[1:ind-1]
            jump*=x
        end
        flat_index+=factor*jump
    end
    return t.data[flat_index]
end

function +(x::Tensor, y::Tensor)
    if(x.shape!=y.shape)
        raise_dimension_mismatch(x.shape, y.shape)
    end
    data = Float64[]
    for (x_elt, y_elt) in zip(x.data, y.data)
        data = push!(data, x_elt+y_elt)
    end
    return Tensor(data, x.shape)
end

function print(x::Tensor)
    print(x.data)
end