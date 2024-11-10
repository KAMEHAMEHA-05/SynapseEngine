module CoreUtils

export Tensor, zeroTensor, prod, reshape, valueAt, +, *, -, /, discreteSummation, unitmatmul

include("Exceptions.jl")
using .Exceptions: raise_dimension_mismatch, raise_indexoutofbounds
using Revise
import Base: +, *, /, -, ^, size, reshape

mutable struct Tensor
    ndims::Int
    shape::Vector{Int}
    data::Vector{Float64}

    function Tensor(data::Vector{Float64}, shape::Vector{Int})
        ndims = maximum([length(shape), 0])
        return new(ndims, shape, data)
    end
end

function zeroTensor(shape::Vector{Int})
    data_units = 1
    for x in shape
        data_units*=x
    end
    data = zeros(data_units)
    return Tensor(data, shape)
end

function prod(array::Vector{Int})
    prod = 1
    for x in array
        prod*=x
    end
    return prod
end

function reshape(t::Tensor, shape::Vector{Int})
    t.shape = shape 
    t.ndims = length(shape)
end

function valueAt(t::Tensor, index::Vector{Int})
    raise_indexoutofbounds(t.shape[end:-1:1], index)
    
    flat_index = 1
    for (ind, factor) in enumerate(index)
        factor-=1
        jump=1
        for x in t.shape[1:ind-1]
            jump*=x
        end
        flat_index+=factor*jump
    end
    raise_indexoutofbounds(prod(t.shape), flat_index)
    return t.data[flat_index]
end

function +(x::Tensor, y::Tensor)
    raise_dimension_mismatch(x.shape, y.shape)

    data = Float64[]
    for (x_elt, y_elt) in zip(x.data, y.data)
        data = push!(data, x_elt+y_elt)
    end
    return Tensor(data, x.shape)
end

function *(x::Tensor, y::Tensor)
    raise_dimension_mismatch(x.shape, y.shape)

    data = Float64[]
    for (x_elt, y_elt) in zip(x.data, y.data)
        data = push!(data, x_elt*y_elt)
    end
    return Tensor(data, x.shape)
end

function -(x::Tensor, y::Tensor)
    raise_dimension_mismatch(x.shape, y.shape)

    data = Float64[]
    for (x_elt, y_elt) in zip(x.data, y.data)
        data = push!(data, x_elt-y_elt)
    end
    return Tensor(data, x.shape)
end

function /(x::Tensor, y::Tensor)
    raise_dimension_mismatch(x.shape, y.shape)

    data = Float64[]
    for (x_elt, y_elt) in zip(x.data, y.data)
        data = push!(data, x_elt/y_elt)
    end
    return Tensor(data, x.shape)
end

function discreteSummation(tensor_vector::Vector{Tensor})
    shape = tensor_vector[1].shape
    summed = zeroTensor(shape)
    for x in tensor_vector
        summed+=x
    end
    return summed
end

end

function unitmatmul(x::Tensor, y::Tensor)
    if length(x.shape) != 3 || length(y.shape) != 3
        raise_dimension_mismatch(x.shape, y.shape)
    end
    if x.shape[3] != y.shape[2]
        raise_dimension_mismatch(x.shape, y.shape)
    end
    out = Tensor()
    out.ndims = 3
    out.shape = [1, 1, y.shape[3]]

    for i in 1:y.shape[3]
        column = [y.data[(j - 1) * y.shape[3] + i] for j in 1:y.shape[2]]
        
        sum = 0.0
        for (a, b) in zip(x.data, column)
            sum += a * b
        end
        push!(out.data, sum)
    end

    return out
end
