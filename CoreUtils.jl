module CoreUtils

export Tensor, zeroTensor, prod, reshape, valueAt
export +, *, -, /, ^
export discreteSummation, unitmatmul, squeeze, unsqueeze
export transpose_2d, expand_dims, batch_flatten
export slice, concatenate, mean, sum

using Revise
import Base: +, *, /, -, ^, size, reshape, sum, mean

include("Exceptions.jl")
using .Exceptions: raise_dimension_mismatch, raise_indexoutofbounds

"""
    Tensor

A mutable struct representing a multi-dimensional tensor.
Stores dimensions, shape, and data in a flat vector.
"""
mutable struct Tensor
    ndims::Int
    shape::Vector{Int}
    data::Vector{Float64}

    function Tensor(data::Vector{Float64}, shape::Vector{Int})
        # Validate that data length matches shape
        if prod(shape) != length(data)
            throw(DimensionMismatch("Data length doesn't match specified shape"))
        end
        ndims = length(shape)
        new(ndims, shape, data)
    end

    # Convenience constructor for empty tensor
    Tensor() = new(0, Int[], Float64[])
end

# Helper functions
"""
    prod(array::Vector{Int})

Compute the product of all elements in an integer array.
"""
function prod(array::Vector{Int})
    isempty(array) && return 1
    return reduce(*, array)
end

"""
    calculate_flat_index(shape::Vector{Int}, index::Vector{Int})

Calculate the flat array index from multi-dimensional indices.
"""
function calculate_flat_index(shape::Vector{Int}, index::Vector{Int})
    if length(shape) != length(index)
        throw(DimensionMismatch("Index dimension doesn't match tensor dimension"))
    end
    
    flat_index = 1
    stride = 1
    for i in 1:length(shape)
        if index[i] < 1 || index[i] > shape[i]
            throw(BoundsError("Index out of bounds"))
        end
        flat_index += (index[i] - 1) * stride
        stride *= shape[i]
    end
    return flat_index
end

# Core tensor operations
"""
    zeroTensor(shape::Vector{Int})

Create a tensor of given shape filled with zeros.
"""
function zeroTensor(shape::Vector{Int})
    data = zeros(prod(shape))
    return Tensor(data, shape)
end

"""
    valueAt(t::Tensor, index::Vector{Int})

Get value at specified multi-dimensional index.
"""
function valueAt(t::Tensor, index::Vector{Int})
    flat_index = calculate_flat_index(t.shape, index)
    return t.data[flat_index]
end

"""
    setValueAt!(t::Tensor, index::Vector{Int}, value::Float64)

Set value at specified multi-dimensional index.
"""
function setValueAt!(t::Tensor, index::Vector{Int}, value::Float64)
    flat_index = calculate_flat_index(t.shape, index)
    t.data[flat_index] = value
end

# Arithmetic operations
for (op, func) in [(:+, :+), (:-, :-), (:*, :*), (:/, :/)]
    @eval function $op(x::Tensor, y::Tensor)
        if x.shape != y.shape
            throw(DimensionMismatch("Tensor shapes must match"))
        end
        data = $func.(x.data, y.data)
        return Tensor(data, copy(x.shape))
    end
end

"""
    discreteSummation(tensor_vector::Vector{Tensor})

Sum a vector of tensors element-wise.
"""
function discreteSummation(tensor_vector::Vector{Tensor})
    isempty(tensor_vector) && throw(ArgumentError("Empty tensor vector"))
    return reduce(+, tensor_vector)
end

"""
    unitmatmul(x::Tensor, y::Tensor)

Matrix multiplication for x (1×1×n) and y (m×n×p).
Returns a 1×1×p tensor.
"""
function unitmatmul(x::Tensor, y::Tensor)
    # Validate x is 1×1×n
    if length(x.shape) != 3 || x.shape[1] != 1 || x.shape[2] != 1
        throw(DimensionMismatch("First tensor must be 1×1×n"))
    end
    
    # Validate y is m×n×p
    if length(y.shape) != 3
        throw(DimensionMismatch("Second tensor must be m×n×p"))
    end
    
    # Check matching dimensions
    if x.shape[3] != y.shape[2]
        throw(DimensionMismatch("Inner dimensions must match: $(x.shape[3]) ≠ $(y.shape[2])"))
    end
    
    out = Tensor()
    out.ndims = 3
    out.shape = [1, 1, y.shape[3]]
    out.data = Vector{Float64}(undef, y.shape[3])

    for i in axes(out.data, 1)
        column = Vector{Float64}(undef, y.shape[2])
        for j in axes(column, 1)
            column[j] = y.data[(j - 1) * y.shape[3] + i]
        end
        out.data[i] = sum(x.data .* column)
    end

    return out
end

# Shape operations
"""
    reshape(tensor::Tensor, new_shape::Vector{Int})

Reshape tensor to new dimensions while preserving data.
"""
function reshape(tensor::Tensor, new_shape::Vector{Int})
    if prod(tensor.shape) != prod(new_shape)
        throw(DimensionMismatch("New shape must have same number of elements"))
    end
    return Tensor(copy(tensor.data), new_shape)
end

"""
    squeeze(tensor::Tensor)

Remove dimensions of size 1 from tensor shape.
"""
function squeeze(tensor::Tensor)
    new_shape = filter(x -> x != 1, tensor.shape)
    isempty(new_shape) && return reshape(tensor, [1])
    return reshape(tensor, new_shape)
end

"""
    unsqueeze(tensor::Tensor, dim::Int)

Add dimension of size 1 at specified position.
"""
function unsqueeze(tensor::Tensor, dim::Int)
    if dim < 1 || dim > length(tensor.shape) + 1
        throw(ArgumentError("Invalid dimension"))
    end
    new_shape = copy(tensor.shape)
    insert!(new_shape, dim, 1)
    return reshape(tensor, new_shape)
end

# New functionality
"""
    slice(tensor::Tensor, dim::Int, index::Int)

Extract a slice along specified dimension.
"""
function slice(tensor::Tensor, dim::Int, index::Int)
    if dim < 1 || dim > tensor.ndims
        throw(ArgumentError("Invalid dimension"))
    end
    if index < 1 || index > tensor.shape[dim]
        throw(BoundsError(tensor, [dim, index]))
    end
    
    new_shape = copy(tensor.shape)
    splice!(new_shape, dim)
    stride = prod(tensor.shape[1:dim-1])
    step = prod(tensor.shape[1:dim])
    
    indices = Vector{Int}()
    base_idx = (index - 1) * stride + 1
    for i in 0:prod(tensor.shape[dim+1:end])-1
        start_idx = base_idx + i * step
        for offset in 0:stride-1
            push!(indices, start_idx + offset)
        end
    end
    
    return Tensor(tensor.data[indices], new_shape)
end

"""
    concatenate(tensors::Vector{Tensor}, dim::Int)

Concatenate tensors along specified dimension.
"""
function concatenate(tensors::Vector{Tensor}, dim::Int)
    isempty(tensors) && throw(ArgumentError("Empty tensor vector"))
    
    base_shape = copy(tensors[1].shape)
    total_size = base_shape[dim]
    
    # Validate shapes
    for t in @view tensors[2:end]
        if length(t.shape) != length(base_shape)
            throw(DimensionMismatch("All tensors must have same number of dimensions"))
        end
        for (i, sz) in enumerate(base_shape)
            if i != dim && t.shape[i] != sz
                throw(DimensionMismatch("Incompatible shapes for concatenation"))
            end
        end
        total_size += t.shape[dim]
    end
    
    new_shape = copy(base_shape)
    new_shape[dim] = total_size
    result = zeroTensor(new_shape)
    
    offset = 1
    for t in tensors
        size_before = prod(new_shape[1:dim-1])
        size_after = prod(new_shape[dim+1:end])
        
        for i in axes(1:size_before, 1)
            for j in axes(1:t.shape[dim], 1)
                for k in axes(1:size_after, 1)
                    new_idx = (i-1)*total_size*size_after + (offset+j-2)*size_after + k
                    old_idx = (i-1)*t.shape[dim]*size_after + (j-1)*size_after + k
                    result.data[new_idx] = t.data[old_idx]
                end
            end
        end
        offset += t.shape[dim]
    end
    
    return result
end

"""
    sum(tensor::Tensor, dims::Vector{Int})

Compute sum along specified dimensions.
"""
function sum(tensor::Tensor, dims::Vector{Int})
    new_shape = copy(tensor.shape)
    new_shape[dims] .= 1
    
    result = zeroTensor(new_shape)
    
    # Use CartesianIndices for iteration
    for idx in CartesianIndices(tuple(tensor.shape...))
        result_coords = collect(Tuple(idx))
        result_coords[dims] .= 1
        result_idx = CartesianIndex(tuple(result_coords...))
        
        # Use linear indices for data access
        result.data[LinearIndices(tuple(new_shape...))[result_idx]] += 
            tensor.data[LinearIndices(tuple(tensor.shape...))[idx]]
    end
    
    return result
end

# Statistical operations
"""
    mean(tensor::Tensor, dims::Vector{Int})

Compute mean along specified dimensions.
"""
function mean(tensor::Tensor, dims::Vector{Int})
    sum_result = sum(tensor, dims)
    n = prod([tensor.shape[d] for d in dims])
    return Tensor(sum_result.data ./ n, sum_result.shape)
end

"""
    sum(tensor::Tensor, dims::Vector{Int})

Compute sum along specified dimensions.
"""
function sum(tensor::Tensor, dims::Vector{Int})
    new_shape = copy(tensor.shape)
    new_shape[dims] .= 1
    
    result = zeroTensor(new_shape)
    
    # Use CartesianIndices for iteration
    for idx in CartesianIndices(tuple(tensor.shape...))
        result_coords = collect(Tuple(idx))
        result_coords[dims] .= 1
        result_idx = CartesianIndex(tuple(result_coords...))
        
        # Use linear indices for data access
        result.data[LinearIndices(tuple(new_shape...))[result_idx]] += 
            tensor.data[LinearIndices(tuple(tensor.shape...))[idx]]
    end
    
    return result
end

# Helper function for sum
function cart_to_indices(flat_idx::Int, shape::Vector{Int})
    indices = Int[]
    remaining = flat_idx - 1
    stride = 1
    for dim in shape
        push!(indices, (remaining ÷ stride) % dim + 1)
        remaining = remaining % (stride * dim)
        stride *= dim
    end
    return indices
end

end # module