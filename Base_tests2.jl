include("Exceptions.jl")
using .Exceptions: raise_dimension_mismatch, raise_indexoutofbounds
using Revise
import Base: +, *, /, -, size, reshape

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
    ndims = maximum([length(shape), 0])
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
   
function discreteSummation(tensor_vector::Vector{Tensor})
    shape = tensor_vector[1].shape
    summed = zeroTensor(shape)
    for x in tensor_vector
        summed+=x
    end
    return summed
end

mutable struct Neuron
    name::String
    weight::Tensor
    bias::Tensor
    intake::Vector{Tensor}
    state::Tensor
    out::Tensor

    function Neuron(name::String)
        intake = Tensor[]
        state = discreteSummation(intake)
        return new(name, [], [], [], intake, state, [])
    end

end

mutable struct DenseLayer
    layer::Vector{Neuron}

    function DenseLayer(name::String, units::Int)
        layer =  [Neuron(name) for x in 1:units]
        return new(layer)
    end
end

mutable struct Model 
    layers::Vector{DenseLayer}

    function Model(layers::Vector{DenseLayer} = DenseLayer[])
        return new(layers)
    end
end



#-------------------------------------<<< TESTING STUFF >>>----------------------------------------------------------------------------------
arr = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
arr1 = [[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0], [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0], [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]]
shape = [2,2,2]
shape2 = [2,2,2]
tensor = Tensor(arr, shape)
index = [1, 1, 1]
println(valueAt(tensor, index))
newtensor = Tensor(arr, shape) + Tensor(arr, shape2)
println(newtensor)
ntensor2 = discreteSummation([tensor, tensor, tensor])
println(ntensor2)



