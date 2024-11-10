module Layers

export DenseLayer, ReLU

include("CoreUtils.jl")
using .CoreUtils: Tensor, discreteSummation

mutable struct Neuron
    name::String
    weight::Vector{Tensor}
    bias::Tensor
    intake::Vector{Tensor}
    state::Tensor
    out::Tensor

    function Neuron(name::String, input_size::Vector{Int}, output_size::Vector{Int})
        intake = [zeroTensor(input_size)] # modify this so that intake and state are made common for the layer and not individual neurons
        state = zeroTensor(input_size)
        weight = []
        bias = zeroTensor(output_size) 
        out = zeroTensor(output_size) 
        neuron = new(name, weight, bias, intake, state, out)
        return neuron
    end
end

mutable struct DenseLayer
    layer::Vector{Neuron}
    input_size::Vector{Int}
    output_size::Vector{Int}

    function DenseLayer(name::String, units::Int, input_size::Vector{Int})
        layer =  Vector{Neuron}()
        for x in 1:units 
            neuron = Neuron(name, input_size, [1, 1, units])
            push!(layer, neuron)
        end
        return new(layer, input_size, [1,units])
    end
end

function ReLU(inp::Tensor)
    for (i, x) in enumerate(inp.data)
        if (x<0)
            inp.data[i]=0
        end
    end
    return inp
end



function flatten(inp::Tensor)
    inp.shape = [1,1,prod(inp.shape)]
    return inp
end

end

