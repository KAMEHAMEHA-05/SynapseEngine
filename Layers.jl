module Layers

export DenseLayer, ReLU

include("CoreUtils.jl")
using .CoreUtils: Tensor, discreteSummation

mutable struct DenseLayer
    name::String
    weight::Tensor
    bias::Tensor
    intake::Vector{Tensor}
    state::Tensor
    out::Tensor

    function DenseLayer(name::String, units::Int)
        intake = Tensor[]
        state = discreteSummation(intake)
        neuron =  new(name, [], [], [], intake, state, [])
        return fill(neuron, 1, units)
    end

end

function ReLU(inp::Tensor)
    for (i, x) in enumerate(inp.data)
        if (x<0)
            inp[i]=0
        end
    end
    return inp
end

end

