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
        neuron =  new(name, zeroTensor(intake[0].shape), zeroTensor(intake[0].shape), 0, intake, state, zeroTensor(intake[0].shape))
        return structVector(units, neuron)
    end

end
