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

    function Tensor()
        return new(0, Vector{Int}(), Vector{Float64}())
    end
end

function zeroTensor(shape::Vector{Int})
    ndims = maximum([length(shape), 0])
    data_units = prod(shape)
    data = zeros(data_units)
    return Tensor(data, shape)
end

function oneTensor(shape::Vector{Int})
    ndims = maximum([length(shape), 0])
    data_units = prod(shape)
    data = ones(data_units)
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

function unitmatmul(x::Tensor, y::Tensor)
    # Check dimensions for matrix multiplication compatibility
    if length(x.shape) != 3 || length(y.shape) != 3
        println("shape under length")
        raise_dimension_mismatch(x.shape, y.shape)
    end
    if x.shape[3] != y.shape[2]
        println("shape mismatch")
        raise_dimension_mismatch(x.shape, y.shape)
    end
    out = Tensor()
    out.ndims = 3
    out.shape = [1, 1, y.shape[3]]

    for i in 1:y.shape[3]
        # Extract the i-th column of y
        column = [y.data[(j - 1) * y.shape[3] + i] for j in 1:y.shape[2]]
        
        # Compute dot product of x.data and the extracted column
        sum = 0.0
        for (a, b) in zip(x.data, column)
            sum += a * b
        end
        push!(out.data, sum)
    end

    return out
end


function discreteSummation(tensor_vector::Vector{Tensor})
    shape = tensor_vector[1].shape
    summed = zeroTensor(shape)
    for x in tensor_vector
        summed+=x
    end
    return summed
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

mutable struct Model 
    layers::Vector{DenseLayer}
    input_size::Vector{Int}

    loss::String
    optimizer::String
    learning_rate::Float64
    metrics::Vector{String}

    function Model(input_size::Vector{Int})
        return new(DenseLayer[], input_size, "-", "-", 0.00, ["-"])
    end
end

function addLayer!(model::Model, dense::DenseLayer)
    push!(model.layers, dense)
    model.input_size = dense.output_size
end

function compile(model::Model, loss::String, optimizer::String, learning_rate::Float64, metrics::Vector{String})
    model.loss = loss 
    model.optimizer = optimizer
    model.learning_rate = learning_rate
    model.metrics = metrics
end

function perform(neuron::Neuron)
    # println("Neuron: ", neuron.name)
    # println("State Shape: ", neuron.state.shape)
    # println("Weight Shape: ", neuron.weight.shape)
    # println("Bias Shape: ", neuron.bias.shape)
    temp = Vector{Tensor}()
    input_size = neuron.state.shape
    output_size = neuron.bias.shape
    for x in neuron.intake
        weight = oneTensor([1, prod(input_size), output_size[3]])
        push!(neuron.weight, weight)
        push!(temp, unitmatmul(x, weight))
    end
    neuron.state = discreteSummation(temp)
    neuron.state = neuron.state + neuron.bias
    neuron.out = getfield(Main, Symbol(neuron.name))(neuron.state)
    return neuron.out
end


function forwardPropagate(model::Model, input::Tensor)
    input = flatten(input)
    current_in = [input]
    layer_out = Vector{Tensor}()
    for layer in model.layers
        layer_out = Vector{Tensor}()
        for neuron in layer.layer
            neuron.intake = current_in
            perform(neuron)
            push!(layer_out, neuron.out)
        end
        current_in = layer_out
    end
    return layer_out
end







#-------------------------------------<<< TESTING STUFF >>>----------------------------------------------------------------------------------
function base_test()
    arr = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
    arr1 = [[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0], [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0], [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]]
    shape = [1,2,4]
    shape2 = [2,2,2]
    tensor = Tensor(arr, shape)
    # index = [1, 1, 1]
    # println(valueAt(tensor, index))
    # newtensor = Tensor(arr, shape) + Tensor(arr, shape2)
    # println(newtensor)
    # ntensor2 = discreteSummation([tensor, tensor, tensor])
    # println(ntensor2)

    input_size = [1,2,4]
    model = Model(input_size)
    addLayer!(model, DenseLayer("ReLU", 32, [1,1,8]))
    addLayer!(model, DenseLayer("ReLU", 64, [1,1,32]))
    addLayer!(model, DenseLayer("ReLU", 128, [1,1,64]))
    addLayer!(model, DenseLayer("ReLU", 64, [1,1,128]))
    addLayer!(model, DenseLayer("ReLU", 32, [1,1,64]))
    addLayer!(model, DenseLayer("ReLU", 1, [1,1,32]))

    # addLayer!(model, DenseLayer("ReLU", 2, [1,1,8]))
    # addLayer!(model, DenseLayer("ReLU", 2, [1,1,2]))
    # addLayer!(model, DenseLayer("ReLU", 1, [1,1,2]))

    compile(model, "mean-squared-error", "SGD", 0.01, ["accuracy"])
    forwardPropagate(model, tensor)
    #println(model)
end

base_test()


