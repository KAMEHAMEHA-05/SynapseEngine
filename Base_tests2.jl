include("Exceptions.jl")
using .Exceptions: raise_dimension_mismatch, raise_indexoutofbounds
using Revise
import Base: +, *, /, -, size, reshape
using Random  # For randn and rand
using LinearAlgebra 

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

function ReLU_derivative(x::Tensor)
    result = Tensor(zeros(length(x.data)), x.shape)
    for i in 1:length(x.data)
        result.data[i] = x.data[i] > 0 ? 1.0 : 0.0
    end
    return result
end

function flatten(inp::Tensor)
    inp.shape = [1,1,prod(inp.shape)]
    return inp
end

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

mutable struct Neuron
    name::String
    weight::Vector{Tensor}
    bias::Tensor
    state::Tensor
    out::Tensor
    init_function::String

    weight_gradients::Vector{Tensor}
    bias_gradient::Tensor
    delta::Tensor

    function Neuron(name::String, input_size::Vector{Int}, output_size::Vector{Int}, init_function::String="xavier_uniform")
        state = zeroTensor(input_size)
        weight = []
        bias = zeroTensor(output_size) 
        out = zeroTensor(output_size) 
        weight_gradients = Tensor[]
        bias_gradient = zeroTensor(output_size)
        delta = zeroTensor(output_size)
        neuron = new(name, weight, bias, state, out, init_function, weight_gradients, bias_gradient, delta)
        return neuron
    end
end

mutable struct DenseLayer
    layer::Vector{Neuron}
    input_size::Vector{Int}
    output_size::Vector{Int}
    init_function::String  # Add this field

    function DenseLayer(name::String, units::Int, input_size::Vector{Int}, init_function::String="xavier_uniform")
        layer = Vector{Neuron}()
        for x in 1:units 
            neuron = Neuron(name, input_size, [1, 1, units], init_function)
            push!(layer, neuron)
        end
        return new(layer, input_size, [1,units], init_function)
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

    for layer in model.layers
        for neuron in layer.layer
            input_size = neuron.state.shape
            output_size = neuron.bias.shape
            weight_shape = [1, prod(input_size), output_size[3]]
            
            # Initialize weights using the specified initialization function
            weight = getfield(Main, Symbol(neuron.init_function))(weight_shape)
            push!(neuron.weight, weight)
            
            # Initialize bias (typically with zeros, but could add bias initialization option)
            neuron.bias = zeroTensor(output_size)
        end
    end
end

function perform(neuron::Neuron, intake::Vector{Tensor})
    # println("Neuron: ", neuron.name)
    # println("State Shape: ", neuron.state.shape)
    # println("Weight Shape: ", neuron.weight.shape)
    # println("Bias Shape: ", neuron.bias.shape)
    temp = Vector{Tensor}()
    input_size = neuron.state.shape
    output_size = neuron.bias.shape
    for x in intake
        weight = oneTensor([1, prod(input_size), output_size[3]])
        push!(neuron.weight, weight)
        push!(temp, unitmatmul(x, weight))
    end
    neuron.state = discreteSummation(temp)
    neuron.state = neuron.state + neuron.bias
    neuron.out = getfield(Main, Symbol(neuron.name))(neuron.state)
    return neuron.out
end

function mse_derivative(predicted::Tensor, target::Tensor)
    raise_dimension_mismatch(predicted.shape, target.shape)
    error = Tensor(Float64[], predicted.shape)
    for i in 1:length(predicted.data)
        push!(error.data, 2 * (predicted.data[i] - target.data[i]))
    end
    return error
end

function forwardPropagate(model::Model, input::Tensor)
    input = flatten(input)
    current_in = [input]
    layer_out = Vector{Tensor}()
    for layer in model.layers
        layer_out = Vector{Tensor}()
        for neuron in layer.layer
            perform(neuron, current_in)
            push!(layer_out, neuron.out)
        end
        current_in = layer_out
    end
    return layer_out
end

function backward(neuron::Neuron, input::Tensor, upstream_gradient::Tensor)
    # Calculate local gradient based on activation function
    if neuron.name == "ReLU"
        local_grad = ReLU_derivative(neuron.state)
    else
        error("Unsupported activation function: $(neuron.name)")
    end
    
    # Calculate delta (local gradient * upstream gradient)
    neuron.delta = Tensor(Float64[], local_grad.shape)
    for i in 1:length(local_grad.data)
        push!(neuron.delta.data, local_grad.data[i] * upstream_gradient.data[i])
    end
    
    # Calculate weight gradients
    input_flat = flatten(input)
    for i in 1:length(neuron.weight)
        grad = Tensor(Float64[], neuron.weight[i].shape)
        for j in 1:length(input_flat.data)
            for k in 1:length(neuron.delta.data)
                push!(grad.data, input_flat.data[j] * neuron.delta.data[k])
            end
        end
        if i <= length(neuron.weight_gradients)
            neuron.weight_gradients[i] = grad
        else
            push!(neuron.weight_gradients, grad)
        end
    end
    
    # Calculate bias gradient
    neuron.bias_gradient = neuron.delta
    
    # Calculate and return gradient for previous layer
    prev_gradient = Tensor(Float64[], input.shape)
    for i in 1:length(input_flat.data)
        sum = 0.0
        for j in 1:length(neuron.delta.data)
            for w in neuron.weight
                sum += w.data[i * length(neuron.delta.data) + j] * neuron.delta.data[j]
            end
        end
        push!(prev_gradient.data, sum)
    end
    
    return prev_gradient
end

function backpropagate!(model::Model, input::Tensor, target::Tensor, output::Vector{Tensor})
    if length(output) != length(target.data)
        raise_dimension_mismatch([length(output)], [length(target.data)])
    end
    
    # Calculate initial gradient from loss function
    current_gradient = mse_derivative(output[1], target)
    
    # Iterate through layers in reverse
    layer_input = input
    for layer_idx in length(model.layers):-1:1
        layer = model.layers[layer_idx]
        layer_gradients = Vector{Tensor}()
        
        # Calculate gradients for each neuron in the layer
        for neuron_idx in length(layer.layer):-1:1
            neuron = layer.layer[neuron_idx]
            neuron_gradient = backward(neuron, layer_input, current_gradient)
            push!(layer_gradients, neuron_gradient)
        end
        
        # Update current gradient for next layer
        current_gradient = discreteSummation(layer_gradients)
        
        # Update layer input for next iteration
        if layer_idx > 1
            layer_input = model.layers[layer_idx-1].layer[1].out
        end
    end
end

# Update weights and biases using gradient descent
function update_parameters!(model::Model)
    for layer in model.layers
        for neuron in layer.layer
            # Update weights
            for i in 1:length(neuron.weight)
                weight_update = Tensor(Float64[], neuron.weight[i].shape)
                for j in 1:length(neuron.weight[i].data)
                    grad_value = j <= length(neuron.weight_gradients[i].data) ? 
                                neuron.weight_gradients[i].data[j] : 0.0
                    update = -model.learning_rate * grad_value
                    push!(weight_update.data, update)
                end
                neuron.weight[i] = neuron.weight[i] + weight_update
            end
            # Update bias
            bias_update = Tensor(Float64[], neuron.bias.shape)
            for i in 1:length(neuron.bias.data)
                update = -model.learning_rate * neuron.bias_gradient.data[i]
                push!(bias_update.data, update)
            end
            neuron.bias = neuron.bias + bias_update
        end
    end
end

# Training function that combines forward and backward passes
function train_step!(model::Model, input::Tensor, target::Tensor)
    # Forward pass
    output = forwardPropagate(model, input)
    
    # Backward pass
    backpropagate!(model, input, target, output)
    
    # Update parameters
    update_parameters!(model)
    
    # Calculate and return loss
    loss = 0.0
    for (pred, targ) in zip(output[1].data, target.data)
        loss += (pred - targ)^2
    end
    return loss / length(output[1].data)
end

# Training loop for multiple epochs
function train!(model::Model, inputs::Vector{Tensor}, targets::Vector{Tensor}, epochs::Int)
    for epoch in 1:epochs
        total_loss = 0.0
        for (input, target) in zip(inputs, targets)
            loss = train_step!(model, input, target)
            total_loss += loss
        end
        avg_loss = total_loss / length(inputs)
        println("Epoch $epoch: Average Loss = $avg_loss")
    end
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
    # addLayer!(model, DenseLayer("ReLU", 32, [1,1,8]))
    # addLayer!(model, DenseLayer("ReLU", 64, [1,1,32]))
    # addLayer!(model, DenseLayer("ReLU", 128, [1,1,64]))
    # addLayer!(model, DenseLayer("ReLU", 64, [1,1,128]))
    # addLayer!(model, DenseLayer("ReLU", 32, [1,1,64]))
    # addLayer!(model, DenseLayer("ReLU", 1, [1,1,32]))

    # addLayer!(model, DenseLayer("ReLU", 2, [1,1,8]))
    # addLayer!(model, DenseLayer("ReLU", 2, [1,1,2]))
    # addLayer!(model, DenseLayer("ReLU", 1, [1,1,2]))

    addLayer!(model, DenseLayer("ReLU", 32, [1,1,8], "xavier_uniform"))
    addLayer!(model, DenseLayer("ReLU", 64, [1,1,32], "he_normal"))
    addLayer!(model, DenseLayer("ReLU", 128, [1,1,64], "uniform"))
    addLayer!(model, DenseLayer("ReLU", 64, [1,1,128], "he_normal"))
    addLayer!(model, DenseLayer("ReLU", 32, [1,1,64], "xavier_uniform"))
    addLayer!(model, DenseLayer("ReLU", 1, [1,1,32], "xavier_uniform"))

    compile(model, "mean-squared-error", "SGD", 0.01, ["accuracy"])
    forwardPropagate(model, tensor)

    model = Model([1,2,4])
    addLayer!(model, DenseLayer("ReLU", 32, [1,1,8]))
    addLayer!(model, DenseLayer("ReLU", 1, [1,1,32]))
    compile(model, "mean-squared-error", "SGD", 0.01, ["accuracy"])

    # Prepare your training data
    inputs = [Tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0], [1,2,4])]
    targets = [Tensor([1.0], [1,1,1])]

    # Train the model
    train!(model, inputs, targets, 10)  # Train for 10 epochs
    #println(model)
end

base_test()


