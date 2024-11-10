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
    out = deepcopy_tensor(inp)
    for (i, x) in enumerate(out.data)
        if (x < 0)
            out.data[i] = 0
        end
    end
    return out
end

# Sigmoid activation function and its derivative
function Sigmoid(inp::Tensor)
    out = deepcopy(inp)
    for (i, x) in enumerate(inp.data)
        out.data[i] = 1.0 / (1.0 + exp(-x))
    end
    return out
end

function Sigmoid_derivative(x::Tensor)
    derivative = Tensor(Float64[], x.shape)
    for val in x.data
        sig = 1.0 / (1.0 + exp(-val))
        push!(derivative.data, sig * (1.0 - sig))
    end
    return derivative
end

# Tanh activation function and its derivative
function Tanh(inp::Tensor)
    out = deepcopy(inp)
    for (i, x) in enumerate(inp.data)
        out.data[i] = tanh(x)
    end
    return out
end

function Tanh_derivative(x::Tensor)
    derivative = Tensor(Float64[], x.shape)
    for val in x.data
        push!(derivative.data, 1.0 - tanh(val)^2)
    end
    return derivative
end

# LeakyReLU activation function and its derivative
function LeakyReLU(inp::Tensor, alpha::Float64=0.01)
    out = deepcopy(inp)
    for (i, x) in enumerate(inp.data)
        out.data[i] = x > 0 ? x : alpha * x
    end
    return out
end

function LeakyReLU_derivative(x::Tensor, alpha::Float64=0.01)
    derivative = Tensor(Float64[], x.shape)
    for val in x.data
        push!(derivative.data, val > 0 ? 1.0 : alpha)
    end
    return derivative
end

# ELU (Exponential Linear Unit) activation function and its derivative
function ELU(inp::Tensor, alpha::Float64=1.0)
    out = deepcopy(inp)
    for (i, x) in enumerate(inp.data)
        out.data[i] = x > 0 ? x : alpha * (exp(x) - 1.0)
    end
    return out
end

function ELU_derivative(x::Tensor, alpha::Float64=1.0)
    derivative = Tensor(Float64[], x.shape)
    for val in x.data
        push!(derivative.data, val > 0 ? 1.0 : alpha * exp(val))
    end
    return derivative
end

# SELU (Scaled Exponential Linear Unit) activation function and its derivative
function SELU(inp::Tensor)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    out = deepcopy(inp)
    for (i, x) in enumerate(inp.data)
        out.data[i] = scale * (x > 0 ? x : alpha * (exp(x) - 1.0))
    end
    return out
end

function SELU_derivative(x::Tensor)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    derivative = Tensor(Float64[], x.shape)
    for val in x.data
        push!(derivative.data, scale * (val > 0 ? 1.0 : alpha * exp(val)))
    end
    return derivative
end

# Softplus activation function and its derivative
function Softplus(inp::Tensor)
    out = deepcopy(inp)
    for (i, x) in enumerate(inp.data)
        out.data[i] = log(1.0 + exp(x))
    end
    return out
end

function Softplus_derivative(x::Tensor)
    derivative = Tensor(Float64[], x.shape)
    for val in x.data
        push!(derivative.data, 1.0 / (1.0 + exp(-val)))
    end
    return derivative
end

# Softmax activation function and its derivative
function Softmax(inp::Tensor)
    out = deepcopy(inp)
    max_val = maximum(inp.data)
    exp_sum = sum(exp.(inp.data .- max_val))
    
    for (i, x) in enumerate(inp.data)
        out.data[i] = exp(x - max_val) / exp_sum
    end
    return out
end

function Softmax_derivative(x::Tensor)
    softmax_output = Softmax(x)
    derivative = Tensor(Float64[], x.shape)
    n = length(x.data)
    
    # For each output unit
    for i in 1:n
        grad = softmax_output.data[i] * (1.0 - softmax_output.data[i])
        push!(derivative.data, grad)
    end
    return derivative
end

# Swish activation function (x * sigmoid(x)) and its derivative
function Swish(inp::Tensor, beta::Float64=1.0)
    out = deepcopy(inp)
    for (i, x) in enumerate(inp.data)
        sigmoid_x = 1.0 / (1.0 + exp(-beta * x))
        out.data[i] = x * sigmoid_x
    end
    return out
end

function Swish_derivative(x::Tensor, beta::Float64=1.0)
    derivative = Tensor(Float64[], x.shape)
    for val in x.data
        sigmoid_x = 1.0 / (1.0 + exp(-beta * val))
        push!(derivative.data, beta * val * sigmoid_x * (1.0 - sigmoid_x) + sigmoid_x)
    end
    return derivative
end

# GELU (Gaussian Error Linear Unit) activation function and its derivative
function GELU(inp::Tensor)
    out = deepcopy(inp)
    for (i, x) in enumerate(inp.data)
        out.data[i] = 0.5 * x * (1.0 + tanh(sqrt(2.0/π) * (x + 0.044715 * x^3)))
    end
    return out
end

function GELU_derivative(x::Tensor)
    derivative = Tensor(Float64[], x.shape)
    for val in x.data
        cdf = 0.5 * (1.0 + tanh(sqrt(2.0/π) * (val + 0.044715 * val^3)))
        pdf = exp(-(val^2)/2.0) / sqrt(2.0 * π)
        push!(derivative.data, cdf + val * pdf)
    end
    return derivative
end

# Mish activation function (x * tanh(softplus(x))) and its derivative
function Mish(inp::Tensor)
    out = deepcopy(inp)
    for (i, x) in enumerate(inp.data)
        soft_plus = log(1.0 + exp(x))
        out.data[i] = x * tanh(soft_plus)
    end
    return out
end

function Mish_derivative(x::Tensor)
    derivative = Tensor(Float64[], x.shape)
    for val in x.data
        soft_plus = log(1.0 + exp(val))
        tanh_sp = tanh(soft_plus)
        sech_sp = 1.0 / cosh(soft_plus)
        sigmoid = 1.0 / (1.0 + exp(-val))
        
        grad = tanh_sp + val * sigmoid * sech_sp^2
        push!(derivative.data, grad)
    end
    return derivative
end

function flatten(inp::Tensor)
    out = deepcopy_tensor(inp)
    out.shape = [1, 1, prod(out.shape)]
    return out
end

end

