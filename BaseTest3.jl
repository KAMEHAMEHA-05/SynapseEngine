include("Exceptions.jl")
using .Exceptions: raise_dimension_mismatch, raise_indexoutofbounds
using Revise
import Base: +, *, /, -, size, reshape
using Random  # For randn and rand
using LinearAlgebra 

mutable struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    grad::Union{Nothing, A}
    parents::Vector{Tensor}
    backward::Union{Nothing, Function}
end

Base.size(t::Tensor) = size(t.data)
Base.getindex(t::Tensor, I...) = t.data[I...]
Base.setindex!(t::Tensor, v, I...) = (t.data[I...] = v)
Base.IndexStyle(::Type{<:Tensor}) = IndexStyle(Array)

Tensor(data::AbstractArray{T,N}) where {T,N} =
    Tensor{T,N,typeof(data)}(data, nothing, [], ()->nothing)

function ensure_grad!(t::Tensor)
    if t.grad === nothing
        t.grad = zeros(eltype(t.data), size(t.data))
    end
end

function Base.:+(a::Tensor, b::Tensor)
    out = Tensor(a.data + b.data)
    out.parents = [a, b]
    out.backward = ()->begin
        ensure_grad!(a)
        ensure_grad!(b)
        a.grad .+= out.grad
        b.grad .+= out.grad
    end
    return out
end


function Base.:-(a::Tensor, b::Tensor)
    out = Tensor(a.data .- b.data)
    out.parents = [a, b]
    out.backward = ()->begin
        ensure_grad!(a)
        ensure_grad!(b)
        a.grad .+= out.grad
        b.grad .-= out.grad
    end
    return out
end

function Base.:*(a::Tensor, b::Tensor)
    out = Tensor(a.data .* b.data)
    out.parents = [a, b]
    out.backward = ()->begin
        ensure_grad!(a)
        ensure_grad!(b)
        a.grad .+= b.data .* out.grad
        b.grad .+= a.data .* out.grad
    end
    return out
end

function Base.:/(a::Tensor, b::Tensor)
    out = Tensor(a.data ./ b.data)
    out.parents = [a, b]
    out.backward = ()->begin
        ensure_grad!(a)
        ensure_grad!(b)
        a.grad .+= (1 ./ b.data) .* out.grad
        b.grad .-= (a.data ./ (b.data .^ 2)) .* out.grad
    end
    return out
end

function matmul(a::Tensor, b::Tensor)
    out = Tensor(a.data * b.data)
    out.parents = [a, b]
    out.backward = ()->begin
        ensure_grad!(a)
        ensure_grad!(b)
        a.grad .+= out.grad * b.data'
        b.grad .+= a.data' * out.grad
    end
    return out
end

function ReLU(t::Tensor)
    out = Tensor(max.(t.data, 0))
    out.parents = [t]
    out.backward = ()->begin
        ensure_grad!(t)
        t.grad .+= (t.data .> 0) .* out.grad
    end
    return out
end

function Linear(t::Tensor)
    out = Tensor(t.data)
    out.parents = [t]
    out.backward = ()->begin
        ensure_grad!(t)
        t.grad .+= 1* out.grad
    end
    return out
end

struct Node{T,N,A<:AbstractArray{T,N}, F}
    w::Tensor{T,N,A}
    b::Tensor{T,N,A}
    activation::F
end

function (node::Node)(x::Tensor)
    z = matmul(node.w, x) .+ node.b
    return node.activation(z)
end

mutable struct Layer{Tw, Tb, F}
    w::Tw
    b::Tb
    activation::F
    trainable::Bool
    # next::Vector{Layer}
    # prev::Vector{Layer}
    # output::Tensor
    # ready::Int
end

function Layer(w::AbstractArray{T,2}, b::AbstractArray{T,1}, activation::F; trainable=true) where {T,F}
    return Layer(
        Tensor(w),
        Tensor(b),
        activation,
        trainable
    )
end
function Layer(::Type{T}, in_dim::Int, out_dim::Int, init::Function, activation::F; trainable=true) where {T,F}
    W, b = init(T, out_dim, in_dim)
    return Layer(Tensor(W), Tensor(b), activation, trainable)
end

function (layer::Layer)(x::Tensor)
    z = matmul(layer.w, x) + layer.b
    return layer.activation(z)
end

struct Model
    layers::Vector{Layer}
    forward::Function
end

function (model::Model)(x::Tensor)
    return model.forward(x)
end

function zeroes_init(::Type{T}, out_dim, in_dim) where T
    return zeros(T, out_dim, in_dim), zeros(T, out_dim)
end

function ones_init(::Type{T}, out_dim, in_dim) where T
    return ones(T, out_dim, in_dim), ones(T, out_dim)
end

function xavier_init(::Type{T}, out_dim, in_dim) where T
    scale = T(sqrt(1.0 / in_dim))
    return randn(T, out_dim, in_dim) .* scale, zeros(T, out_dim)
end

function identity_init(::Type{T}, out_dim, in_dim) where T
    @assert out_dim == in_dim "identity_init requires square weight matrix"
    return Matrix{T}(I, out_dim, in_dim), zeros(T, out_dim)
end


# T = Float16
# x1 = Tensor(randn(T, 3))
# println(x1.data)
# x2 = Layer(T, 3, 5, ones_init, ReLU)(x1)
# x3 = Layer(T, 5, 2, ones_init, ReLU)(x2)
# x4 = Layer(T, 2, 1, ones_init, ReLU)(x3)

T = Float16
x = Tensor([1.133, -0.012184, -1.824])
println(x.data)
x = Layer(T, 3, 5, ones_init, ReLU)(x)
x1 = Layer(T, 5, 7, ones_init, ReLU)(x)
x2 = Layer(T, 5, 7, ones_init, ReLU)(x)
x3 = Layer(T, 7, 2, ones_init, Linear)(x1) + Layer(T, 7, 2, ones_init, Linear)(x2)
x = Layer(T, 2, 2, identity_init, ReLU)(x3)
x = Layer(T, 2, 1, ones_init, ReLU)(x)

println(x.data)


