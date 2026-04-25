include("Exceptions.jl")
using .Exceptions: raise_dimension_mismatch, raise_indexoutofbounds
using Revise
import Base: +, *, /, -, size, reshape
using Random  # For randn and rand
using LinearAlgebra 

abstract type BackwardOp end

mutable struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    grad::Union{Nothing, A}
    parents::Vector{Tensor}
    backward::Union{Nothing, BackwardOp}
    visited::Bool
end

Base.size(t::Tensor) = size(t.data)
Base.getindex(t::Tensor, I...) = t.data[I...]
Base.setindex!(t::Tensor, v, I...) = (t.data[I...] = v)
Base.IndexStyle(::Type{<:Tensor}) = IndexStyle(Array)

Tensor(data::AbstractArray{T,N}) where {T,N} =
    Tensor{T,N,typeof(data)}(data, nothing, [], nothing, false)

function ensure_grad!(t::Tensor)
    if t.grad === nothing
        t.grad = similar(t.data)   
        fill!(t.grad, 0)           
    end
end
struct AddBackward     <: BackwardOp; a::Tensor; b::Tensor; end
struct SubBackward     <: BackwardOp; a::Tensor; b::Tensor; end
struct MulBackward     <: BackwardOp; a::Tensor; b::Tensor; end
struct DivBackward     <: BackwardOp; a::Tensor; b::Tensor; end
struct MatMulBackward  <: BackwardOp; a::Tensor; b::Tensor; end
struct SumBackward     <: BackwardOp; t::Tensor;            end
struct ReLUBackward    <: BackwardOp; t::Tensor;            end
struct LeakyReLUBackward <: BackwardOp; t::Tensor; alpha::Float32; end
struct LinearBackward  <: BackwardOp; t::Tensor;            end


function Base.:+(a::Tensor, b::Tensor)
    a.visited = false
    b.visited = false
    out = Tensor(a.data + b.data)
    out.parents = [a, b]
    # out.backward = ()->begin
    #     ensure_grad!(a)
    #     ensure_grad!(b)
    #     a.grad .+= out.grad
    #     b.grad .+= out.grad
    # end
    out.backward = AddBackward(a, b)
    return out
end


function Base.:-(a::Tensor, b::Tensor)
    a.visited = false
    b.visited = false
    out = Tensor(a.data .- b.data)
    out.parents = [a, b]
    # out.backward = ()->begin
    #     ensure_grad!(a)
    #     ensure_grad!(b)
    #     a.grad .+= out.grad
    #     b.grad .-= out.grad
    # end
    out.backward = SubBackward(a, b)
    return out
end

function Base.:*(a::Tensor, b::Tensor)
    a.visited = false
    b.visited = false
    out = Tensor(a.data .* b.data)
    out.parents = [a, b]
    # out.backward = ()->begin
    #     ensure_grad!(a)
    #     ensure_grad!(b)
    #     a.grad .+= b.data .* out.grad
    #     b.grad .+= a.data .* out.grad
    # end
    out.backward = MulBackward(a, b)
    return out
end

function Base.:/(a::Tensor, b::Tensor)
    a.visited = false
    b.visited = false
    out = Tensor(a.data ./ b.data)
    out.parents = [a, b]
    # out.backward = ()->begin
    #     ensure_grad!(a)
    #     ensure_grad!(b)
    #     a.grad .+= (1 ./ b.data) .* out.grad
    #     b.grad .-= (a.data ./ (b.data .^ 2)) .* out.grad
    # end
    out.backward = DivBackward(a, b)
    return out
end

function matmul(a::Tensor, b::Tensor)
    a.visited = false
    b.visited = false
    out = Tensor(a.data * b.data)
    out.parents = [a, b]
    # out.backward = ()->begin
    #     ensure_grad!(a)
    #     ensure_grad!(b)
    #     a.grad .+= out.grad * b.data'
    #     b.grad .+= a.data' * out.grad
    # end
    out.backward = MatMulBackward(a, b)
    return out
end

function Base.sum(t::Tensor)
    t.visited = false
    out = Tensor([sum(t.data)])
    out.parents = [t]
    # out.backward = ()->begin
    #     ensure_grad!(t)
    #     t.grad .+= out.grad[1]
    # end
    out.backward = SumBackward(t)
    return out
end

function ReLU(t::Tensor)
    t.visited = false
    out = Tensor(max.(t.data, 0))
    out.parents = [t]
    # out.backward = ()->begin
    #     ensure_grad!(t)
    #     t.grad .+= (t.data .> 0) .* out.grad
    # end
    out.backward = ReLUBackward(t)
    return out
end

function LeakyReLU(t::Tensor, alpha=0.01)
    t.visited = false
    out = Tensor(max.(t.data, alpha .* t.data))
    out.parents = [t]
    # out.backward = ()->begin
    #     ensure_grad!(t)
    #     t.grad .+= ((t.data .> 0) .+ alpha .* (t.data .<= 0)) .* out.grad
    # end
    out.backward = LeakyReLUBackward(t, alpha)
    return out
end

function Linear(t::Tensor)
    t.visited = false
    out = Tensor(t.data)
    out.parents = [t]
    # out.backward = ()->begin
    #     ensure_grad!(t)
    #     t.grad .+= 1* out.grad
    # end
    out.backward = LinearBackward(t)
    return out
end

function backward!(op::AddBackward, grad)
    ensure_grad!(op.a); ensure_grad!(op.b)
    op.a.grad .+= grad
    op.b.grad .+= grad
end

function backward!(op::SubBackward, grad)
    ensure_grad!(op.a); ensure_grad!(op.b)
    op.a.grad .+= grad
    op.b.grad .-= grad
end

function backward!(op::MulBackward, grad)
    ensure_grad!(op.a); ensure_grad!(op.b)
    op.a.grad .+= op.b.data .* grad
    op.b.grad .+= op.a.data .* grad
end

function backward!(op::DivBackward, grad)
    ensure_grad!(op.a); ensure_grad!(op.b)
    op.a.grad .+= (1 ./ op.b.data) .* grad
    op.b.grad .-= (op.a.data ./ (op.b.data .^ 2)) .* grad
end

function backward!(op::MatMulBackward, grad)
    ensure_grad!(op.a); ensure_grad!(op.b)
    op.a.grad .+= grad * op.b.data'
    op.b.grad .+= op.a.data' * grad
end

function backward!(op::SumBackward, grad)
    ensure_grad!(op.t)
    op.t.grad .+= grad[1]
end

function backward!(op::ReLUBackward, grad)
    ensure_grad!(op.t)
    op.t.grad .+= (op.t.data .> 0) .* grad
end

function backward!(op::LeakyReLUBackward, grad)
    ensure_grad!(op.t)
    op.t.grad .+= ((op.t.data .> 0) .+ op.alpha .* (op.t.data .<= 0)) .* grad
end

function backward!(op::LinearBackward, grad)
    ensure_grad!(op.t)
    op.t.grad .+= grad
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

function zeroes_init(::Type{T}, out_dim, in_dim) where T
    return zeros(T, out_dim, in_dim), zeros(T, out_dim)
end

function ones_init(::Type{T}, out_dim, in_dim) where T
    return ones(T, out_dim, in_dim), ones(T, out_dim)
end

function xavier_init(::Type{T}, out_dim, in_dim) where T
    scale = T(sqrt(2.0 / in_dim))
    return randn(T, out_dim, in_dim) .* scale, zeros(T, out_dim)
end

function identity_init(::Type{T}, out_dim, in_dim) where T
    @assert out_dim == in_dim "identity_init requires square weight matrix"
    return Matrix{T}(I, out_dim, in_dim), zeros(T, out_dim)
end

function mse_loss(pred::Tensor, target::Tensor)
    diff = pred - target
    return sum(diff * diff) / Tensor(fill(eltype(pred.data)(length(pred.data)), (1,)))
end


# T = Float16
# x1 = Tensor(randn(T, 3))
# println(x1.data)
# x2 = Layer(T, 3, 5, ones_init, ReLU)(x1)
# x3 = Layer(T, 5, 2, ones_init, ReLU)(x2)
# x4 = Layer(T, 2, 1, ones_init, ReLU)(x3)

# T = Float16
# x = Tensor([1.133, -0.012184, -1.824])
# println(x.data)
# x = Layer(T, 3, 5, ones_init, ReLU)(x)
# x1 = Layer(T, 5, 7, ones_init, ReLU)(x)
# x2 = Layer(T, 5, 7, ones_init, ReLU)(x)
# x3 = Layer(T, 7, 2, ones_init, Linear)(x1) + Layer(T, 7, 2, ones_init, Linear)(x2)
# x = Layer(T, 2, 2, identity_init, ReLU)(x3)
# x = Layer(T, 2, 1, ones_init, ReLU)(x)
# println(x.data)

function build_topo(t::Tensor)
    seen = Set{UInt}()
    order = Tensor[]
    function dfs(node)
        id = objectid(node)
        id in seen && return
        push!(seen, id)
        for p in node.parents
            dfs(p)
        end
        push!(order, node)
    end
    dfs(t)
    return order
end

function backprop(t::Tensor)
    topo = build_topo(t)
    for node in topo
        if node.grad !== nothing
            fill!(node.grad, 0)   
        else
            ensure_grad!(node)    
        end
    end
    fill!(t.grad, 1)              
    for node in reverse(topo)
        if node.backward !== nothing
            backward!(node.backward, node.grad)
        end
    end
end

#backprop(loss)

struct Model
    layers::Vector{Layer}
    forward::Function
end

using Random
Random.seed!(48)

# T = Float32
# l1 = Layer(T, 3, 5, xavier_init, LeakyReLU)
# l2 = Layer(T, 5, 7, xavier_init, LeakyReLU)
# l3 = Layer(T, 5, 7, xavier_init, LeakyReLU)
# l4 = Layer(T, 7, 2, xavier_init, Linear)
# l5 = Layer(T, 7, 2, xavier_init, Linear)
# l6 = Layer(T, 7, 2, xavier_init, LeakyReLU)
# l7 = Layer(T, 2, 1, xavier_init, Linear)

# model = Model(
#     [l1, l2, l3, l4, l5, l6, l7],
#     function(x::Tensor)
#         x = l1(x)
#         x1 = l2(x)
#         x2 = l3(x)
#         # x3 = l4(x1) + l5(x2)
#         x = l6(x2)
#         return l7(x)
#     end
# )

T = Float32
l1 = Layer(T, 1, 1, xavier_init, LeakyReLU)

model = Model(
    [l1],
    function(x::Tensor)
        return l1(x)
    end
)

function (model::Model)(x::Tensor)
    return model.forward(x)
end

# x = model(Tensor([1.133, -0.012184, -1.824]))
# println("Model output: ", x.data)

# y = Tensor([60.0])
# loss = mse_loss(x, y)
# println("MSE Loss: ", loss.data)

function clip_grad!(layer::Layer, max_norm::Real)
    for param in [layer.w, layer.b]
        if param.grad !== nothing
            norm = sqrt(sum(param.grad .^ 2))
            if norm > max_norm
                param.grad .*= max_norm / norm
            end
        end
    end
end

function train!(model::Model, x::Tensor, y::Tensor, lr::Real, loss_fn::F = mse_loss) where F
    pred = model(x)
    loss = loss_fn(pred, y)
    backprop(loss)
    for layer in model.layers
        clip_grad!(layer, 1.0)  # Clip gradients to prevent exploding gradients
        if layer.trainable
            if layer.w.grad !== nothing
                layer.w.data .-= lr .* layer.w.grad
            end
            if layer.b.grad !== nothing
                layer.b.data .-= lr .* layer.b.grad
            end
        end
    end
    return loss.data[1]
end

function fit!(model::Model, x::Tensor, y::Tensor, epochs::Int, lr::Real, loss_fn::F = mse_loss) where F
    for epoch in 1:epochs
        l = train!(model, x, y, lr, loss_fn)
        if(l<1.00e-6 || l===NaN)
            println("Early stopping at epoch $epoch: Loss = $l")
            break
        end
        # println("Epoch $epoch: Loss = $l")
    end
end

# fit!(model, Tensor([1.133, -0.012184, -1.824]), Tensor([60.0]), 100, 0.019, mse_loss)
fit!(model, Tensor(Float32[1.133]), Tensor(Float32[60.0]), 1000, 0.1, mse_loss)



# println(model(Tensor([1.133, -0.012184, -1.824])))
println(model(Tensor([1.133])))





