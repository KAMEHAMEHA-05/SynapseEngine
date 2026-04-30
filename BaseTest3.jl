include("Exceptions.jl")
using .Exceptions: raise_dimension_mismatch, raise_indexoutofbounds
using Revise
import Base: +, *, /, -, size, reshape
using Random  # For randn and rand
using LinearAlgebra 
using NNLib

abstract type BackwardOp end
const OpCacheKey = Tuple{UInt, UInt, DataType}

mutable struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    grad::Union{Nothing, A}
    parents::Vector{Tensor}
    backward::Union{Nothing, BackwardOp}
end

Base.size(t::Tensor) = size(t.data)
Base.getindex(t::Tensor, I...) = t.data[I...]
Base.setindex!(t::Tensor, v, I...) = (t.data[I...] = v)
Base.IndexStyle(::Type{<:Tensor}) = IndexStyle(Array)

Tensor(data::AbstractArray{T,N}) where {T,N} =
    Tensor{T,N,typeof(data)}(data, nothing, [], nothing)

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
struct SumBackward <: BackwardOp
    t::Tensor
    dims::Union{Nothing, Int, Tuple}
    keepdims::Bool
    input_size::Tuple
end
struct ScalarDivBackward <: BackwardOp
    t::Tensor
    n::Real
end
struct SoftmaxBackward <: BackwardOp
    t::Tensor
    out::Tensor
    dims::Int
end
struct ReLUBackward    <: BackwardOp; t::Tensor;            end
struct LeakyReLUBackward <: BackwardOp; t::Tensor; alpha::Float32; end
struct LinearBackward  <: BackwardOp; t::Tensor;            end


function matmul(a::Tensor, b::Tensor, cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(a), objectid(b), MatMulBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        mul!(out.data, a.data, b.data)
    else
        out = Tensor(a.data * b.data)
        cache !== nothing && (cache[key] = out)
        out.parents  = [a, b]
        out.backward = MatMulBackward(a, b)
    end
    return out
end

function Base.:+(a::Tensor, b::Tensor, cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(a), objectid(b), AddBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= a.data .+ b.data
    else
        out = Tensor(a.data .+ b.data)
        cache !== nothing && (cache[key] = out)
        out.parents  = [a, b]
        out.backward = AddBackward(a, b)
    end
    
    return out
end

function Base.:-(a::Tensor, b::Tensor, cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(a), objectid(b), SubBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= a.data .- b.data
    else
        out = Tensor(a.data .- b.data)
        cache !== nothing && (cache[key] = out)
        out.parents  = [a, b]
        out.backward = SubBackward(a, b)
    end
    return out
end

function Base.:*(a::Tensor, b::Tensor, cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(a), objectid(b), MulBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= a.data .* b.data
    else
        out = Tensor(a.data .* b.data)
        cache !== nothing && (cache[key] = out)
        out.parents  = [a, b]
        out.backward = MulBackward(a, b)
    end
    return out
end

function Base.:/(a::Tensor, b::Tensor, cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(a), objectid(b), DivBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= a.data ./ b.data
    else
        out = Tensor(a.data ./ b.data)
        cache !== nothing && (cache[key] = out)
        out.parents  = [a, b]
        out.backward = DivBackward(a, b)
    end
    return out
end

function Base.sum(t::Tensor, cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(t), objectid(t), SumBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= [sum(t.data)]
    else
        out = Tensor([sum(t.data)])
        cache !== nothing && (cache[key] = out)
        out.parents  = [t]
        out.backward = SumBackward(t)
    end
    return out
end

function ReLU(t::Tensor; cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(t), objectid(t), ReLUBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= max.(t.data, 0)
    else
        out = Tensor(max.(t.data, 0))
        cache !== nothing && (cache[key] = out)
        out.parents  = [t]
        out.backward = ReLUBackward(t)
    end
    return out
end

function LeakyReLU(t::Tensor, alpha::Float32=0.01f0; cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(t), objectid(t), LeakyReLUBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= max.(t.data, alpha .* t.data)
    else
        out = Tensor(max.(t.data, alpha .* t.data))
        cache !== nothing && (cache[key] = out)
        out.parents  = [t]
        out.backward = LeakyReLUBackward(t, alpha)
    end
    return out
end

function Linear(t::Tensor; cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(t), objectid(t), LinearBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= t.data
    else
        out = Tensor(copy(t.data))
        cache !== nothing && (cache[key] = out)
        out.parents  = [t]
        out.backward = LinearBackward(t)
    end
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
    if op.dims === nothing
        op.t.grad .+= grad[1]
    else
        g_shape = ntuple(
            i -> i in (op.dims isa Int ? (op.dims,) : op.dims) ? 1 : size(op.t.data, i),
            ndims(op.t.data)
        )
        op.t.grad .+= reshape(op.keepdims ? grad : reshape(grad, g_shape), g_shape)
    end
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


mutable struct Layer{Tw, Tb, F}
    w::Tw
    b::Tb
    activation::F
    trainable::Bool
    _cache::Union{Nothing, Dict{OpCacheKey, Tensor}}
end

function Layer(w::AbstractArray{T,2}, b::AbstractArray{T,1}, activation::F; trainable=true) where {T,F}
    w_tensor = Tensor(w)
    b_tensor = Tensor(b)
    w_tensor.grad = similar(w); fill!(w_tensor.grad, 0)
    b_tensor.grad = similar(b); fill!(b_tensor.grad, 0)
    return Layer(w_tensor, b_tensor, activation, trainable, nothing)
end

function Layer(::Type{T}, in_dim::Int, out_dim::Int, init::Function, activation::F; trainable=true) where {T,F}
    W, b = init(T, out_dim, in_dim)
    w_tensor = Tensor(W)
    b_tensor = Tensor(b)
    w_tensor.grad = similar(W); fill!(w_tensor.grad, 0)
    b_tensor.grad = similar(b); fill!(b_tensor.grad, 0)
    return Layer(w_tensor, b_tensor, activation, trainable, nothing)
end

function (layer::Layer)(x::Tensor)
    c  = layer._cache
    mm = matmul(layer.w, x, c)
    z  = Base.:+(mm, layer.b, c)
    return layer.activation(z; cache=c)  
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

function mse_loss(pred::Tensor, target::Tensor, cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    diff = Base.:-(pred, target, cache)
    sq   = Base.:*(diff, diff, cache)
    n    = Tensor(fill(eltype(pred.data)(length(pred.data)), (1,)))
    key_n = (objectid(pred), UInt(0), DivBackward)
    if cache !== nothing
        if haskey(cache, key_n)
            n = cache[key_n]
            fill!(n.data, eltype(pred.data)(length(pred.data)))
        else
            cache[key_n] = n
        end
    end
    return Base.:/(sum(sq, cache), n, cache)
end


mutable struct Model
    layers::Vector{Layer}
    forward::Function
    _topo_caches::Dict{UInt, Vector{Tensor}}
    _param_ids::Set{UInt}
    _input::Union{Nothing, Tensor}
    _stable_topo::Union{Nothing, Vector{Tensor}}   
    _last_sig::Union{Nothing, UInt}
end

function Model(layers::Vector, forward::Function)
    param_ids = Set{UInt}()
    shared_cache = Dict{OpCacheKey, Tensor}()   
    for layer in layers
        push!(param_ids, objectid(layer.w))
        push!(param_ids, objectid(layer.b))
        layer._cache = shared_cache              
    end
    return Model(layers, forward, Dict{UInt, Vector{Tensor}}(), param_ids, nothing, nothing, nothing)
end

function (model::Model)(x::AbstractArray)
    if model._input === nothing
        model._input = Tensor(copy(x))
        model._input.grad = similar(model._input.data)
        fill!(model._input.grad, 0)
    elseif size(x) != size(model._input.data)
        for layer in model.layers
            empty!(layer._cache)
        end
        model._input = Tensor(copy(x))
        model._input.grad = similar(model._input.data)
        fill!(model._input.grad, 0)
    else
        model._input.data .= x
        fill!(model._input.grad, 0)
    end
    return model.forward(model._input)
end

function (model::Model)(x::Tensor)
    model(x.data)
end

function get_cache(model::Model)
    isempty(model.layers) && return nothing
    return model.layers[1]._cache
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

function graph_changed(model::Model, loss::Tensor)
    isempty(model._topo_caches) && return true   
    sig = quick_sig(loss, model._param_ids)
    return !haskey(model._topo_caches, sig)
end


function quick_sig(loss::Tensor, param_ids::Set{UInt})
    h = UInt(0)
    for p in loss.parents
        h = hash(objectid(p), h)
        for pp in p.parents
            h = hash(objectid(pp), h)
        end
    end
    return h
end

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

function graph_signature(topo::Vector{Tensor}, param_ids::Set{UInt})
    h = UInt(0)
    for node in topo
        if objectid(node) in param_ids
            h = hash(objectid(node), h)
        end
    end
    return h
end

function backprop!(model::Model, loss::Tensor)
    if model._stable_topo === nothing
        topo = build_topo(loss)
        sig  = graph_signature(topo, model._param_ids)
        model._topo_caches[sig] = topo
        model._stable_topo      = topo
        model._last_sig         = sig
    else
        sig = UInt(0)
        for p in loss.parents
            sig = hash(objectid(p), sig)
        end
        if sig != model._last_sig
            topo = build_topo(loss)
            new_sig = graph_signature(topo, model._param_ids)
            model._topo_caches[new_sig] = topo
            model._stable_topo          = topo
            model._last_sig             = sig
        end
    end

    topo = model._stable_topo
    for node in topo
        node.grad !== nothing && fill!(node.grad, 0)
    end
    loss.grad === nothing ? (loss.grad = ones(eltype(loss.data), size(loss.data))) : fill!(loss.grad, 1)
    for node in reverse(topo)
        node.grad === nothing && ensure_grad!(node)
        node.backward !== nothing && backward!(node.backward, node.grad)
    end
end

#backprop(loss)
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

# x = model(Tensor([1.133, -0.012184, -1.824]))
# println("Model output: ", x.data)

# y = Tensor([60.0])
# loss = mse_loss(x, y)
# println("MSE Loss: ", loss.data)

function clip_grad!(layer::Layer, max_norm::Real)
    for param in [layer.w, layer.b]
        if param.grad !== nothing
            norm = norm(param.grad)
            if norm > max_norm
                param.grad .*= max_norm / norm
            end
        end
    end
end

function train!(model::Model, x::Tensor, y::Tensor, lr::Real, loss_fn::F = mse_loss) where F
    pred = model(x)
    loss = loss_fn(pred, y, get_cache(model))
    backprop!(model, loss)
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


function backprop(loss::Tensor)
    topo = build_topo(loss)
    for node in topo
        if node.grad !== nothing
            fill!(node.grad, 0)
        end
    end
    if loss.grad === nothing
        loss.grad = ones(eltype(loss.data), size(loss.data))
    else
        fill!(loss.grad, 1)
    end
    for node in reverse(topo)
        if node.grad === nothing
            ensure_grad!(node)
        end
        if node.backward !== nothing
            backward!(node.backward, node.grad)
        end
    end
end




