# Core/ops.jl

import Base: +, *, /, -, size, reshape
using Random  
using LinearAlgebra 
using NNlib

const OpCacheKey = Tuple{UInt, UInt, DataType}

#-----------Base: Add & AddBackward----------------

struct AddBackward     <: BackwardOp; a::Tensor; b::Tensor; end

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

function backward!(op::AddBackward, grad)
    ensure_grad!(op.a); ensure_grad!(op.b)
    op.a.grad .+= grad
    op.b.grad .+= grad
end

#-----------Base: Sub & SubBackward----------------

struct SubBackward     <: BackwardOp; a::Tensor; b::Tensor; end

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

function backward!(op::SubBackward, grad)
    ensure_grad!(op.a); ensure_grad!(op.b)
    op.a.grad .+= grad
    op.b.grad .-= grad
end

#-----------Base: Mul & MulBackward----------------

struct MulBackward     <: BackwardOp; a::Tensor; b::Tensor; end

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

function backward!(op::MulBackward, grad)
    ensure_grad!(op.a); ensure_grad!(op.b)
    op.a.grad .+= op.b.data .* grad
    op.b.grad .+= op.a.data .* grad
end

#-----------Base: Div & DivBackward----------------

struct DivBackward     <: BackwardOp; a::Tensor; b::Tensor; end

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

function backward!(op::DivBackward, grad)
    ensure_grad!(op.a); ensure_grad!(op.b)
    op.a.grad .+= (1 ./ op.b.data) .* grad
    op.b.grad .-= (op.a.data ./ (op.b.data .^ 2)) .* grad
end


#-----------MatMul & MatMulBackward----------------

struct MatMulBackward  <: BackwardOp; a::Tensor; b::Tensor; end

function matmul(a::Tensor, b::Tensor, cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
    key = (objectid(a), objectid(b), MatMulBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        if ndims(a.data) <= 2 && ndims(b.data) <= 2
            mul!(out.data, a.data, b.data)
        else
            out.data .= NNlib.batched_mul(a.data, b.data)
        end
    else
        result = if ndims(a.data) <= 2 && ndims(b.data) <= 2
            a.data * b.data
        else
            NNlib.batched_mul(a.data, b.data)
        end
        out = Tensor(result)
        cache !== nothing && (cache[key] = out)
        out.parents = [a, b]
        out.backward = MatMulBackward(a, b)
    end
    return out
end

function backward!(op::MatMulBackward, grad)
    ensure_grad!(op.a)
    ensure_grad!(op.b)
    a_nd = ndims(op.a.data)
    b_nd = ndims(op.b.data)
    if a_nd == 2 && b_nd == 1
        
        op.a.grad .+= grad * op.b.data'
        op.b.grad .+= op.a.data' * grad
    elseif a_nd <= 2 && b_nd <= 2
        
        op.a.grad .+= grad * op.b.data'
        op.b.grad .+= op.a.data' * grad
    else
        
        op.a.grad .+= NNlib.batched_mul(grad,
                           NNlib.batched_adjoint(op.b.data))
        op.b.grad .+= NNlib.batched_mul(
                           NNlib.batched_adjoint(op.a.data), grad)
    end
end

#-----------Base: Sum, reduce_sum & SumBackward----------------

struct SumBackward <: BackwardOp
    t::Tensor
    dims::Union{Nothing, Int, Tuple}
    keepdims::Bool
    input_size::Tuple
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
        out.backward = SumBackward(t, nothing, false, size(t.data))
    end
    return out
end

function reduce_sum(t::Tensor, dims; keepdims=false, cache=nothing)
    key = (objectid(t), UInt(hash((dims, keepdims))), SumBackward)

    result = sum(t.data, dims=dims)
    if !keepdims
        result = dropdims(result, dims=dims)
    end

    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= result
    else
        out = Tensor(result)
        cache !== nothing && (cache[key] = out)
        out.parents  = [t]
        out.backward = SumBackward(t, dims, keepdims, size(t.data))
    end
    return out
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

#-----------ReLU & ReLUBackward----------------

struct ReLUBackward    <: BackwardOp; t::Tensor; end

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

function backward!(op::ReLUBackward, grad)
    ensure_grad!(op.t)
    op.t.grad .+= (op.t.data .> 0) .* grad
end

#-----------LeakyReLU & LeakyReLUBackward----------------

struct LeakyReLUBackward <: BackwardOp; t::Tensor; alpha::Float32; end

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

function backward!(op::LeakyReLUBackward, grad)
    ensure_grad!(op.t)
    op.t.grad .+= ((op.t.data .> 0) .+ op.alpha .* (op.t.data .<= 0)) .* grad
end

#-----------Linear & LinearBackward----------------

struct LinearBackward  <: BackwardOp; t::Tensor; end

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

function backward!(op::LinearBackward, grad)
    ensure_grad!(op.t)
    op.t.grad .+= grad
end

#-----------Softmax & SoftmaxBackward----------------

struct SoftmaxBackward <: BackwardOp
    t::Tensor
    out::Tensor
    dims::Int
end

function softmax(t::Tensor, dims::Int=1; cache=nothing)
    key = (objectid(t), UInt(dims), SoftmaxBackward)
    x   = t.data .- maximum(t.data, dims=dims)
    ex  = exp.(x)
    result = ex ./ sum(ex, dims=dims)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data .= result
    else
        out = Tensor(result)
        cache !== nothing && (cache[key] = out)
        out.parents  = [t]
        out.backward = SoftmaxBackward(t, out, dims)
    end
    return out
end

function backward!(op::SoftmaxBackward, grad)
    ensure_grad!(op.t)
    s   = op.out.data
    dot = sum(grad .* s, dims=op.dims)
    op.t.grad .+= s .* (grad .- dot)
end

#-----------ScalarDivBackward----------------

struct ScalarDivBackward <: BackwardOp
    t::Tensor
    n::Real
end

function backward!(op::ScalarDivBackward, grad)
    ensure_grad!(op.t)
    op.t.grad .+= grad[1] / op.n
end
