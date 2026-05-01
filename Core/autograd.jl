# Core/autograd.jl

using LinearAlgebra
using NNlib

#-----------Supplementary functions----------------

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

#-----------Backpropagation----------------

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