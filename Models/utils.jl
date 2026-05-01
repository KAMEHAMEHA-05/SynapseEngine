# Models/utils.jl

using LinearAlgebra
using ..Core

function graph_changed(model::Model, loss::Tensor)
    isempty(model._topo_caches) && return true   
    sig = quick_sig(loss, model._param_ids)
    return !haskey(model._topo_caches, sig)
end

function get_cache(model::Model)
    isempty(model.layers) && return nothing
    return model.layers[1]._cache
end

function register_params!(model::Model, tensors::Tensor...)
    for t in tensors
        push!(model._param_ids, objectid(t))
    end
end

function model_params(model::Model)
    params = Tensor[]
    for layer in model.layers
        layer.trainable || continue
        push!(params, layer.w)
        push!(params, layer.b)
    end
    return params
end

function clip_grad_tensor!(p::Tensor, max_norm::Real)
    p.grad === nothing && return
    n = norm(p.grad)
    n > max_norm && (p.grad .*= max_norm / n)
end