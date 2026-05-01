# Models/model.jl

using ..Core
using ..Layers

mutable struct Model
    layers::Vector{Dense}
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
    T = eltype(model.layers[1].w.data)
    x = T.(x)  
    
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