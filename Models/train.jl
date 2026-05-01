# Models/train.jl

using ..Core
using ..Losses
using ..Optimizers

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

function train!(model::Model, x, y::Tensor, opt::Optimizer, loss_fn::F=mse_loss) where F
    pred = model(x)
    loss = loss_fn(pred, y, get_cache(model))
    backprop!(model, loss)
    params = model_params(model)
    for p in params; clip_grad_tensor!(p, 1.0f0); end
    step!(opt, params)
    return Float32(loss.data[1])
end

function fit!(model::Model, x, y::Tensor, opt::Optimizer, epochs::Int, loss_fn::F=mse_loss) where F
    for epoch in 1:epochs
        l = train!(model, x, y, opt, loss_fn)
        if l < 1f-6 || isnan(l)
            println("Early stopping at epoch $epoch: Loss = $l")
            break
        end
        # println("Epoch $epoch: Loss = $l")
    end
end