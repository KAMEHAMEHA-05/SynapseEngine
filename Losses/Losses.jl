# Losses/Losses.jl

module Losses

using ..Core

function mse_loss(pred::Tensor, target::Tensor, cache=nothing)
    diff = Base.:-(pred, target, cache)
    sq   = Base.:*(diff, diff, cache)
    s    = Base.sum(sq, cache)
    n    = eltype(pred.data)(length(pred.data))
    key  = (objectid(s), UInt(0), ScalarDivBackward)
    if cache !== nothing && haskey(cache, key)
        out = cache[key]
        out.data[1] = s.data[1] / n
    else
        out = Tensor([s.data[1] / n])
        cache !== nothing && (cache[key] = out)
    end
    if isempty(out.parents); out.parents = [s]; end
    out.backward = ScalarDivBackward(s, n)
    return out
end

export mse_loss

end
