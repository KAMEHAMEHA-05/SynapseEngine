# Optimizers/utils.jl

include("../Core/Core.jl")

function clip_grad_tensor!(p::Tensor, max_norm::Real)
    p.grad === nothing && return
    n = norm(p.grad)
    n > max_norm && (p.grad .*= max_norm / n)
end