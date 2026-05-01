module SynapseEngine

include("../Core/Core.jl")
include("../Layers/Layers.jl")
include("../Initializers/Initializers.jl")
include("../Losses/Losses.jl")
include("../Optimizers/Optimizers.jl")
include("../Models/Models.jl")

# Re-export user-facing API
using .Core
using .Layers
using .Initializers
using .Losses
using .Optimizers
using .Models

export Tensor, BackwardOp, matmul, backprop, ensure_grad!, OpCacheKey, reduce_sum, ReLU, LeakyReLU, Linear, softmax, ScalarDivBackward, quick_sig, build_topo, graph_signature, backward!,
       Dense, clip_grad!,
       zeroes_init, ones_init, xavier_init, identity_init,
       mse_loss,
       Optimizer, SGD, Adam, AdamW, step!, clip_grad_tensor!,
       Model, get_cache, register_params!, model_params, clip_grad_tensor!, backprop!, train!, fit!

end