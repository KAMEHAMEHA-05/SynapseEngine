# Models/Models.jl

module Models

using ..Core
using ..Layers
using ..Initializers
using ..Losses
using ..Optimizers

include("model.jl")
include("utils.jl")
include("train.jl")

export Model,
       get_cache,
       register_params!,
       model_params,
       clip_grad_tensor!,
       backprop!,
       train!,
       fit!
end 





