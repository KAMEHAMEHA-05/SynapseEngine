include("Core/Core.jl")
include("Layers/Layers.jl")
include("Initializers/Initializers.jl")
include("Losses/Losses.jl")
include("Optimizers/Optimizers.jl")
include("Models/Models.jl")


using .Core
using .Layers: Dense
using .Initializers
using .Losses: mse_loss
using .Optimizers: Adam, SGD
using .Models: Model, train!

using Random
Random.seed!(42)

T = Float32
l1 = Dense(T, 2, 4, xavier_init, LeakyReLU)
l2 = Dense(T, 4, 1, xavier_init, Linear)
model = Model([l1, l2], x -> l2(l1(x)))

#opt = Adam(lr=1f-3)
opt = SGD(lr=1f-3)
xor_inputs  = [Float32[0,0], Float32[0,1], Float32[1,0], Float32[1,1]]
xor_targets = [Float32[0],   Float32[1],   Float32[1],   Float32[0]]

for epoch in 1:3000
    total_loss = 0f0
    for i in shuffle(1:4)
        total_loss += train!(model, xor_inputs[i], Tensor(xor_targets[i]), opt)
    end
    epoch % 500 == 0 && println("epoch=$epoch  loss=$total_loss")
    if total_loss < 1f-4
        println("Converged at epoch $epoch")
        break
    end
end