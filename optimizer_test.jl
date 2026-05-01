include("BaseTest3.jl")

# XOR — needs nonlinearity, tests multi-layer gradient flow
T = Float32
l1 = Layer(T, 2, 4, xavier_init, LeakyReLU)
l2 = Layer(T, 4, 1, xavier_init, Linear)

model = Model([l1, l2], x -> l2(l1(x)))

xor_inputs  = [Float32[0,0], Float32[0,1], Float32[1,0], Float32[1,1]]
xor_targets = [Float32[0],   Float32[1],   Float32[1],   Float32[0]]

# opt = SGD(lr=1f-2)
opt = Adam(lr=1f-3)

for epoch in 1:5000
    total_loss = 0f0
    for (x, y) in zip(xor_inputs, xor_targets)
        total_loss += train!(model, x, Tensor(y), opt)
    end
    if total_loss < 1f-4
        println("Converged at epoch $epoch")
        break
    end
end

# fit!(model, Tensor(xor_inputs), Tensor(xor_targets), opt, 5000, mse_loss)

for (x, y) in zip(xor_inputs, xor_targets)
    pred = model(x)
    println("input=$x  target=$y  pred=$(round.(pred.data, digits=3))")
end