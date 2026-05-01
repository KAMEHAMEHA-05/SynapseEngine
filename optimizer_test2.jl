include("BaseTest3.jl")

using Random
Random.seed!(42)

T = Float32
l1 = Layer(T, 2, 4, xavier_init, ReLU)
l2 = Layer(T, 4, 1, xavier_init, Linear)
model = Model([l1, l2], x -> l2(l1(x)))

opt = Adam(lr=1f-3)

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

for (x, y) in zip(xor_inputs, xor_targets)
    pred = model(x)
    println("input=$x  target=$y  pred=$(round.(pred.data, digits=3))")
end

println("l1.w.grad = ", l1.w.grad)
println("l1.b.grad = ", l1.b.grad)
println("l2.w.grad = ", l2.w.grad)
println("l2.b.grad = ", l2.b.grad)

println("l1.w.data = ", l1.w.data)
println("l2.w.data = ", l2.w.data)