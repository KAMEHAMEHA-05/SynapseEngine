mutable struct Tensor
    ndims::Int
    shape::Vector{Int}
    data::Vector{Float64}

    function Tensor(data::Vector{Float64}, shape::Vector{Int})
        ndims = maximum([length(shape), 0])
        return new(ndims, shape, data)
    end
end


function valueAt(t::Tensor, index::Vector{Int})
    flat_index = 1
    for (ind, factor) in enumerate(index)
        jump=1
        for x in t.shape[1:ind-1]
            jump*=x
        end
        flat_index+=factor*jump
    end
    return t.data[flat_index]
end

arr = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
shape = [2,2,2]
tensor = Tensor(arr, shape)
index = [1, 1, 1]
print(valueAt(tensor, index))

