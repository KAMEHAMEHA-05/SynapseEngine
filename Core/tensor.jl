# Core/tensor.jl

abstract type BackwardOp end

mutable struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    grad::Union{Nothing, A}
    parents::Vector{Tensor}
    backward::Union{Nothing, BackwardOp}
end

Tensor(data::AbstractArray{T,N}) where {T,N} =
    Tensor{T,N,typeof(data)}(data, nothing, [], nothing)

Base.size(t::Tensor) = size(t.data)
Base.getindex(t::Tensor, I...) = t.data[I...]
Base.setindex!(t::Tensor, v, I...) = (t.data[I...] = v)
Base.IndexStyle(::Type{<:Tensor}) = IndexStyle(Array)


function ensure_grad!(t::Tensor)
    if t.grad === nothing
        t.grad = similar(t.data)   
        fill!(t.grad, 0)           
    end
end