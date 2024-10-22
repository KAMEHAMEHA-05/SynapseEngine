module Exceptions
export DimensionMismatchException, raise_dimension_mismatch, raise_indexoutofbounds

struct DimensionMismatchException <: Exception
    expected::Vector{Int}
    actual::Vector{Int}
    msg::String
end

struct IndexOutofBoundsException <: Exception
    expected::Vector{Int}
    actual::Vector{Int}
    msg::String
end

# function Base.showerror(io::IO, e::DimensionMismatchException)
#     print(io, "DimensionMismatchException => "," Expected: ", e.expected, "(operator 1), Got: ", e.actual, "(operator 2)")
# end

function raise_dimension_mismatch(expected::Vector{Int}, actual::Vector{Int})
    # println("Expected shape: ", expected)
    # println("Actual shape: ", actual)
    if !isequal(expected, actual)
        msg = "Shapes of Operator_1 and Operator_2 do not match."
        throw(DimensionMismatchException(expected, actual, msg))
    end
end

function raise_indexoutofbounds(expected::Vector{Int}, actual::Vector{Int})
    for (ind,(exp, act)) in enumerate(zip(expected, actual))
        if(act>exp)
            msg = "Index $ind (actual) out of bounds ($act) (expected)."
            throw(IndexOutofBoundsException(expected, actual, msg))
        elseif (act<1)
            msg = "Index $ind below 1."
            throw(IndexOutofBoundsException(expected, actual, msg))
        end
    end
end

function raise_indexoutofbounds(expected::Int, actual::Int)
    if(actual>expected)
        msg = "Index ($actual) out of bounds ($expected)."
        throw(IndexOutofBoundsException([expected], [actual], msg))
    elseif (actual<1)
        msg = "Index $actual below 1."
        throw(IndexOutofBoundsException([expected], [actual], msg))
    end
end

end