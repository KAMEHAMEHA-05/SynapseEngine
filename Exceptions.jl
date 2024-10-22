module Exceptions
export DimensionMismatchException, raise_dimension_mismatch

struct DimensionMismatchException <: Exception
    expected::Vector{Int}
    actual::Vector{Int}
    msg::String
end

function Base.showerror(io::IO, e::DimensionMismatchException)
    print(io, "DimensionMismatchException => "," Expected: ", e.expected, "(operator 1), Got: ", e.actual, "(operator 2)")
end

function raise_dimension_mismatch(expected::Vector{Int}, actual::Vector{Int})
    msg = "Shapes of Operator_1 and Operator_2 do not match."
    throw(DimensionMismatchException(expected, actual, msg))
end

end