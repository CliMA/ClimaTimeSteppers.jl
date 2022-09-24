lower_plus_diagonal(matrix::T) where {T} =
    T(LinearAlgebra.LowerTriangular(matrix))
diagonal(matrix::T) where {T} = T(LinearAlgebra.Diagonal(matrix))
lower(matrix) = lower_plus_diagonal(matrix) - diagonal(matrix)

lower_triangular_inv(matrix::T) where {T} =
    T(inv(LinearAlgebra.LowerTriangular(matrix)))

to_enumerated_rows(x) = x
function to_enumerated_rows(matrix::AbstractMatrix)
    rows = tuple(1:size(matrix, 1)...)
    nonzero_indices = map(i -> findall(matrix[i, :] .!= 0), rows)
    enumerated_rows = map(
        i -> tuple(zip(nonzero_indices[i], matrix[i, nonzero_indices[i]])...),
        rows,
    )
    return enumerated_rows
end

linear_combination_terms(enumerated_row, vectors) =
    map(((j, val),) -> broadcasted(*, val, vectors[j]), enumerated_row)
function linear_combination(enumerated_row, vectors)
    length(enumerated_row) == 0 && return nothing
    terms = linear_combination_terms(enumerated_row, vectors)
    length(enumerated_row) == 1 && return terms[1]
    return broadcasted(+, terms...)
end

@generated foreachval(f::F, ::Val{N}) where {F, N} =
    quote
        Base.@nexprs $N i -> f(Val(i))
        return nothing
    end
