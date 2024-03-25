"""
    SparseTuple(entries, indices)
    SparseTuple(f, indices)
    SparseTuple()

A statically sized vector-like object that can only be accessed at certain
indices without a `BoundsError` being thrown. The `entries` may be specified
directly, or through a function `f(index)`. If no arguments are provided, the
result is an empty `SparseTuple` that cannot be accessed at any index.

A `SparseTuple` can be used to represent an ordered set with a fixed size and a
concrete element type that is embedded within a larger `Tuple` of arbitrary size
and element type. The full `Tuple` can be reconstructed using `dense_tuple`.
"""
struct SparseTuple{P <: NTuple{<:Any, Pair}}
    pairs::P
end
function SparseTuple(entries, indices)
    length(entries) != length(indices) && error("Number of entries does not match number of indices")
    any(index -> count(==(index), indices) > 1, indices) && error("Indices are not unique")
    return SparseTuple(Tuple(map(=>, indices, entries)))
end
SparseTuple(f::F, indices) where {F <: Function} = SparseTuple(map(f, indices), indices)
SparseTuple() = SparseTuple(())

function Base.getindex(sparse_tuple::SparseTuple, index)
    pair_index = findfirst(pair -> pair[1] == index, sparse_tuple.pairs)
    isnothing(pair_index) && throw(BoundsError(sparse_tuple, index))
    return sparse_tuple.pairs[pair_index][2]
end

"""
    dense_tuple(sparse_tuple, tuple_length, default_entry)

Turns a `SparseTuple` into an `NTuple{tuple_length}`, setting the value at each
inaccessible index to the given `default_entry`.
"""
dense_tuple(sparse_tuple, tuple_length, default_entry) =
    ntuple(tuple_length) do index
        pair_index = findfirst(pair -> pair[1] == index, sparse_tuple.pairs)
        isnothing(pair_index) ? default_entry : sparse_tuple.pairs[pair_index][2]
    end

"""
    sparse_matrix_rows(matrix_entries, row_indices, col_indices)

Turns the entries at the given row and column indices within a matrix into a
`SparseTuple` of matrix rows, where each row is itself a `SparseTuple` that only
stores nonzero entries.
"""
function sparse_matrix_rows(matrix_entries, row_indices, col_indices)
    size(matrix_entries) != (length(row_indices), length(col_indices)) &&
        error("Numbers of row and column indices do not match matrix shape")
    sparse_rows = map(eachrow(matrix_entries)) do row_vector
        nonzero_indices = findall(!iszero, row_vector)
        SparseTuple(row_vector[nonzero_indices], col_indices[nonzero_indices])
    end
    return SparseTuple(sparse_rows, row_indices)
end

"""
    is_accessible(sparse_tuple)

Checks whether the given `SparseTuple` has any accessible indices.
"""
is_accessible(sparse_tuple) = !isempty(sparse_tuple.pairs)

"""
    sparse_broadcasted_dot(sparse_tuple, vector)

A `Base.AbstractBroadcasted` that represents `dot(sparse_tuple, vector)`, where
the first argument is a `SparseTuple` and the second is any vector-like object
that can be accessed at the same indices as `SparseTuple`. If `sparse_tuple` has
no accessible indices, the result is an `EmptySum()`.

Since the number of accessible indices in `sparse_tuple` is inferrable, this
function will be type-stable as long as `sparse_tuple` and `vector` have
inferrable element types at those indices.
"""
sparse_broadcasted_dot(sparse_tuple, vector) =
    broadcasted_sum(map(pair -> Base.broadcasted(*, pair[2], vector[pair[1]]), sparse_tuple.pairs))

broadcasted_sum(summands) =
    if isempty(summands)
        EmptySum()
    elseif length(summands) == 1
        summands[1]
    else
        Base.broadcasted(+, summands...)
    end

"""
    EmptySum()

A `Base.AbstractBroadcasted` that represents `+()`. An `EmptySum()` cannot be
materialized, but it can be added to, subtracted from, or multiplied by any
value in a broadcast expression without incurring a runtime performance penalty.
"""
struct EmptySum <: Base.AbstractBroadcasted end
Base.broadcastable(empty_sum::EmptySum) = empty_sum

struct EmptySumStyle <: Base.BroadcastStyle end
Base.BroadcastStyle(::Type{<:EmptySum}) = EmptySumStyle()
Base.BroadcastStyle(style::EmptySumStyle, ::Base.BroadcastStyle) = style
Base.broadcasted(::EmptySumStyle, ::typeof(+), summands...) =
    broadcasted_sum(filter(summand -> !(summand isa EmptySum), summands))
Base.broadcasted(::EmptySumStyle, ::typeof(-), arg) = arg
Base.broadcasted(::EmptySumStyle, ::typeof(-), arg, ::EmptySum) = arg
Base.broadcasted(::EmptySumStyle, ::typeof(*), _...) = EmptySum()
