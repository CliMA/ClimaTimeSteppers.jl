
"""
    SparseContainer(compressed_data, sparse_index_map)

A compact container that allows dense-like
indexing into a sparse container.

# Example

```julia
using Test
a1 = ones(3) .* 1
a2 = ones(3) .* 2
a3 = ones(3) .* 3
a4 = ones(3) .* 4
v = SparseContainer((a1,a2,a3,a4), (1,3,5,7))
@test v[1] == ones(3) .* 1
@test v[3] == ones(3) .* 2
@test v[5] == ones(3) .* 3
@test v[7] == ones(3) .* 4```
"""
struct SparseContainer{T,SIM}
    data::T
    function SparseContainer(compressed_data::T, sparse_index_map::Tuple) where {T}
        return new{T, sparse_index_map}(compressed_data)
    end
end

Base.parent(sc::SparseContainer) = sc.data
@inline Base.getindex(sc::SparseContainer, i::Int) = _getindex_sparse(sc, Val(i))
@generated function _getindex_sparse(sc::SparseContainer{T,SIM}, ::Val{i}) where {T, SIM, i}
    j = findfirst(k -> k == i, SIM)
    j == nothing && error("No index $i found in sparse index map $(SIM)")
    return :(sc.data[$j])
end
