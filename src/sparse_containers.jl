
"""
    SparseContainer(compressed_data, sparse_index_map)

A compact container that allows dense-like
indexing into a sparse, uniform, container.

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
@test v[7] == ones(3) .* 4
```
"""
struct SparseContainer{ET, SIM, T}
    data::T
    function SparseContainer(compressed_data::T, sparse_index_map::Tuple) where {T}
        @assert all(map(x-> eltype(compressed_data) .== typeof(x), compressed_data))
        return new{eltype(compressed_data), sparse_index_map, T}(compressed_data)
    end
end

Base.parent(sc::SparseContainer) = sc.data
@inline function Base.getindex(sc::SparseContainer{ET}, i::Int) where {ET}
    return _getindex_sparse(sc, Val(i))::ET
end
@generated function _getindex_sparse(sc::SparseContainer{ET,SIM}, ::Val{i})::ET where {ET, SIM, i}
    j = findfirst(k -> k == i, SIM)
    j == nothing && error("No index $i found in sparse index map $(SIM)")
    return :(sc.data[$j])
end
