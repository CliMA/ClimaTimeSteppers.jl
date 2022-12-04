
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
struct SparseContainer{SIM, T}
    data::T
    function SparseContainer(
            compressed_data::T,
            sparse_index_map::Tuple
        ) where {N, ET, T <: NTuple{N, ET}}
        @assert all(map(x-> eltype(compressed_data) .== typeof(x), compressed_data))
        return new{sparse_index_map, T}(compressed_data)
    end
end

Base.parent(sc::SparseContainer) = sc.data
sc_eltype(::Type{NTuple{N, T}}) where {N, T} = T
sc_eltype(::SparseContainer{SIM, T}) where {SIM, T} = sc_eltype(T)
@inline function Base.getindex(sc::SparseContainer, i::Int)
    return _getindex_sparse(sc, Val(i))::sc_eltype(sc)
end
@generated function _getindex_sparse(sc::SparseContainer{SIM}, ::Val{i}) where {SIM, i}
    j = findfirst(k -> k == i, SIM)
    j == nothing && error("No index $i found in sparse index map $(SIM)")
    return :(sc.data[$j])
end
