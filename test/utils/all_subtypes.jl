using InteractiveUtils: subtypes

"""
    all_subtypes(T)

Recursively collect all concrete (non-abstract) subtypes of `T`.
"""
all_subtypes(::Type{T}) where {T} =
    isabstracttype(T) ? vcat(all_subtypes.(subtypes(T))...) : [T]
