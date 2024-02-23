"""
    fused_increment(u, dt, sc::SparseCoeffs{S}, tend, ::Val{i}) where {i, S}

Returns a broadcasted object in the form

    `u + ∑ⱼ dt scⱼ  tendⱼ for j in 1:i`
    or
    `u + ∑ⱼ dt scᵢⱼ tendⱼ for j in 1:(i-1)`

depending on the dimensions of the coefficients in `sc`. The
broadcasted object drops zero coefficients from the expression
at compile-time using the mask `mask` (made from `SparseCoeffs(coeffs)`).

Returns `u` when `i ≤ 1` or if the mask is all true (all coefficients are zero).


# Example loops that this is fusing:

For 2D coefficients case
```julia
for j in 1:(i - 1)
    iszero(a_imp[i, j]) && continue
    @. U += dt * a_imp[i, j] * T_imp[j]
end

For 1D coefficients case
```julia
for j in 1:s
    iszero(b_exp[j]) && continue
    @. temp += dt * b_exp[j] * T_lim[j]
end
```
"""
function fused_increment end

@inline fused_increment(u, dt, sc, tend, v) = fused_increment(u, dt, sc, tend, v, get_S(sc))

# ij case (S::NTuple{2})
@generated function fused_increment(u, dt, sc::T, tend, ::Val{i}, ::NTuple{2}) where {i, S, T <: SparseCoeffs{S}}
    i ≤ 1 && return :(u)
    terms = []
    for j in 1:(i - 1)
        zero_coeff(T, i, j) && continue
        push!(terms, :(Base.Broadcast.broadcasted(*, dt * sc[$i, $j], tend[$j])))
    end
    isempty(terms) && return :(u)
    expr = Meta.parse(join(terms, ","))
    if length(terms) == 1
        return :(Base.Broadcast.broadcasted(+, u, $expr))
    else
        return :(Base.Broadcast.broadcasted(+, u, $expr...))
    end
end

# j case (S::NTuple{1})
@generated function fused_increment(u, dt, sc::T, tend, ::Val{s}, ::NTuple{1}) where {s, S, T <: SparseCoeffs{S}}
    all(j -> zero_coeff(T, j), 1:s) && return :(u)
    terms = []
    for j in 1:s
        zero_coeff(T, j) && continue
        push!(terms, :(Base.Broadcast.broadcasted(*, dt * sc[$j], tend[$j])))
    end
    expr = Meta.parse(join(terms, ","))
    if length(terms) == 1
        return :(Base.Broadcast.broadcasted(+, u, $expr))
    else
        return :(Base.Broadcast.broadcasted(+, u, $expr...))
    end
end

"""
    fused_increment!(u, dt, sc, tend, v)

Calls [`fused_increment`](@ref) and materializes
a broadcast expression in the form:

    `@. u += ∑ⱼ dt scⱼ tendⱼ`

In the edge case (coeffs are zero, `j` range is empty),
this lowers to `nothing` (no-op)
"""
function fused_increment!(u, dt, sc, tend, v)
    bc = fused_increment(u, dt, sc, tend, v)
    if bc isa Base.Broadcast.Broadcasted # Only material if not trivial assignment
        Base.Broadcast.materialize!(u, bc)
    end
    nothing
end

"""
    assign_fused_increment!(U, u, dt, sc, tend, v)

Calls [`fused_increment`](@ref) and materializes
a broadcast expression in the form:

    `@. u += ∑ⱼ dt scⱼ tendⱼ`

In the edge case (coeffs are zero, `j` range is empty),
this lowers to

    `@. U = u`
"""
function assign_fused_increment!(U, u, dt, sc, tend, v)
    bc = fused_increment(u, dt, sc, tend, v)
    Base.Broadcast.materialize!(U, bc)
    return nothing
end
