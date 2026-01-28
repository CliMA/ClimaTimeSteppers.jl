"""
    fused_increment(u, dt, sc::SparseCoeffs, tend, ::Val{i}) where {i}

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

# =================================================== ij case (S::NTuple{2})
# recursion: _rfused_increment_j always returns a Tuple
@inline _rfused_increment_ij(js::Tuple{}, i, u, dt, sc::SparseCoeffs, tend) = ()

@inline _rfused_increment_ij(js::Tuple{Int}, i, u, dt, sc::T, tend) where {T <: SparseCoeffs} =
    _rfused_increment_ij(js[1], i, u, dt, sc, tend)

@inline _rfused_increment_ij(j::Int, i, u, dt, sc::T, tend) where {T <: SparseCoeffs} =
    if zero_coeff(T, i, j)
        ()
    else
        (Base.Broadcast.broadcasted(*, dt * sc[i, j], tend[j]),)
    end

@inline _rfused_increment_ij(js::Tuple, i, u, dt, sc::T, tend) where {T <: SparseCoeffs} =
    (_rfused_increment_ij(first(js), i, u, dt, sc, tend)..., _rfused_increment_ij(Base.tail(js), i, u, dt, sc, tend)...)

# top-level function
@inline function _fused_increment_ij(js::Tuple, i, u, dt, sc::T, tend) where {T <: SparseCoeffs}
    return if all(j -> zero_coeff(T, i, j), js)
        u
    else
        Base.Broadcast.broadcasted(
            +,
            u,
            _rfused_increment_ij(js, i, u, dt, sc, tend)..., # recurse...
        )
    end
end

# top-level function, if the tuple is empty, just return u
@inline _fused_increment_ij(js::Tuple{}, i, u, dt, sc::SparseCoeffs, tend) = u

# wrapper ij case (S::NTuple{2})
@inline fused_increment(u, dt, sc::SparseCoeffs, tend, ::Val{i}, ::NTuple{2}) where {i} =
    _fused_increment_ij(ntuple(j -> j, Val(i - 1)), i, u, dt, sc, tend)

# =================================================== j case (S::NTuple{1})
# recursion: _rfused_increment_j always returns a Tuple
@inline _rfused_increment_j(js::Tuple{}, u, dt, sc::SparseCoeffs, tend) = ()

@inline _rfused_increment_j(js::Tuple{Int}, u, dt, sc::T, tend) where {T <: SparseCoeffs} =
    _rfused_increment_j(js[1], u, dt, sc, tend)

@inline _rfused_increment_j(j::Int, u, dt, sc::T, tend) where {T <: SparseCoeffs} =
    if zero_coeff(T, j)
        ()
    else
        (Base.Broadcast.broadcasted(*, dt * sc[j], tend[j]),)
    end

@inline _rfused_increment_j(js::Tuple, u, dt, sc::T, tend) where {T <: SparseCoeffs} =
    (_rfused_increment_j(first(js), u, dt, sc, tend)..., _rfused_increment_j(Base.tail(js), u, dt, sc, tend)...)

# top-level function
@inline function _fused_increment_j(js::Tuple, u, dt, sc::T, tend) where {T <: SparseCoeffs}
    return if all(j -> zero_coeff(T, j), js)
        u
    else
        Base.Broadcast.broadcasted(
            +,
            u,
            _rfused_increment_j(js, u, dt, sc, tend)..., # recurse...
        )
    end
end

# top-level function, if the tuple is empty, just return u
@inline _fused_increment_j(js::Tuple{}, u, dt, sc::SparseCoeffs, tend) = u

# wrapper j case (S::NTuple{1})
@inline fused_increment(u, dt, sc::SparseCoeffs, tend, ::Val{s}, ::NTuple{1}) where {s} =
    _fused_increment_j(ntuple(i -> i, s), u, dt, sc, tend)

"""
    fused_increment!(u, dt, sc, tend, v)

Calls [`fused_increment`](@ref) and materializes
a broadcast expression in the form:

    `@. u += ∑ⱼ dt scⱼ tendⱼ`

In the edge case (coeffs are zero, `j` range is empty),
this lowers to `nothing` (no-op)
"""
@inline function fused_increment!(u, dt, sc, tend, v)
    bc = fused_increment(u, float(dt), sc, tend, v)
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
@inline function assign_fused_increment!(U, u, dt, sc, tend, v)
    bc = fused_increment(u, float(dt), sc, tend, v)
    Base.Broadcast.materialize!(U, bc)
    return nothing
end

"""
    fused_increment_row!(U, dt, sc, tend, ::Val{n}, ::Val{row})

Calls a row-shifted fused implicit increment and materializes
a broadcast expression in the form:

    `@. U += ∑ⱼ dt a_imp[row, j] tendⱼ`   for `j = 1:(n-1)`

This is equivalent to [`fused_increment!`](@ref) for stage `n`,
but with coefficients taken from tableau row `row` instead of row `n`.

In the edge case (coeffs are zero, `j` range is empty),
this lowers to `nothing` (no-op)
"""
@inline function fused_increment_row!(U, dt, sc::SparseCoeffs, tend,
                                      ::Val{n}, ::Val{row}) where {n,row}
    # same range as Val(n): j = 1:(n-1), but row = row
    bc = _fused_increment_ij(ntuple(j -> j, Val(n - 1)), row, U, float(dt), sc, tend)
    bc isa Base.Broadcast.Broadcasted && Base.Broadcast.materialize!(U, bc)
    nothing
end

"""
    fused_initialize_Newton!(U, dt, sc, tend, c, ::Val{i})

Initialize the Newton iterate for stage `i` by replacing the implicit
tableau contribution from row `i` with the contribution from row `i-1`,
and extrapolating the previous stage tendency forward to the base point
of stage `i`.

Performs:

    U ← U
        - ∑ⱼ₌₁^{i-1} dt * a_imp[i,   j] * T_imp[j]
        + ∑ⱼ₌₁^{i-1} dt * a_imp[i-1, j] * T_imp[j]
        + (c[i] - c[i-1]) * dt * T_imp[i-1]
"""
@inline function fused_initialize_Newton!(U, dt, sc::SparseCoeffs, tend, c, ::Val{i}) where {i}

    # subtract row i (range 1:(i-1))
    fused_increment!(U, -dt, sc, tend, Val(i))

    # add row i-1 over the SAME range (1:(i-1))
    fused_increment_row!(U, dt, sc, tend, Val(i), Val(i-1))

    # extrapolate last tendency to the new stage base point
    last_index = length(c) - 1
    i == 2 && @. U += (c[2] - c[1]) * float(dt) * tend[last_index]  
    i > 2 && @. U += (c[i] - c[i - 1]) * float(dt) * tend[i - 1]

    nothing
end

