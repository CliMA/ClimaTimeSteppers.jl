"""
    fused_increment(u, dt, sc::SparseCoeffs, tend, v)

Return a lazy broadcasted expression equivalent to

    `u + ∑ⱼ dt scⱼ  tendⱼ for j in 1:i`
    or
    `u + ∑ⱼ dt scᵢⱼ tendⱼ for j in 1:(i-1)`

depending on the dimensions of the coefficients in `sc`. Zero
coefficients are dropped at compile-time via `SparseCoeffs`.

Implemented as `broadcasted(+, u, fused_raw_increment(dt, sc, tend, v))`;
when all coefficients are zero the `NullBroadcasted` returned by
[`fused_raw_increment`](@ref) makes the result equivalent to `u`.


# Examples

Fused 2D coefficients case:
```julia
for j in 1:(i - 1)
    iszero(a_imp[i, j]) && continue
    @. U += dt * a_imp[i, j] * T_imp[j]
end
```

Fused 1D coefficients case:
```julia
for j in 1:s
    iszero(b_exp[j]) && continue
    @. temp += dt * b_exp[j] * T_lim[j]
end
```
"""
function fused_increment end

@inline fused_increment(u, dt, sc::SparseCoeffs, tend, v) =
    Base.Broadcast.broadcasted(+, u, fused_raw_increment(dt, sc, tend, v))

# =================================================== ij case (S::NTuple{2})
# recursion: _rfused_increment_ij always returns a Tuple
@inline _rfused_increment_ij(js::Tuple{}, i, u, dt, sc::SparseCoeffs, tend) = ()

@inline _rfused_increment_ij(
    js::Tuple{Int},
    i,
    u,
    dt,
    sc::T,
    tend,
) where {T <: SparseCoeffs} = _rfused_increment_ij(js[1], i, u, dt, sc, tend)

@inline _rfused_increment_ij(j::Int, i, u, dt, sc::T, tend) where {T <: SparseCoeffs} =
    if zero_coeff(T, i, j)
        ()
    else
        (Base.Broadcast.broadcasted(*, dt * sc[i, j], tend[j]),)
    end

@inline _rfused_increment_ij(js::Tuple, i, u, dt, sc::T, tend) where {T <: SparseCoeffs} = (
    _rfused_increment_ij(first(js), i, u, dt, sc, tend)...,
    _rfused_increment_ij(Base.tail(js), i, u, dt, sc, tend)...,
)



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

@inline _rfused_increment_j(js::Tuple, u, dt, sc::T, tend) where {T <: SparseCoeffs} = (
    _rfused_increment_j(first(js), u, dt, sc, tend)...,
    _rfused_increment_j(Base.tail(js), u, dt, sc, tend)...,
)



# Helper to transform recursive unpacked tuple elements into a single nested `+` broadcast object.
@inline _sum_broadcasted(x) = x
@inline _sum_broadcasted(x, y...) = Base.Broadcast.broadcasted(+, x, y...)

"""
    fused_raw_increment(dt, sc::SparseCoeffs, tend, v)

Return a lazy broadcasted expression equivalent to

    `∑ⱼ dt scⱼ  tendⱼ for j in 1:i`
    or
    `∑ⱼ dt scᵢⱼ tendⱼ for j in 1:(i-1)`

depending on the dimensions of the coefficients in `sc`. Zero
coefficients are dropped at compile-time via `SparseCoeffs`.

Return `NullBroadcasted()` when the index range is empty or all
coefficients are zero.
"""
function fused_raw_increment end

# Cast `dt` to a float at the entry, mirroring the contract of
# `fused_increment!` / `assign_fused_increment!`. Without this, callers
# passing a non-float `dt` (e.g. `ClimaUtilities.ITime`) blow up at the
# eager `dt * sc[i, j]` inside the recursion in `_rfused_increment_*`
# rather than at materialize time.
@inline fused_raw_increment(dt, sc::SparseCoeffs{S}, tend, v) where {S} =
    fused_raw_increment(float(dt), sc, tend, v, Val(length(S)))

# =================================================== ij case (S::NTuple{2})
# top-level function
@inline function _fused_raw_increment_ij(
    js::Tuple,
    i,
    dt,
    sc::T,
    tend,
) where {T <: SparseCoeffs}
    return if all(j -> zero_coeff(T, i, j), js)
        NullBroadcasted()
    else
        _sum_broadcasted(_rfused_increment_ij(js, i, nothing, dt, sc, tend)...)
    end
end

# top-level function, if the tuple is empty, return NullBroadcasted()
@inline _fused_raw_increment_ij(js::Tuple{}, i, dt, sc::SparseCoeffs, tend) =
    NullBroadcasted()

# wrapper ij case (S::NTuple{2})
@inline fused_raw_increment(
    dt,
    sc::SparseCoeffs,
    tend,
    ::Val{i},
    ::Val{2},
) where {i} = _fused_raw_increment_ij(ntuple(j -> j, Val(i - 1)), i, dt, sc, tend)

# =================================================== j case (S::NTuple{1})
# top-level function
@inline function _fused_raw_increment_j(
    js::Tuple,
    dt,
    sc::T,
    tend,
) where {T <: SparseCoeffs}
    return if all(j -> zero_coeff(T, j), js)
        NullBroadcasted()
    else
        _sum_broadcasted(_rfused_increment_j(js, nothing, dt, sc, tend)...)
    end
end

# top-level function, if the tuple is empty, return NullBroadcasted()
@inline _fused_raw_increment_j(js::Tuple{}, dt, sc::SparseCoeffs, tend) = NullBroadcasted()

# wrapper j case (S::NTuple{1})
@inline fused_raw_increment(
    dt,
    sc::SparseCoeffs,
    tend,
    ::Val{s},
    ::Val{1},
) where {s} = _fused_raw_increment_j(ntuple(i -> i, s), dt, sc, tend)

"""
    fused_increment!(u, dt, sc, tend, v)

Materialize the fused increment `∑ⱼ dt scⱼ tendⱼ` in place:

    `@. u += ∑ⱼ dt scⱼ tendⱼ`

No-op when all coefficients are zero.
"""
@inline function fused_increment!(u, dt, sc, tend, v)
    inc = fused_raw_increment(dt, sc, tend, v)
    if !(inc isa NullBroadcasted)
        @. u += inc
    end
    nothing
end

"""
    assign_fused_increment!(U, u, dt, sc, tend, v)

Materialize the fused increment into `U`:

    `@. U = u + ∑ⱼ dt scⱼ tendⱼ`

When all coefficients are zero, this reduces to `@. U = u` (via
`NullBroadcasted`).
"""
@inline function assign_fused_increment!(U, u, dt, sc, tend, v)
    inc = fused_raw_increment(dt, sc, tend, v)
    @. U = u + inc
    return nothing
end

"""
    assign_with_increments!(U, base, inc_exp, inc_imp)

Assign `U .= base + inc_exp + inc_imp`, safely handling the case where either
`inc_exp` or `inc_imp` (or both) is a `NullBroadcasted`.

The standard `@. U = base + inc_exp + inc_imp` expands to nested calls to
`Base.broadcasted`, which routes through `combine_styles` / `result_style`.
Because `NullBroadcasted <: AbstractBroadcasted` (not `<: BroadcastStyle`),
combining it with a real `Broadcasted` argument triggers a `MethodError` for
`result_style(::NullBroadcasted)`. This helper avoids that by dispatching on
the concrete types of both increments at compile time.
"""
@inline function assign_with_increments!(
    U,
    base,
    inc_exp::NullBroadcasted,
    inc_imp::NullBroadcasted,
)
    @. U = base
end
@inline function assign_with_increments!(U, base, inc_exp::NullBroadcasted, inc_imp)
    @. U = base + inc_imp
end
@inline function assign_with_increments!(U, base, inc_exp, inc_imp::NullBroadcasted)
    @. U = base + inc_exp
end
@inline function assign_with_increments!(U, base, inc_exp, inc_imp)
    @. U = base + inc_exp + inc_imp
end
