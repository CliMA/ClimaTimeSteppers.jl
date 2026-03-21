export AbstractClimaODEFunction
export ClimaODEFunction, ForwardEulerODEFunction

abstract type AbstractClimaODEFunction end

"""
    ClimaODEFunction(; T_imp!, [dss!], [cache!], [cache_imp!])
    ClimaODEFunction(; T_exp!, [T_lim!], [T_imp!], [lim!], [dss!], [cache!], [cache_imp!])
    ClimaODEFunction(; T_exp_T_lim!, [T_imp!], [lim!], [dss!], [cache!], [cache_imp!])

Container for all tendency and auxiliary functions used by IMEX and Rosenbrock
time-stepping algorithms. Tendencies set to `nothing` are skipped, avoiding
unnecessary allocations.

# Keyword Arguments

**Tendency functions** (at least one must be provided):
- `T_exp!(du, u, p, t)`: explicit tendency (not limited)
- `T_lim!(du, u, p, t)`: explicit tendency passed through the limiter
- `T_exp_T_lim!(du_exp, du_lim, u, p, t)`: fused alternative to separate `T_exp!`/`T_lim!`
- `T_imp!`: implicit tendency — typically an [`ClimaTimeSteppers.ODEFunction`](@ref) carrying Jacobian info
- `T_imp_subproblem!`: optional implicit tendency for a **preconditioning** Newton
   solve executed before the main `T_imp!` solve.  The subproblem mutates `u` in
   place, and its result becomes the initial guess for the main solve — it does
   **not** define an additive tendency.  Setting `T_imp_subproblem! = T_imp!`
   applies the implicit equation twice (compounding), giving different results
   from the standard single-solve path.

**Auxiliary functions** (default to no-ops):
- `lim!(u, p, t, u_ref)`: limiter applied after incrementing `u` from `u_ref` by `T_lim!`
- `dss!(u, p, t)`: direct stiffness summation (spectral element continuity)
- `cache!(u, p, t)`: update the parameter cache `p` to reflect state `u`
- `cache_imp!(u, p, t)`: update cache components needed by `T_imp!` (defaults to `cache!`)
- `initialize_subproblem!(u, p, γdt)`: set up the subproblem before the Newton solve

Internally, `T_exp!` and `T_lim!` are merged into a single `T_exp_T_lim!`
at construction time.
"""
struct ClimaODEFunction{TEL, TIS, TI, L, D, IS, C, CI} <: AbstractClimaODEFunction
    T_exp_T_lim!::TEL
    T_imp_subproblem!::TIS
    T_imp!::TI
    lim!::L
    dss!::D
    initialize_subproblem!::IS
    cache!::C
    cache_imp!::CI
    _has_lim::Bool  # true when the limiter path should be used
    function ClimaODEFunction(;
        T_exp_T_lim! = nothing,
        T_lim! = nothing,
        T_exp! = nothing,
        T_imp_subproblem! = nothing,
        T_imp! = nothing,
        lim! = Returns(nothing),
        dss! = Returns(nothing),
        initialize_subproblem! = Returns(nothing),
        cache! = Returns(nothing),
        cache_imp! = cache!,
    )
        # Normalize T_exp!/T_lim! into fused T_exp_T_lim!
        if !isnothing(T_exp_T_lim!)
            @assert isnothing(T_exp!) "`T_exp_T_lim!` was passed, `T_exp!` must be `nothing`"
            @assert isnothing(T_lim!) "`T_exp_T_lim!` was passed, `T_lim!` must be `nothing`"
            _has_lim = true
        elseif !isnothing(T_exp!) && !isnothing(T_lim!)
            # Wrap separate T_exp! + T_lim! into fused form
            T_exp_T_lim! = (du_exp, du_lim, u, p, t) -> begin
                T_exp!(du_exp, u, p, t)
                T_lim!(du_lim, u, p, t)
            end
            _has_lim = true
        elseif !isnothing(T_exp!)
            # Explicit-only: wrap T_exp! into fused form, T_lim output unused
            T_exp_T_lim! = (du_exp, _du_lim, u, p, t) -> T_exp!(du_exp, u, p, t)
            _has_lim = false
        elseif !isnothing(T_lim!)
            # Limiter-only: wrap T_lim! into fused form
            T_exp_T_lim! = (_du_exp, du_lim, u, p, t) -> T_lim!(du_lim, u, p, t)
            _has_lim = true
        else
            _has_lim = false
        end
        args = (
            T_exp_T_lim!,
            T_imp_subproblem!,
            T_imp!,
            lim!,
            dss!,
            initialize_subproblem!,
            cache!,
            cache_imp!,
        )
        return new{typeof.(args)...}(args..., _has_lim)
    end
end

has_T_exp(f::ClimaODEFunction) = !isnothing(f.T_exp_T_lim!)
has_T_lim(f::ClimaODEFunction) = f._has_lim

"""Called by `init` to set up the initial cache state. No-op for non-Clima functions."""
initialize_function!(f, u0, p, t0) = nothing
initialize_function!(f::ClimaODEFunction, u0, p, t0) =
    isnothing(f.cache!) || f.cache!(u0, p, t0)

"""
    ForwardEulerODEFunction(f; jac_prototype, Wfact, tgrad)

An ODE function whose call signature is `f(un, u, p, t, dt)`, computing a
forward-Euler-style update `un .= u .+ dt * tendency(u, p, t)`.

# Arguments
- `f`: callable with signature `f(un, u, p, t, dt)`

# Keyword Arguments
- `jac_prototype`: prototype matrix for the Jacobian
- `Wfact`: function `Wfact(W, u, p, dtγ, t)` computing ``W = J \\Delta t \\gamma - I``
- `tgrad`: function `tgrad(∂f∂t, u, p, t)` for the explicit time derivative
"""
struct ForwardEulerODEFunction{F, J, W, T}
    f::F
    jac_prototype::J
    Wfact::W
    tgrad::T
end
ForwardEulerODEFunction(f; jac_prototype = nothing, Wfact = nothing, tgrad = nothing) =
    ForwardEulerODEFunction(f, jac_prototype, Wfact, tgrad)
(f::ForwardEulerODEFunction{F})(un, u, p, t, dt) where {F} = f.f(un, u, p, t, dt)

"""
    OffsetODEFunction(f, α, β, γ, x)

Internal wrapper used by multirate methods. Evaluates `f` with a time offset
and adds a constant forcing term:

```math
f(u, p, \\alpha + \\beta t) + \\gamma \\cdot x
```

Supports 3-arg (out-of-place), 4-arg, 5-arg (`α`), and 6-arg (`α, β`)
in-place call forms. The fields `α`, `β`, `γ`, and `x` are mutable so that
multirate outer solvers can update them between stages.
"""
mutable struct OffsetODEFunction{F, S, A}
    f::F
    α::S
    β::S
    γ::S
    x::A
end
function OffsetODEFunction(f, α, β, γ, x)
    α, β, γ = promote(α, β, γ)
    OffsetODEFunction{typeof(f), typeof(γ), typeof(x)}(f, α, β, γ, x)
end

function (o::OffsetODEFunction)(u, p, t)
    o.f(u, p, o.α + o.β * t) .+ o.γ .* o.x
end
function (o::OffsetODEFunction)(du, u, p, t)
    o.f(du, u, p, o.α + o.β * t)
    du .+= o.γ .* o.x
end
function (o::OffsetODEFunction)(du, u, p, t, α)
    o.f(du, u, p, o.α + o.β * t, α)
    du .+= α .* o.γ .* o.x
end
function (o::OffsetODEFunction)(du, u, p, t, α, β)
    o.f(du, u, p, o.α + o.β * t, α, β)
    du .+= α .* o.γ .* o.x
end
