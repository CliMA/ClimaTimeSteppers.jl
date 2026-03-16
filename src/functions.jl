export AbstractClimaODEFunction
export ClimaODEFunction, ForwardEulerODEFunction

abstract type AbstractClimaODEFunction end

"""
    ClimaODEFunction(; T_imp!, [dss!], [cache!], [cache_imp!])
    ClimaODEFunction(; T_exp!, T_lim!, [T_imp!], [lim!], [dss!], [cache!], [cache_imp!])
    ClimaODEFunction(; T_exp_T_lim!, [T_imp!], [lim!], [dss!], [cache!], [cache_imp!])

Container for all functions used to advance through a timestep:
    - `T_imp!(T_imp, u, p, t)`: sets the implicit tendency
    - `T_exp!(T_exp, u, p, t)`: sets the component of the explicit tendency that
      is not passed through the limiter
    - `T_lim!(T_lim, u, p, t)`: sets the component of the explicit tendency that
      is passed through the limiter
    - `T_exp_T_lim!(T_exp, T_lim, u, p, t)`: fused alternative to the separate
      functions `T_exp!` and `T_lim!`
    - `lim!(u, p, t, u_ref)`: applies the limiter to every state `u` that has
      been incremented from `u_ref` by the explicit tendency component `T_lim!`
    - `dss!(u, p, t)`: applies direct stiffness summation to every state `u`,
      except for intermediate states generated within the implicit solver
    - `cache!(u, p, t)`: updates the cache `p` to reflect the state `u` before
      the first timestep and on every subsequent timestepping stage
    - `cache_imp!(u, p, t)`: updates the components of the cache `p` that are
      required to evaluate `T_imp!` and its Jacobian within the implicit solver

By default, `lim!`, `dss!`, and `cache!` all do nothing, and `cache_imp!` is
identical to `cache!`. Any of the tendency functions can be set to `nothing` in
order to avoid corresponding allocations in the integrator.

Internally, `T_exp!` and `T_lim!` are merged into `T_exp_T_lim!` at
construction time. These keyword arguments are still accepted for backward
compatibility.
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
        lim! = (u, p, t, u_ref) -> nothing,
        dss! = (u, p, t) -> nothing,
        initialize_subproblem! = (u, p, γdt) -> nothing,
        cache! = (u, p, t) -> nothing,
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

An ODE function wrapper where `f(un, u, p, t, dt)` provides a forward Euler update
```
un .= u .+ dt * f(u, p, t)
```

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
    OffsetODEFunction(f,α,β,γ,x)

An ODE function wrapper which evaluates `f` with an offset.

Evaluates as
```math
f(u,p,α+β*t) .+ γ .* x
```

It supports the 3, 4, 5, and 6 argument forms.
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
