import DiffEqBase
export AbstractClimaODEFunction
export ClimaODEFunction, ForwardEulerODEFunction

abstract type AbstractClimaODEFunction <: DiffEqBase.AbstractODEFunction{true} end

struct ClimaODEFunction{TEL, TL, TE, TI, L, D, PE, PI} <: AbstractClimaODEFunction
    T_exp_T_lim!::TEL
    T_lim!::TL
    T_exp!::TE
    T_imp!::TI
    lim!::L
    dss!::D
    post_explicit!::PE
    post_implicit!::PI
    function ClimaODEFunction(;
        T_exp_T_lim! = nothing, # nothing or (uₜ_exp, uₜ_lim, u, p, t) -> ...
        T_lim! = nothing, # nothing or (uₜ, u, p, t) -> ...
        T_exp! = nothing, # nothing or (uₜ, u, p, t) -> ...
        T_imp! = nothing, # nothing or (uₜ, u, p, t) -> ...
        lim! = (u, p, t, u_ref) -> nothing,
        dss! = (u, p, t) -> nothing,
        post_explicit! = (u, p, t) -> nothing,
        post_implicit! = (u, p, t) -> nothing,
    )
        args = (T_exp_T_lim!, T_lim!, T_exp!, T_imp!, lim!, dss!, post_explicit!, post_implicit!)

        if !isnothing(T_exp_T_lim!)
            @assert isnothing(T_exp!) "`T_exp_T_lim!` was passed, `T_exp!` must be `nothing`"
            @assert isnothing(T_lim!) "`T_exp_T_lim!` was passed, `T_lim!` must be `nothing`"
        end
        if !isnothing(T_exp!) && !isnothing(T_lim!)
            @warn "Both `T_exp!` and `T_lim!` are not `nothing`, please use `T_exp_T_lim!` instead."
        end
        return new{typeof.(args)...}(args...)
    end
end

has_T_exp(f::ClimaODEFunction) = !isnothing(f.T_exp!) || !isnothing(f.T_exp_T_lim!)
has_T_lim(f::ClimaODEFunction) = !isnothing(f.lim!) && (!isnothing(f.T_lim!) || !isnothing(f.T_exp_T_lim!))

# Don't wrap a AbstractClimaODEFunction in an ODEFunction (makes ODEProblem work).
DiffEqBase.ODEFunction{iip}(f::AbstractClimaODEFunction) where {iip} = f
DiffEqBase.ODEFunction(f::AbstractClimaODEFunction) = f

"""
    ForwardEulerODEFunction(f; jac_prototype, Wfact, tgrad)

An ODE function wrapper where `f(un, u, p, t, dt)` provides a forward Euler update
```
un .= u .+ dt * f(u, p, t)
```

"""
struct ForwardEulerODEFunction{F, J, W, T} <: DiffEqBase.AbstractODEFunction{true}
    f::F
    jac_prototype::J
    Wfact::W
    tgrad::T
end
ForwardEulerODEFunction(f; jac_prototype = nothing, Wfact = nothing, tgrad = nothing) =
    ForwardEulerODEFunction(f, jac_prototype, Wfact, tgrad)
(f::ForwardEulerODEFunction{F})(un, u, p, t, dt) where {F} = f.f(un, u, p, t, dt)

# Don't wrap a ForwardEulerODEFunction in an ODEFunction.
DiffEqBase.ODEFunction{iip}(f::ForwardEulerODEFunction) where {iip} = f
DiffEqBase.ODEFunction(f::ForwardEulerODEFunction) = f

"""
    OffsetODEFunction(f,α,β,γ,x)

An ODE function wrapper which evaluates `f` with an offset.

Evaluates as
```math
f(u,p,α+β*t) .+ γ .* x
```

It supports the 3, 4, 5, and 6 argument forms.
"""
mutable struct OffsetODEFunction{iip, F, S, A} <: DiffEqBase.AbstractODEFunction{iip}
    f::F
    α::S
    β::S
    γ::S
    x::A
end
function OffsetODEFunction{iip}(f, α, β, γ, x) where {iip}
    α, β, γ = promote(α, β, γ)
    OffsetODEFunction{iip, typeof(f), typeof(γ), typeof(x)}(f, α, β, γ, x)
end
OffsetODEFunction(f::DiffEqBase.AbstractODEFunction{iip}, α, β, γ, x) where {iip} =
    OffsetODEFunction{iip}(f, α, β, γ, x)

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
