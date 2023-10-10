import DiffEqBase
export AbstractClimaODEFunction
export ClimaODEFunction, ForwardEulerODEFunction

abstract type AbstractClimaODEFunction <: DiffEqBase.AbstractODEFunction{true} end

struct ClimaODEFunction{TL, TE, TI, L, D, PE, PI} <: AbstractClimaODEFunction
    T_lim!::TL
    T_exp!::TE
    T_imp!::TI
    lim!::L
    dss!::D
    post_explicit!::PE
    post_implicit!::PI
end
function ClimaODEFunction(;
    T_lim! = nothing,
    T_exp! = nothing,
    T_imp! = nothing,
    lim! = nothing,
    dss! = nothing,
    post_explicit! = nothing,
    post_implicit! = nothing,
)
    isnothing(T_lim!) && (T_lim! = (uₜ, u, p, t) -> nothing)
    isnothing(T_exp!) && (T_exp! = (uₜ, u, p, t) -> nothing)
    T_imp! = nothing
    isnothing(lim!) && (lim! = (u, p, t, u_ref) -> nothing)
    isnothing(dss!) && (dss! = (u, p, t) -> nothing)
    isnothing(post_explicit!) && (post_explicit! = (u, p, t) -> nothing)
    isnothing(post_implicit!) && (post_implicit! = (u, p, t) -> nothing)
    return ClimaODEFunction(T_lim!, T_exp!, T_imp!, lim!, dss!, post_explicit!, post_implicit!)
end

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
