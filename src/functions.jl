import DiffEqBase
export ClimaODEFunction, ForwardEulerODEFunction

abstract type AbstractClimaTimeSteppersLogger end
struct NullLogger <: AbstractClimaTimeSteppersLogger end
struct DebugLogger <: AbstractClimaTimeSteppersLogger end

struct AlgMeta{M}
    meta::M
end
Base.show(io::IO, am::AlgMeta) = Base.show(io, am.meta)

abstract type AbstractMetaFunc end

# Drop meta object by default when called
# our test suit defines a new AbstractMetaFunc
# where `meta_tuple(...) = (meta, )`.
@inline function (f::AbstractMetaFunc)(meta::AlgMeta, args...)
    f.f(meta_tuple(f, meta)..., args...)
end

struct MetaFunc{F} <: AbstractMetaFunc; f::F; end
MetaFunc(x::Nothing) = x
meta_tuple(::MetaFunc, meta) = ()

struct ClimaODEFunction{TL, TE, TI, L, D, PESCB, PISCB, LO} <: DiffEqBase.AbstractODEFunction{true}
    T_lim!::TL
    T_exp!::TE
    T_imp!::TI
    lim!::L
    apply_filter!::D
    post_explicit_stage_callback!::PESCB
    post_implicit_stage_callback!::PISCB
    logger::LO
end
function ClimaODEFunction(;
    T_lim! = nothing, # nothing or (uₜ, u, p, t) -> ...
    T_exp! = nothing, # nothing or (uₜ, u, p, t) -> ...
    T_imp! = nothing, # nothing or (uₜ, u, p, t) -> ...
    lim! = (u, p, t, u_ref) -> nothing,
    apply_filter! = (u, p, t) -> nothing,
    post_explicit_stage_callback! = (u, p, t) -> nothing,
    post_implicit_stage_callback! = (u, p, t) -> nothing,
    logger = CTS.NullLogger()
)
    ClimaODEFunction(
        MetaFunc(T_lim!),
        MetaFunc(T_exp!),
        MetaFunc(T_imp!),
        MetaFunc(lim!),
        MetaFunc(apply_filter!),
        MetaFunc(post_explicit_stage_callback!),
        MetaFunc(post_implicit_stage_callback!),
        logger
    )
end

import SciMLBase
# Wrap incoming functions in MetaFunc structs.
SciMLBase.ODEFunction(
    T_imp!::Function;
    Wfact::Function,
    kwargs...,
) = SciMLBase.ODEFunction(MetaFunc(T_imp!); Wfact=MetaFunc(Wfact), kwargs...)

# Don't wrap a ClimaODEFunction in an ODEFunction (makes ODEProblem work).
DiffEqBase.ODEFunction{iip}(f::ClimaODEFunction) where {iip} = f
DiffEqBase.ODEFunction(f::ClimaODEFunction) = f

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
