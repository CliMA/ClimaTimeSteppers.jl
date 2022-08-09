export ForwardEulerODEFunction

"""
    ForwardEulerODEFunction(f)

An ODE function wrapper where `f(un, u, p, t, dt)` provides a forward Euler update
```
un .= u .+ dt * f(u, p, t)
```

"""
struct ForwardEulerODEFunction{F} <: DiffEqBase.AbstractODEFunction{true}
    f::F
end
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
mutable struct OffsetODEFunction{iip,F,S,A} <: DiffEqBase.AbstractODEFunction{iip}
    f::F
    α::S
    β::S
    γ::S
    x::A
end
function OffsetODEFunction{iip}(f,α,β,γ,x) where {iip}
    α,β,γ = promote(α,β,γ)
    OffsetODEFunction{iip, typeof(f), typeof(γ), typeof(x)}(f,α,β,γ,x)
end
OffsetODEFunction(f::DiffEqBase.AbstractODEFunction{iip},α,β,γ,x) where {iip} =
    OffsetODEFunction{iip}(f,α,β,γ,x)

function (o::OffsetODEFunction)(u,p,t)
    o.f(u,p,o.α+o.β*t) .+ o.γ .* o.x
end
function (o::OffsetODEFunction)(du,u,p,t)
    o.f(du,u,p,o.α+o.β*t)
    du .+= o.γ .* o.x
end
function (o::OffsetODEFunction)(du,u,p,t,α)
    o.f(du,u,p,o.α+o.β*t,α)
    du .+= α .* o.γ .* o.x
end
function (o::OffsetODEFunction)(du,u,p,t,α,β)
    o.f(du,u,p,o.α+o.β*t,α,β)
    du .+= α .* o.γ .* o.x
end
