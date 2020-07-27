"""
    OffsetODEFunction(f,γ,x)

An ODE function wrapper which evaluates `f` with an offset.

Evaluates as
```math
f(u,p,t) .+ γ .* x
```

It supports the 3, 4, 5, and 6 argument forms.
"""
mutable struct OffsetODEFunction{iip,F,S,A} <: DiffEqBase.AbstractODEFunction{iip}
    f::F
    γ::S
    x::A
end
OffsetODEFunction{iip}(f,γ,x) where {iip} = OffsetODEFunction{iip, typeof(f), typeof(γ), typeof(x)}(f,γ,x)
OffsetODEFunction(f::DiffEqBase.AbstractODEFunction{iip},γ,x) where {iip} = 
    OffsetODEFunction{iip}(f,γ,x)

function (o::OffsetODEFunction)(u,p,t)
    o.f(u,p,t) .+ o.γ .* o.x
end
function (o::OffsetODEFunction)(du,u,p,t)
    o.f(du,u,p,t) 
    du .+= o.γ .* o.x
end
function (o::OffsetODEFunction)(du,u,p,t,α)
    o.f(du,u,p,t,α) 
    du .+= α .* o.γ .* o.x
end
function (o::OffsetODEFunction)(du,u,p,t,α,β)
    o.f(du,u,p,t,α,β)
    du .+= α .* o.γ .* o.x
end