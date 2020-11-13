struct Polynomial{m,S}
    coefs::NTuple{m,S}
    function Polynomial(coefs::S...) where {S}
        new{length(coefs),S}(coefs)
    end
end
(p::Polynomial)(x) = polyeval(x, p.coefs)


"""
    OffsetODEFunction(f,α,β,γ,x)

An ODE function wrapper which evaluates `f` with an offset.

Evaluates as
```math
f(u,p,α+β*τ) .+ ∑_j γ_j(τ) .* x_j
```

It supports the 3, 4, 5, and 6 argument forms.
"""
mutable struct OffsetODEFunction{iip,F,S,N,P,A} <: DiffEqBase.AbstractODEFunction{iip}
    f::F
    α::S
    β::S
    γ::NTuple{N,P}
    x::NTuple{N,A}
end
function OffsetODEFunction{iip}(f,α,β,γ::Tuple,x::Tuple) where {iip}
    α,β = promote(α,β)
    @assert length(γ) == length(x)
    N = length(γ)
    OffsetODEFunction{iip, typeof(f), typeof(α), N, eltype(γ), eltype(x)}(f,α,β,γ,x)
end
OffsetODEFunction(f::DiffEqBase.AbstractODEFunction{iip},α,β,γ,x) where {iip} =
    OffsetODEFunction{iip}(f,α,β,γ,x)

function (o::OffsetODEFunction)(u,p,t)
    N = length(o.γ)
    y = o.f(u,p,o.α+o.β*t)
    for j = 1:N
        y .+= o.γ[j](t) .* o.x[j]
    end
end
function (o::OffsetODEFunction)(du,u,p,t)
    N = length(o.γ)
    o.f(du,u,p,o.α+o.β*t)
    for j = 1:N
        du .+= o.γ[j](t) .* o.x[j]
    end
end
function (o::OffsetODEFunction)(du,u,p,t,α)
    N = length(o.γ)
    o.f(du,u,p,o.α+o.β*t,α)
    for j = 1:N
        du .+= (α * o.γ[j](t)) .* o.x[j]
    end
end
function (o::OffsetODEFunction)(du,u,p,t,α,β)
    N = length(o.γ)
    o.f(du,u,p,o.α+o.β*t,α,β)
    for j = 1:N
        du .+= (α * o.γ[j](t)) .* o.x[j]
    end
end
