using DiffEqBase, TimeMachine, LinearAlgebra, StaticArrays


const const_prob = ODEProblem{true}(
    (du,u,p,t,α=true,β=false) -> (du .= α .* p .+ β .* du),
    [0.0],(0.0,1.0),2.0)

function const_sol(u0,p,t)
    u0 + p*t
end




"""
Single variable linear ODE

```math
\\frac{du}{dt} = αu
```
with initial condition ``u_0=\\frac{1}{2}``, parameter ``α=1.01``, and solution
```math
u(t) = u_0 e^{αt}
```

This is an in-place variant of the one from DiffEqProblemLibrary.jl.
"""
const linear_prob = IncrementingODEProblem{true}(
    (du,u,p,t,α=true,β=false) -> (du .= α .* p .* u .+ β .* du),
    [1/2],(0.0,1.0),1.01)

# DiffEqProblemLibrary.jl uses the `analytic` argument to ODEFunction to store the exact solution
# IncrementingODEFunction doesn't have that
function linear_sol(u0,p,t)
    u0 .* exp(p*t)
end


"""
Two variable linear ODE

```math
\\frac{du_1}{dt} = α u_2 \\quad \\frac{du_2}{dt} = -α u_1
```
with initial condition ``u_0=[0,1]``, parameter ``α=2``, and solution
```math
u(t) = [cos(αt) sin(αt); -sin(αt) cos(αt) ] u_0
```
"""
const sincos_prob = IncrementingODEProblem{true}(
    (du,u,p,t,α=true,β=false) -> (du[1] = α*p*u[2]+β*du[1]; du[2] = -α*p*u[1]+β*du[2]),
    [0.0,1.0], (0.0,1.0), 2.0)

function sincos_sol(u0,p,t)
    s,c = sincos(p*t)
    [c s; -s c]*u0
end

"""
IMEX problem with autonomous linear part

```math
\\frac{du}{dt} = u + cos(t)/α
```
with initial condition ``u_0 = 1/2``, parameter ``α=4``, and solution
```math
u(t) = \frac{e^t + sin(t) - cos(t)}{2 α} + u_0 e^t
```
"""
const imex_autonomous_prob = SplitODEProblem(
    (du, u, p, t, α=true, β=false) -> (du .= α .* u        .+ β .* du),
    (du, u, p, t, α=true, β=false) -> (du .= α .* cos(t)/p .+ β .* du),
    [0.5], (0.0,1.0), 4.0)

function imex_autonomous_sol(u0,p,t)
    (exp(t)  + sin(t) - cos(t)) / 2p .+ exp(t) .* u0
end

const imex_nonautonomous_prob = SplitODEProblem(
    (du, u, p, t, α=true, β=false) -> (du .= α .* cos(t) * u .+ β .* du),
    (du, u, p, t, α=true, β=false) -> (du .= α .* cos(t)/p   .+ β .* du),
    [0.5], (0.0,2.0), 4.0)

function imex_nonautonomous_sol(u0,p,t)
    (exp(sin(t)) .* (1 .+ p .* u0) .- 1) ./ p
end


"""
Test problem (4.2) from RobertsSarsharSandu2018arxiv
@article{RobertsSarsharSandu2018arxiv,
    title={Coupled Multirate Infinitesimal GARK Schemes for Stiff Systems with
            Multiple Time Scales},
    author={Roberts, Steven and Sarshar, Arash and Sandu, Adrian},
    journal={arXiv preprint arXiv:1812.00808},
    year={2019}
}

Note: The actual rates are all over the place with this test and passing largely
        depends on final dt size
"""
kpr_param = (
    ω = 100,
    λf = -10,
    λs = -1,
    ξ = 0.1,
    α = 1,
)

function kpr_rhs(Q,param,t)
    yf, ys = Q
    ω, λf, λs, ξ, α = param

    ηfs = ((1 - ξ) / α) * (λf - λs)
    ηsf = -ξ * α * (λf - λs)
    Ω = @SMatrix [
        λf ηfs
        ηsf λs
    ]

    g = @SVector [
        (-3 + yf^2 - cos(ω * t)) / 2yf,
        (-2 + ys^2 - cos(t)) / 2ys,
    ]
    h = @SVector [
        ω * sin(ω * t) / 2yf,
        sin(t) / 2ys,
    ]
    return Ω * g - h
end

function kpr_fast!(dQ, Q, param, t, α=1, β=0)
    dQf,_ = kpr_rhs(Q,param,t)
    dQ[1] = α*dQf + β*dQ[1]
    dQ[2] = β*dQ[2]
end

function kpr_slow!(dQ, Q, param, t, α=1, β=0)
    _,dQs = kpr_rhs(Q,param,t)
    dQ[1] = β*dQ[1]
    dQ[2] = α*dQs + β*dQ[2]
end

function kpr_sol(u0,param,t)
    #@assert u0 == [sqrt(4) sqrt(3)]
    ω, λf, λs, ξ, α = param
    return [
        sqrt(3 + cos(ω * t))
        sqrt(2 + cos(t))
    ]
end

const kpr_multirate_prob = SplitODEProblem(
    IncrementingODEFunction{true}(kpr_fast!),
    IncrementingODEFunction{true}(kpr_slow!),
    [sqrt(4), sqrt(3)], (0.0, 5π/2), kpr_param,
)

const kpr_singlerate_prob = IncrementingODEProblem{true}(
    (dQ,Q,param,t,α=1,β=0) -> (dQ .= α .* kpr_rhs(Q,param,t) .+ β .* dQ),
    [sqrt(4), sqrt(3)], (0.0, 5π/2), kpr_param,
)
