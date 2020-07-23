using DiffEqBase, TimeMachine, LinearAlgebra, Test

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
IMEX problem with constant linear part

```math
\\frac{du}{dt} = u + cos(t)/α
```
with initial condition ``u_0 = 1/2``, parameter ``α=4``, and solution
```math
u(t) = \frac{e^t + sin(t) - cos(t)}{2 α} + u_0 e^t
```
"""    
const imex_linconst_prob = SplitODEProblem(
    (du, u, p, t) -> (du .= u),
    (du, u, p, t) -> (du .= cos(t)/p),
    [0.5], (0.0,1.0), 4.0)

function imex_linconst_sol(u0,p,t)
    (exp(t)  + sin(t) - cos(t)) / 2p .+ exp(t) .* u0
end

