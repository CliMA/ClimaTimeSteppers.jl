export IncrementODEProblem


"""
    IncrementODEProblem(f, u, (tstart,tstop), param; kwargs...) <: AbstractODEProblem

A variant of `ODEProblem` which uses a 6-argument call to `f`.
"""
struct IncrementODEProblem{uType, tType, P, F, K} <: DiffEqBase.AbstractODEProblem{uType, tType, true}
    """
    The ODE is `du/dt = f(u,p,t)`: this should define a method `f(y, u, p, t, α, β)` which computes
    ```
    du .= α .* f(u, p, t) .+ β .* du
    ```
    for scalar `α` and `β`.
    """
    f::F
    """The initial condition is `u(tspan[1]) = u0`."""
    u0::uType
    """The solution `u(t)` will be computed for `tspan[1] ≤ t ≤ tspan[2]`."""
    tspan::Tuple{tType,tType}
    """Constant parameters to be supplied as the second argument of `f`."""
    p::P
    """A callback to be applied to every solver which uses the problem."""
    kwargs::K
end
IncrementODEProblem(f, u0, tspan, p=DiffEqBase.NullParameters(); kwargs...) = 
    IncrementODEProblem(f, u0, tspan, p, kwargs)
