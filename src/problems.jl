"""
Local problem and function types.
"""

# ODEProblem, SplitODEProblem, IncrementingODEFunction, ODEFunction:
# not exported yet; use qualified (e.g., CTS.ODEProblem) to avoid SciMLBase conflicts.

"""
    ODEProblem(f, u0, tspan, p)

An ordinary differential equation problem.

# Arguments
- `f`: the ODE function (ClimaODEFunction, IncrementingODEFunction, SplitFunction, etc.)
- `u0`: initial state
- `tspan`: time span as a tuple `(t_start, t_end)`
- `p`: parameters
"""
struct ODEProblem{F, U, T, P}
    f::F
    u0::U
    tspan::T
    p::P
end

"""
    SplitFunction(f1, f2)

A split ODE function with fast and slow components: `du/dt = f1(u,p,t) + f2(u,p,t)`.
"""
struct SplitFunction{F1, F2}
    f1::F1
    f2::F2
end

"""
    SplitODEProblem(f1, f2, u0, tspan, p)

Convenience constructor for an ODEProblem with a SplitFunction.
"""
SplitODEProblem(f1, f2, u0, tspan, p) = ODEProblem(SplitFunction(f1, f2), u0, tspan, p)

"""
    IncrementingODEFunction{iip}(f)

An ODE function that supports the incrementing form
`f(du, u, p, t, α=true, β=false)` which computes `du .= α .* f(u,p,t) .+ β .* du`.

The `iip` parameter indicates whether the function is in-place (true) or not.
"""
struct IncrementingODEFunction{iip, F}
    f::F
end
function IncrementingODEFunction{iip}(f::F) where {iip, F}
    IncrementingODEFunction{iip, F}(f)
end
# Call with 4 args: f(du, u, p, t) — uses default α=true, β=false
(o::IncrementingODEFunction)(du, u, p, t) = o.f(du, u, p, t)
# Call with 5 args: f(du, u, p, t, α)
(o::IncrementingODEFunction)(du, u, p, t, α) = o.f(du, u, p, t, α)
# Call with 6 args: f(du, u, p, t, α, β)
(o::IncrementingODEFunction)(du, u, p, t, α, β) = o.f(du, u, p, t, α, β)

"""
    ODEFunction(f; jac_prototype, Wfact, tgrad)

An ODE function wrapper, optionally carrying Jacobian information.

# Keyword Arguments
- `jac_prototype`: a prototype matrix for the Jacobian (e.g., for preallocating)
- `Wfact`: a function `Wfact(W, u, p, dtγ, t)` that computes `W = I - dtγ * J`
- `tgrad`: a function `tgrad(∂f∂t, u, p, t)`
"""
struct ODEFunction{F, JP, WF, TG}
    f::F
    jac_prototype::JP
    Wfact::WF
    tgrad::TG
end
function ODEFunction(f; jac_prototype = nothing, Wfact = nothing, tgrad = nothing)
    ODEFunction(f, jac_prototype, Wfact, tgrad)
end
(o::ODEFunction)(args...) = o.f(args...)

"""
    ODESolution{T, U, P, A}

A minimal ODE solution type containing saved time points and state values.
"""
struct ODESolution{T, U, P, A}
    t::Vector{T}
    u::Vector{U}
    prob::P
    alg::A
end
ODESolution(prob, alg, t::Vector{T}, u::Vector{U}) where {T, U} =
    ODESolution{T, U, typeof(prob), typeof(alg)}(t, u, prob, alg)

"""
    sol(t)

Return the saved state nearest to time `t`. This is **not** an interpolation;
it returns `sol.u[i]` where `i = argmin(|sol.t .- t|)`. Only use this for
quick lookups when saved time points are dense enough for your purpose.
"""
function (sol::ODESolution)(t)
    idx = argmin(abs.(sol.t .- t))
    return sol.u[idx]
end
