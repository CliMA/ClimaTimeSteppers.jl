# Local problem and function types.
#
# ODEProblem, SplitODEProblem, IncrementingODEFunction, ODEFunction:
# not exported; use qualified access (e.g., CTS.ODEProblem).

"""
    ODEProblem(f, u0, tspan, p)

An ordinary differential equation problem ``du/dt = f(u, p, t)``.

# Arguments
- `f`: the ODE function (`ClimaODEFunction`, `IncrementingODEFunction`, `SplitFunction`, etc.)
- `u0`: initial state
- `tspan`: time span `(t_start, t_end)`
- `p`: parameters passed to `f`
"""
struct ODEProblem{F, U, T, P}
    f::F
    u0::U
    tspan::T
    p::P
end

Base.summary(io::IO, prob::ODEProblem) =
    print(io, "ODEProblem with uType $(typeof(prob.u0)) and tType $(typeof(prob.tspan[1]))")
function Base.show(io::IO, mime::MIME"text/plain", prob::ODEProblem)
    summary(io, prob)
    println(io)
    println(io, "timespan: ", prob.tspan)
    println(io, "u0: ", summary(prob.u0))
    print(io, "p: ", summary(prob.p))
end

"""
    SplitFunction(f1, f2)

A split ODE function representing ``du/dt = f_1(u,p,t) + f_2(u,p,t)``.

Used with [`Multirate`](@ref) algorithms, where `f1` is the **fast
tendency** (advanced by the inner solver) and `f2` is the **slow tendency**
(advanced by the outer solver). Note the distinction: `f1`/`f2` are
*tendency functions*, while [`Multirate`](@ref) takes *algorithm* arguments
`fast`/`slow`.

# Arguments
- `f1`: fast tendency (typically an [`IncrementingODEFunction`](@ref)).
- `f2`: slow tendency.
"""
struct SplitFunction{F1, F2}
    f1::F1
    f2::F2
end

"""
    SplitODEProblem(f1, f2, u0, tspan, p)

Convenience constructor: creates an `ODEProblem(SplitFunction(f1, f2), u0, tspan, p)`.

See [`SplitFunction`](@ref) and [`ODEProblem`](@ref).
"""
SplitODEProblem(f1, f2, u0, tspan, p) = ODEProblem(SplitFunction(f1, f2), u0, tspan, p)

"""
    IncrementingODEFunction{true}(f)

An ODE function that supports the incrementing call form
`f(du, u, p, t, α=true, β=false)`, which computes `du .= α .* f(u,p,t) .+ β .* du`.

Required by [`LowStorageRungeKutta2N`](@ref) methods.

# Arguments
- `f`: callable with signature `f(du, u, p, t, α, β)`

The type parameter `iip` (`true` for in-place, `false` for out-of-place) is
specified as a curly-brace parameter: `IncrementingODEFunction{true}(f)`.
"""
struct IncrementingODEFunction{iip, F}
    f::F
end
function IncrementingODEFunction{iip}(f::F) where {iip, F}
    IncrementingODEFunction{iip, F}(f)
end
(o::IncrementingODEFunction)(du, u, p, t) = o.f(du, u, p, t)
(o::IncrementingODEFunction)(du, u, p, t, α) = o.f(du, u, p, t, α)
(o::IncrementingODEFunction)(du, u, p, t, α, β) = o.f(du, u, p, t, α, β)

"""
    ODEFunction(f; jac_prototype, Wfact, tgrad)

An ODE function wrapper that optionally carries Jacobian information for
implicit solvers. Used as the `T_imp!` field of [`ClimaODEFunction`](@ref).

# Arguments
- `f`: callable with standard ODE signature `f(du, u, p, t)`

# Keyword Arguments
- `jac_prototype`: prototype matrix for the Jacobian (determines sparsity/structure)
- `Wfact`: function `Wfact(W, u, p, dtγ, t)` that computes ``W = dt\\gamma\\, J - I``.
- `tgrad`: function `tgrad(∂f∂t, u, p, t)` for the explicit time derivative.
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

Solution object returned by [`solve`](@ref) and [`solve!`](@ref).

`T` is the time element type, `U` is the saved-state type, `P` is the
problem type, and `A` is the algorithm type.

# Fields
- `t::Vector{T}`: saved time points.
- `u::Vector{U}`: saved state values (one per time point).
- `prob::P`: the original [`ODEProblem`](@ref).
- `alg::A`: the algorithm used.

Calling `sol(t)` returns the saved state nearest to time `t`
(**not** an interpolation — nearest-neighbor lookup only).
"""
struct ODESolution{T, U, P, A}
    t::Vector{T}
    u::Vector{U}
    prob::P
    alg::A
end


"""
    (sol::ODESolution)(t)

Return the saved state nearest to time `t`. This is **not** an interpolation;
it returns `sol.u[i]` where `i = argmin(|sol.t .- t|)`. Only use this for
quick lookups when saved time points are dense enough for your purpose.
"""
function (sol::ODESolution)(t)
    isempty(sol.t) && error("ODESolution has no saved states to look up")
    # Generator avoids allocating `abs.(sol.t .- t)`.
    idx = argmin(abs(t′ - t) for t′ in sol.t)
    return sol.u[idx]
end
