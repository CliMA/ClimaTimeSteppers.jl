import DataStructures

"""
    TimeStepperIntegrator

The main integrator type for ClimaTimeSteppers. Created by [`init`](@ref),
advanced by [`step!`](@ref), and run to completion by [`solve!`](@ref).

# Key fields (user-facing)
- `u`: current state vector (mutated in place)
- `p`: parameters
- `t`: current time
- `dt`: current timestep (may differ from `_dt` near tstops)
- `sol`: the [`ODESolution`](@ref) being built
"""
mutable struct TimeStepperIntegrator{
    algType,
    uType,
    tType,
    pType,
    heapType,
    tstopsType,
    saveatType,
    callbackType,
    cacheType,
    solType,
}
    alg::algType
    u::uType
    p::pType
    t::tType
    dt::tType
    _dt::tType # argument to init used to set dt in step!
    dtchangeable::Bool
    tstops::heapType
    _tstops::tstopsType # argument to init used as default argument to reinit!
    saveat::heapType
    _saveat::saveatType # argument to init used as default argument to reinit!
    step::Int
    stepstop::Int
    callback::callbackType
    advance_to_tstop::Bool
    cache::cacheType
    sol::solType
    tdir::tType
end

"""
    SavedValues{tType, savevalType}

Internal container holding saved time points and corresponding values.
Used by the saving callback to accumulate solution data into
[`ODESolution`](@ref).
"""
struct SavedValues{tType, savevalType}
    t::Vector{tType}
    saveval::Vector{savevalType}
end

"""
    SavedValues(tType, savevalType)

Construct empty `SavedValues` for the given element types.
"""
function SavedValues(::Type{tType}, ::Type{savevalType}) where {tType, savevalType}
    SavedValues{tType, savevalType}(Vector{tType}(), Vector{savevalType}())
end


# helper function for setting up min/max heaps for tstops and saveat
function tstops_and_saveat_heaps(t0, tf, tstops, saveat = [])
    # We promote to a common type to ensure that t0 and tf have the same type
    FT = typeof(first(promote(t0, tf)))
    ordering = tf > t0 ? DataStructures.FasterForward : DataStructures.FasterReverse

    # ensure that tstops includes tf and only has values ahead of t0
    tstops = [filter(t -> t0 < t < tf || tf < t < t0, tstops)..., tf]
    tstops = DataStructures.BinaryHeap{FT, ordering}(tstops)

    isnothing(saveat) && (saveat = [t0, tf])

    saveat = DataStructures.BinaryHeap{FT, ordering}(collect(saveat))

    return tstops, saveat
end

compute_tdir(ts) = ts[1] > ts[end] ? sign(ts[end] - ts[1]) : oneunit(ts[1])

"""
    init(prob, alg; dt, kwargs...)

Create a [`TimeStepperIntegrator`](@ref) for the given problem and algorithm.

# Arguments
- `prob`: an [`ODEProblem`](@ref)
- `alg`: a [`TimeSteppingAlgorithm`](@ref)

# Keyword Arguments
- `dt`: timestep size (required; must be positive)
- `tstops`: additional times at which the integrator must stop
- `saveat`: times at which to save the solution (default: only endpoints)
- `save_everystep`: save after every step (default: `false`)
- `callback`: a [`DiscreteCallback`](@ref) or [`CallbackSet`](@ref)
- `advance_to_tstop`: if `true`, [`step!`](@ref) advances to the next tstop
- `save_func`: function `(u, t) -> value` applied before saving (default: `copy`)
- `dtchangeable`: allow dt reduction near tstops (default: `true`)
- `stepstop`: stop after this many steps (`-1` = unlimited)

# Returns
- a [`TimeStepperIntegrator`](@ref)
"""
function init(
    prob::ODEProblem,
    alg::TimeSteppingAlgorithm,
    args...;
    dt,
    tstops = (),
    saveat = nothing,
    save_everystep = false,
    callback = nothing,
    advance_to_tstop = false,
    save_func = (u, t) -> copy(u),
    dtchangeable = true,
    stepstop = -1,
    tdir = compute_tdir(prob.tspan),
    kwargs...,
)
    (; u0, p) = prob
    t0, tf = prob.tspan
    t0, tf, dt = promote(t0, tf, dt)

    # We need zero(oneunit()) because there's no zerounit
    dt > zero(oneunit(dt)) || error("dt must be positive")
    _dt = dt
    dt = tf > t0 ? dt : -dt

    _tstops = tstops
    _saveat = saveat
    tstops, saveat = tstops_and_saveat_heaps(t0, tf, tstops, saveat)

    sol = ODESolution(typeof(t0)[], typeof(save_func(u0, t0))[], prob, alg)
    # SavedValues shares sol.t and sol.u vectors: the callback's push!
    # directly populates the solution. Do not replace these vectors.
    saving_callback =
        NonInterpolatingSavingCallback(save_func, SavedValues(sol.t, sol.u), save_everystep)
    callback = CallbackSet(callback, saving_callback)

    integrator = TimeStepperIntegrator(
        alg,
        u0,
        p,
        t0,
        dt,
        _dt,
        dtchangeable,
        tstops,
        _tstops,
        saveat,
        _saveat,
        0,
        stepstop,
        callback,
        advance_to_tstop,
        init_cache(prob, alg; dt, kwargs...),
        sol,
        tdir,
    )
    initialize_function!(prob.f, u0, p, t0)
    initialize_callbacks!(callback, u0, t0, integrator)
    return integrator
end

"""
    reinit!(integrator, [u0]; t0, tf, erase_sol, tstops, saveat, reinit_callbacks)

Reset the integrator to a new initial condition without reallocating caches.

# Arguments
- `u0`: new initial state (default: original `prob.u0`)

# Keyword Arguments
- `t0`, `tf`: new time span endpoints (default: original `prob.tspan`)
- `erase_sol`: clear saved solution data (default `true`)
- `tstops`, `saveat`: new stop/save schedules (default: reuse originals)
- `reinit_callbacks`: re-run callback initialization (default `true`)
"""
function reinit!(
    integrator::TimeStepperIntegrator,
    u0 = integrator.sol.prob.u0;
    t0 = integrator.sol.prob.tspan[1],
    tf = integrator.sol.prob.tspan[2],
    erase_sol = true,
    tstops = integrator._tstops,
    saveat = integrator._saveat,
    reinit_callbacks = true,
)
    integrator.u .= u0
    integrator.t = t0
    integrator.tstops, integrator.saveat = tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    integrator.step = 0
    if erase_sol
        resize!(integrator.sol.t, 0)
        resize!(integrator.sol.u, 0)
    end
    if reinit_callbacks
        initialize_callbacks!(integrator.callback, u0, t0, integrator)
    else # always reinit the saving callback so that t0 can be saved if needed
        saving_callback = integrator.callback.discrete_callbacks[end]
        saving_callback.initialize(saving_callback, u0, t0, integrator)
    end
end

"""
    solve(prob, alg; kwargs...)

Solve the ODE problem. Equivalent to `init(prob, alg; kwargs...) |> solve!`.

Accepts the same keyword arguments as [`init`](@ref).

# Returns
- an [`ODESolution`](@ref)
"""
function solve(
    prob::ODEProblem,
    alg::TimeSteppingAlgorithm,
    args...;
    kwargs...,
)
    integrator = init(prob, alg, args...; kwargs...)
    solve!(integrator)
end

"""
    solve!(integrator)

Run the integrator to completion (until `tstops` is empty or `stepstop` is reached).

# Returns
- the [`ODESolution`](@ref) stored in `integrator.sol`
"""
NVTX.@annotate function solve!(integrator::TimeStepperIntegrator)
    while !isempty(integrator.tstops) && integrator.step != integrator.stepstop
        __step!(integrator)
    end
    finalize_callbacks!(integrator.callback, integrator.u, integrator.t, integrator)
    return integrator.sol
end

"""
    step!(integrator)
    step!(integrator, dt, stop_at_tdt=false)

Advance the integrator. The single-argument form takes one internal step
(or advances to the next tstop if `advance_to_tstop` was set).
The two-argument form advances by exactly `dt` time units.

# Arguments
- `dt`: time interval to advance (must be positive)
- `stop_at_tdt`: if `true`, add `t + dt` as a tstop to guarantee exact stopping
"""
function step!(integrator::TimeStepperIntegrator)
    if integrator.advance_to_tstop
        tstop = first(integrator.tstops)
        while !reached_tstop(integrator, tstop)
            __step!(integrator)
        end
    else
        __step!(integrator)
    end
end

function step!(integrator::TimeStepperIntegrator, dt, stop_at_tdt = false)
    dt <= zero(dt) && error("dt must be positive")
    stop_at_tdt &&
        !integrator.dtchangeable &&
        error("Cannot stop at t + dt if dtchangeable is false")
    t_plus_dt = integrator.t + tdir(integrator) * dt
    stop_at_tdt && add_tstop!(integrator, t_plus_dt)
    while !reached_tstop(integrator, t_plus_dt, stop_at_tdt)
        __step!(integrator)
    end
end

# helper functions for dealing with time-reversed integrators
tdir(integrator) = integrator.tstops.ordering isa DataStructures.FasterForward ? 1 : -1
is_past_t(integrator, t) = tdir(integrator) * (t - integrator.t) < zero(integrator.t)
reached_tstop(integrator, tstop, stop_at_tstop = integrator.dtchangeable) =
    integrator.t == tstop || (!stop_at_tstop && is_past_t(integrator, tstop))


@inline unrolled_foreach(::Tuple{}, integrator) = nothing
@inline unrolled_foreach(callback, integrator) =
    callback.condition(integrator.u, integrator.t, integrator) ?
    callback.affect!(integrator) : nothing
@inline unrolled_foreach(discrete_callbacks::Tuple{Any}, integrator) =
    unrolled_foreach(first(discrete_callbacks), integrator)
@inline function unrolled_foreach(discrete_callbacks::Tuple, integrator)
    unrolled_foreach(first(discrete_callbacks), integrator)
    unrolled_foreach(Base.tail(discrete_callbacks), integrator)
end

function __step!(integrator)
    (; _dt, dtchangeable, tstops) = integrator

    # update step and dt before incrementing u; if dt is changeable and there is
    # a tstop within dt, reduce dt to tstop - t
    integrator.step += 1
    integrator.dt =
        !isempty(tstops) && dtchangeable ?
        tdir(integrator) * min(_dt, abs(first(tstops) - integrator.t)) :
        tdir(integrator) * _dt
    step_u!(integrator)

    # increment t by dt, rounding to the first tstop if that is roughly
    # equivalent up to machine precision; the specific bound of 100 * eps...
    # is taken from OrdinaryDiffEq.jl
    t_plus_dt = integrator.t + integrator.dt
    t_unit = oneunit(integrator.t)
    max_t_error = 100 * eps(float(integrator.t / t_unit)) * float(t_unit)
    integrator.t =
        !isempty(tstops) && abs(float(first(tstops)) - float(t_plus_dt)) < max_t_error ?
        first(tstops) : t_plus_dt

    # apply callbacks
    discrete_callbacks = integrator.callback.discrete_callbacks
    unrolled_foreach(discrete_callbacks, integrator)

    # remove tstops that were just reached
    while !isempty(tstops) && reached_tstop(integrator, first(tstops))
        pop!(tstops)
    end
end

# solvers need to define this interface
NVTX.@annotate function step_u!(integrator)
    step_u!(integrator, integrator.cache)
end

"""
    get_dt(integrator)

Return the base timestep `dt` (the value passed to [`init`](@ref)).
"""
get_dt(integrator::TimeStepperIntegrator) = integrator._dt

"""
    set_dt!(integrator, dt)

Set the base timestep. `dt` must be positive.
"""
function set_dt!(integrator::TimeStepperIntegrator, dt)
    dt <= zero(dt) && error("dt must be positive")
    integrator._dt = dt
end

"""
    add_tstop!(integrator, t)

Schedule a mandatory stop at time `t`. The integrator will reduce its
timestep if necessary to land exactly on `t`.
"""
function add_tstop!(integrator::TimeStepperIntegrator, t)
    is_past_t(integrator, t) &&
        error("Cannot add a tstop at $t because that is behind the current \
               integrator time $(integrator.t)")
    push!(integrator.tstops, t)
end

"""
    add_saveat!(integrator, t)

Schedule a save point at time `t`. The solution will be recorded at
the first step on or after `t`.
"""
function add_saveat!(integrator::TimeStepperIntegrator, t)
    is_past_t(integrator, t) &&
        error("Cannot add a saveat point at $t because that is behind the \
               current integrator time $(integrator.t)")
    push!(integrator.saveat, t)
end




# ============================================================================ #
# Saving callback
# ============================================================================ #
function NonInterpolatingSavingCallback(save_func, saved_values, save_everystep)
    if save_everystep
        condition = (u, t, integrator) -> true
    else
        condition =
            (u, t, integrator) -> begin
                cond = false
                while !isempty(integrator.saveat) && (
                    first(integrator.saveat) == integrator.t ||
                    is_past_t(integrator, first(integrator.saveat))
                )
                    cond = true
                    pop!(integrator.saveat)
                end
                return cond
            end
    end
    function affect!(integrator)
        push!(saved_values.t, integrator.t)
        push!(saved_values.saveval, save_func(integrator.u, integrator.t))
    end
    initialize(cb, u, t, integrator) = condition(u, t, integrator) && affect!(integrator)
    finalize(cb, u, t, integrator) =
        !save_everystep && !isempty(integrator.saveat) && affect!(integrator)
    DiscreteCallback(condition, affect!; initialize, finalize)
end
