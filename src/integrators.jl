# SortedQueue: 
#
# Stores values in descending order (for forward-in-time integration) so that:
#   first(q) → smallest value (next tstop)  — O(1) via last(data)
#   pop!(q)  → remove smallest              — O(1) via pop!(data)
#   push!(q) → binary-search insert         — O(n), n ≤ 5 typical
#   empty!(q) — O(1)
#
# For reverse-in-time, values are stored in ascending order and the same
# interface returns the largest (most-negative-time) value first.

struct SortedQueue{T}
    data::Vector{T}    # sorted in reverse order of consumption
    forward::Bool      # true = forward-in-time (ascending tstops)
    function SortedQueue{T}(vals, forward::Bool) where {T}
        new{T}(sort!(collect(T, vals); rev = forward), forward)
    end
end

SortedQueue(vals, forward::Bool) =
    SortedQueue{eltype(vals)}(vals, forward)

Base.isempty(q::SortedQueue) = isempty(q.data)
Base.length(q::SortedQueue) = length(q.data)
Base.first(q::SortedQueue) = q.data[end]
Base.empty!(q::SortedQueue) = (empty!(q.data); q)

function Base.pop!(q::SortedQueue)
    pop!(q.data)
end

function Base.push!(q::SortedQueue, val)
    # Insert val into the reverse-sorted vector at the correct position
    data = q.data
    # Use a positional `Ordering` rather than the `rev` keyword: the keyword
    # form of `searchsortedfirst` allocates, the positional form does not.
    if q.forward
        # data is sorted descending; find insertion point for descending order
        i = searchsortedfirst(data, val, Base.Order.Reverse)
    else
        # data is sorted ascending; find insertion point for ascending order
        i = searchsortedfirst(data, val, Base.Order.Forward)
    end
    # If inserting at the end (e.g. when data is empty or val is smaller than
    # all existing elements), use `push!` instead of `insert!`. In Julia 1.12+,
    # `insert!` at index 1 on an empty Memory-backed Vector triggers `_growbeg!`,
    # which allocates a new Memory buffer, whereas `push!` uses existing tail capacity.
    if i == length(data) + 1
        push!(data, val)
    else
        insert!(data, i, val)
    end
    return q
end

"""
    TimeStepperIntegrator

The main integrator type for ClimaTimeSteppers. Created by [`init`](@ref),
advanced by [`step!`](@ref), and run to completion by [`solve!`](@ref).

# Fields

**User-facing:**
- `u`: current state vector (mutated in place).
- `p`: parameters.
- `t`: current time.
- `dt`: current timestep (adjusted near tstops).
- `sol::ODESolution`: the solution being built.

**Configuration:**
- `alg`: the time-stepping algorithm.
- `_dt`: original `dt` passed to [`init`](@ref).
- `dtchangeable::Bool`: whether `dt` may be shortened to hit tstops.
- `tstops`: min-heap of upcoming stop times.
- `_tstops`: original tstops passed to [`init`](@ref) (used by `reinit!`).
- `saveat`: min-heap of upcoming save times.
- `_saveat`: original saveat passed to [`init`](@ref) (used by `reinit!`).
- `step::Int`: current step counter.
- `stepstop::Int`: maximum number of steps (0 = unlimited).
- `callback`: discrete callbacks invoked after each step.
- `advance_to_tstop::Bool`: when `true`, [`step!`](@ref) advances to the next tstop.
- `cache`: solver-specific pre-allocated workspace.
- `tdir::Int`: `+1` for forward integration, `-1` for reverse.
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
    tdir::Int
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

# `false` suppresses `@deprecate`'s default re-export: `SavedValues` is an
# internal type and must stay unexported (otherwise Documenter's exported-symbol
# docstring check fails on it).
@deprecate SavedValues(::Type{tType}, ::Type{savevalType}) where {tType, savevalType} SavedValues{
    tType,
    savevalType,
}(
    Vector{tType}(),
    Vector{savevalType}(),
) false


# helper function for setting up sorted queues for tstops and saveat
function tstops_and_saveat_queues(t0, tf, tstops, saveat = [])
    FT = typeof(first(promote(t0, tf)))
    forward = tf > t0

    # ensure that tstops includes tf and only has values ahead of t0
    tstops = FT[filter(t -> t0 < t < tf || tf < t < t0, tstops)..., tf]
    tstops = SortedQueue{FT}(tstops, forward)

    # drop saveat points outside [t0, tf]; out-of-range points would otherwise
    # produce spurious saves at the nearest endpoint (cf. `add_saveat!`, which
    # rejects times behind the current time)
    isnothing(saveat) && (saveat = (t0, tf))
    lo, hi = minmax(t0, tf)
    saveat = SortedQueue{FT}(filter(t -> lo <= t <= hi, saveat), forward)

    return tstops, saveat
end

compute_tdir(ts) = ts[1] > ts[end] ? -1 : 1

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
- `save`: attach the saving callback that records the solution (default: `true`);
  set to `false` for an integrator whose output is never read (e.g. the inner
  integrator of a multirate method) to avoid per-step saving allocations
- `dtchangeable`: allow dt reduction near tstops (default: `true`)
- `stepstop`: stop after this many steps (`-1` = unlimited)

# Returns
- a [`TimeStepperIntegrator`](@ref)

# Examples
```julia
import ClimaTimeSteppers as CTS

prob = CTS.ODEProblem(f, u0, (0.0, 1.0), p)
integrator = CTS.init(prob, alg; dt = 0.01)
CTS.step!(integrator)   # advance one step
CTS.solve!(integrator)  # run to completion
```

See also [`solve`](@ref), [`solve!`](@ref), [`step!`](@ref), [`reinit!`](@ref).
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
    save = true,
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
    tstops, saveat = tstops_and_saveat_queues(t0, tf, tstops, saveat)

    sol = ODESolution(typeof(t0)[], typeof(save_func(u0, t0))[], prob, alg)
    # SavedValues shares sol.t and sol.u vectors: the callback's push!
    # directly populates the solution. Do not replace these vectors.
    # `save = false` skips the saving callback entirely; this is used for the
    # inner integrator of multirate methods, whose substeps are never saved,
    # so its `solve!` does not allocate a saved state per stage.
    callback = if save
        saving_callback = NonInterpolatingSavingCallback(
            save_func,
            SavedValues(sol.t, sol.u),
            save_everystep,
        )
        CallbackSet(callback, saving_callback)
    else
        CallbackSet(callback)
    end

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
    # Recompute the integration direction: reinit! may flip it (e.g. a forward
    # solve followed by a reverse-time reinit for an adjoint/checkpoint).
    integrator.tdir = compute_tdir((t0, tf))
    integrator.tstops, integrator.saveat = tstops_and_saveat_queues(t0, tf, tstops, saveat)
    integrator.step = 0
    if erase_sol
        resize!(integrator.sol.t, 0)
        resize!(integrator.sol.u, 0)
    end
    if reinit_callbacks
        initialize_callbacks!(integrator.callback, u0, t0, integrator)
    elseif !isempty(integrator.callback.discrete_callbacks)
        # reinit the saving callback (when present) so that t0 can be saved if
        # needed, without touching user callbacks. An integrator built with
        # `save = false` has no saving callback; identify it by type rather than
        # position, since the last callback may be a user callback in that case.
        saving_callback = integrator.callback.discrete_callbacks[end]
        if saving_callback.affect! isa NonInterpolatingSavingAffect
            saving_callback.initialize(saving_callback, u0, t0, integrator)
        end
    end
end

"""
    solve(prob, alg; kwargs...)

Solve the ODE problem. Equivalent to `init(prob, alg; kwargs...) |> solve!`.

Accepts the same keyword arguments as [`init`](@ref).

# Returns
- an [`ODESolution`](@ref)

# Examples
```julia
import ClimaTimeSteppers as CTS

prob = CTS.ODEProblem(f, u0, (0.0, 1.0), p)
sol = CTS.solve(prob, alg; dt = 0.01)
```

See also [`init`](@ref), [`solve!`](@ref).
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

See also [`init`](@ref), [`solve`](@ref), [`step!`](@ref).
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

See also [`init`](@ref), [`solve!`](@ref).
"""
function step!(integrator::TimeStepperIntegrator)
    if integrator.advance_to_tstop && !isempty(integrator.tstops)
        tstop = first(integrator.tstops)
        while !reached_tstop(integrator, tstop)
            __step!(integrator)
        end
    else
        # No tstop to advance to (e.g. after all tstops have been consumed):
        # fall back to a single internal step.
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

# Helper functions for time-reversed integrators. `tdir` returns +1 or -1,
# `is_past_t` checks whether time `t` has been passed, and `reached_tstop`
# checks whether the integrator has reached or passed a stop time.
tdir(integrator) = integrator.tdir
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
    # equivalent up to machine precision
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
struct NonInterpolatingSavingAffect{T, U, F}
    saved_values::SavedValues{T, U}
    save_func::F
end
function (affect::NonInterpolatingSavingAffect)(integrator)
    push!(affect.saved_values.t, integrator.t)
    push!(affect.saved_values.saveval, affect.save_func(integrator.u, integrator.t))
end

function NonInterpolatingSavingCallback(save_func, saved_values, save_everystep)
    if save_everystep
        condition = Returns(true)
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
    affect! = NonInterpolatingSavingAffect(saved_values, save_func)
    initialize(cb, u, t, integrator) = condition(u, t, integrator) && affect!(integrator)
    finalize(cb, u, t, integrator) =
        !save_everystep && !isempty(integrator.saveat) && affect!(integrator)
    DiscreteCallback(condition, affect!; initialize, finalize)
end
