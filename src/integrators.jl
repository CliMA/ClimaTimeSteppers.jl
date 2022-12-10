import DataStructures

"""
    DistributedODEIntegrator <: AbstractODEIntegrator

A simplified variant of [`ODEIntegrator`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/6ec5a55bda26efae596bf99bea1a1d729636f412/src/integrators/type.jl#L77-L123).

"""
mutable struct DistributedODEIntegrator{
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
} <: DiffEqBase.AbstractODEIntegrator{algType, true, uType, tType}
    alg::algType
    u::uType
    p::pType
    t::tType
    dt::tType
    _dt::tType # argument to __init used to set dt in step!
    dtchangeable::Bool
    tstops::heapType
    _tstops::tstopsType # argument to __init used as default argument to reinit!
    saveat::heapType
    _saveat::saveatType # argument to __init used as default argument to reinit!
    step::Int
    stepstop::Int
    callback::callbackType
    advance_to_tstop::Bool
    u_modified::Bool # not used; field is required for compatibility with
                     # DiffEqBase.initialize! and DiffEqBase.finalize!
    cache::cacheType
    sol::solType
end

# helper function for setting up min/max heaps for tstops and saveat
function tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    FT = typeof(tf)
    ordering = tf > t0 ? DataStructures.FasterForward :
        DataStructures.FasterReverse

    # ensure that tstops includes tf and only has values ahead of t0
    tstops = [filter(t -> t0 < t < tf || tf < t < t0, tstops)..., tf]
    tstops = DataStructures.BinaryHeap{FT, ordering}(tstops)

    if isnothing(saveat)
        saveat = [t0, tf]        
    elseif saveat isa Number
        saveat > zero(saveat) || error("saveat value must be positive")
        saveat = tf > t0 ? saveat : -saveat
        saveat = [t0:saveat:tf..., tf]
    else
        # We do not need to filter saveat like tstops because the saving
        # callback will ignore any times that are not between t0 and tf.
        saveat = collect(saveat)
    end
    saveat = DataStructures.BinaryHeap{FT, ordering}(saveat)

    return tstops, saveat
end

# called by DiffEqBase.init and DiffEqBase.solve
function DiffEqBase.__init(
    prob::DiffEqBase.AbstractODEProblem,
    alg::DistributedODEAlgorithm,
    args...;
    dt,
    tstops = (),
    saveat = nothing,
    save_everystep = false,
    save_func = (u, t, integrator) -> copy(u),
    callback = nothing,
    advance_to_tstop = false,
    progress = false,
    dtchangeable = true,   # custom kwarg
    stepstop = -1,         # custom kwarg
    progress_kwargs = (;), # custom kwarg
    kwargs...,
)
    (; u0, p) = prob
    t0, tf = prob.tspan

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    dt = tf > t0 ? dt : -dt

    _tstops = tstops
    _saveat = saveat
    tstops, saveat = tstops_and_saveat_heaps(t0, tf, tstops, saveat)

    sol = DiffEqBase.build_solution(prob, alg, typeof(t0)[], typeof(u0)[])
    saving_callback = NonInterpolatingSavingCallback(
        save_func,
        DiffEqCallbacks.SavedValues(sol.t, sol.u),
        save_everystep,
    )
    progress_callback = if progress == false
        nothing
    elseif progress == true
        if stdout isa Base.TTY
            TerminalProgressCallback(typeof(t0); progress_kwargs...)
        else
            BasicProgressCallback(; progress_kwargs...)
        end
    elseif progress == :terminal
        TerminalProgressCallback(typeof(t0); progress_kwargs...)
    elseif progress == :basic
        BasicProgressCallback(; progress_kwargs...)
    else
        error("progress must be true, false, :terminal, or :basic")
    end
    callback =
        DiffEqBase.CallbackSet(callback, saving_callback, progress_callback)
    isempty(callback.continuous_callbacks) ||
        error("Continuous callbacks are not supported")
    
    integrator = DistributedODEIntegrator(
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
        false,
        cache(prob, alg; dt, kwargs...),
        sol,
    )
    DiffEqBase.initialize!(callback, u0, t0, integrator)
    return integrator
end

DiffEqBase.has_reinit(integrator::DistributedODEIntegrator) = true
function DiffEqBase.reinit!(
    integrator::DistributedODEIntegrator,
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
    integrator.tstops, integrator.saveat =
        tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    integrator.step = 0
    if erase_sol
        resize!(integrator.sol.t, 0)
        resize!(integrator.sol.u, 0)
    end
    if reinit_callbacks
        DiffEqBase.initialize!(integrator.callback, u0, t0, integrator)
    else # always reinit the saving callback so that t0 can be saved if needed
        saving_callback = integrator.callback.discrete_callbacks[end]
        DiffEqBase.initialize!(saving_callback, u0, t0, integrator)
    end
end

# called by DiffEqBase.solve
function DiffEqBase.__solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::DistributedODEAlgorithm,
    args...;
    kwargs...,
)
    integrator = DiffEqBase.__init(prob, alg, args...; kwargs...)
    DiffEqBase.solve!(integrator)
end

# either called directly (after init), or by DiffEqBase.solve (via __solve)
function DiffEqBase.solve!(integrator::DistributedODEIntegrator)
    while !isempty(integrator.tstops) && integrator.step != integrator.stepstop
        __step!(integrator)
    end
    DiffEqBase.finalize!(
        integrator.callback,
        integrator.u,
        integrator.t,
        integrator,
    )
    return integrator.sol
end

function DiffEqBase.step!(integrator::DistributedODEIntegrator)
    if integrator.advance_to_tstop
        tstop = first(integrator.tstops)
        while !reached_tstop(integrator, tstop)
            __step!(integrator)
        end
    else
        __step!(integrator)
    end
end

function DiffEqBase.step!(
    integrator::DistributedODEIntegrator,
    dt,
    stop_at_tdt = false,
)
    # OridinaryDiffEq lets dt be negative if tdir is -1, but that's inconsistent
    dt <= zero(dt) && error("dt must be positive")
    stop_at_tdt && !integrator.dtchangeable &&
        error("Cannot stop at t + dt if dtchangeable is false")
    t_plus_dt = integrator.t + tdir(integrator) * dt
    stop_at_tdt && DiffEqBase.add_tstop!(integrator, t_plus_dt)
    while !reached_tstop(integrator, t_plus_dt, stop_at_tdt)
        __step!(integrator)
    end
end

# helper functions for dealing with time-reversed integrators in the same way
# that OrdinaryDiffEq.jl does
tdir(integrator) =
    integrator.tstops.ordering isa DataStructures.FasterForward ? 1 : -1
is_past_t(integrator, t) =
    tdir(integrator) * (t - integrator.t) < zero(integrator.t)
reached_tstop(integrator, tstop, stop_at_tstop = integrator.dtchangeable) =
    integrator.t == tstop || (!stop_at_tstop && is_past_t(integrator, tstop))

function __step!(integrator)
    (; _dt, dtchangeable, tstops) = integrator

    # update step and dt before incrementing u; if dt is changeable and there is
    # a tstop within dt, reduce dt to tstop - t
    integrator.step += 1
    integrator.dt = !isempty(tstops) && dtchangeable ?
        tdir(integrator) * min(_dt, abs(first(tstops) - integrator.t)) :
        tdir(integrator) * _dt
    step_u!(integrator)

    # increment t by dt, rounding to the first tstop if that is equivalent up to
    # round-off error; they are considered equivalent if they differ by less
    # than 100eps(t), which is roughly identical to what OrdinaryDiffEq.jl does:
    # https://github.com/SciML/OrdinaryDiffEq.jl/blob/129c76bcc35fd9801f36ce090035c1b750a842ec/src/integrators/integrator_utils.jl#L229-L236
    t_plus_dt = integrator.t + integrator.dt
    integrator.t =
        !isempty(tstops) &&
        abs(first(tstops) - t_plus_dt) < 100 * eps(integrator.t) ?
        first(tstops) : t_plus_dt

    # apply callbacks
    discrete_callbacks = integrator.callback.discrete_callbacks
    for callback in discrete_callbacks
        if callback.condition(integrator.u, integrator.t, integrator)
            callback.affect!(integrator)
        end
    end

    # remove tstops that were just reached
    while !isempty(tstops) && reached_tstop(integrator, first(tstops))
        pop!(tstops)
    end
end

# solvers need to define this interface
step_u!(integrator) = step_u!(integrator, integrator.cache)

DiffEqBase.get_dt(integrator::DistributedODEIntegrator) = integrator._dt
function set_dt!(integrator::DistributedODEIntegrator, dt)
    # TODO: figure out interface for recomputing other objects (linear operators, etc)
    dt <= zero(dt) && error("dt must be positive")
    integrator._dt = dt
end

function DiffEqBase.add_tstop!(integrator::DistributedODEIntegrator, t)
    is_past_t(integrator, t) &&
        error("Cannot add a tstop at $t because that is behind the current \
               integrator time $(integrator.t)")
    push!(integrator.tstops, t)
end

function DiffEqBase.add_saveat!(integrator::DistributedODEIntegrator, t)
    is_past_t(integrator, t) &&
        error("Cannot add a saveat point at $t because that is behind the \
               current integrator time $(integrator.t)")
    push!(integrator.saveat, t)
end

# not sure what this should do?
# defined as default initialize: https://github.com/SciML/DiffEqBase.jl/blob/master/src/callbacks.jl#L3
DiffEqBase.u_modified!(i::DistributedODEIntegrator,bool) = nothing

# this is roughly based on SavingCallback from DiffEqCallbacks, except that it
# doesn't interpolate; instead it will save the first step after
function NonInterpolatingSavingCallback(save_func, saved_values, save_everystep)
    if save_everystep
        condition = (u, t, integrator) -> true
    else
        condition = (u, t, integrator) -> begin
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
        push!(
            saved_values.saveval,
            save_func(integrator.u, integrator.t, integrator),
        )
    end
    initialize(cb, u, t, integrator) =
        condition(u,t,integrator) && affect!(integrator)
    finalize(cb, u, t, integrator) =
        !save_everystep && !isempty(integrator.saveat) && affect!(integrator)
    DiffEqBase.DiscreteCallback(condition, affect!; initialize, finalize)
end
