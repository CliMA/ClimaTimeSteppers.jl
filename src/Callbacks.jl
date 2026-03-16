"""
    ClimaTimeSteppers.Callbacks

Pre-built [`DiscreteCallback`](@ref) constructors for common triggering
patterns: wall-clock time, simulation time, and step count.
"""
module Callbacks

import ClimaComms
import ..DiscreteCallback

"""
    Callbacks.initialize!(f!, integrator)

Called when the integrator starts. Extend this for your callback type `f!`
to perform setup (e.g. open files, record initial state). Default: no-op.
"""
function initialize!(f!, integrator) end

"""
    Callbacks.finalize!(f!, integrator)

Called when [`ClimaTimeSteppers.solve!`](@ref) completes. Extend this for your callback type `f!`
to perform teardown (e.g. close files, flush buffers). Default: no-op.
"""
function finalize!(f!, integrator) end


export EveryXWallTimeSeconds, EveryXSimulationTime, EveryXSimulationSteps

"""
    EveryXWallTimeSeconds(f!, Δwt, comm_ctx; atinit=false)

Trigger `f!(integrator)` every `Δwt` wall-clock seconds.

Timing is synchronized across all MPI ranks via `comm_ctx`
([ClimaComms context](https://clima.github.io/ClimaComms.jl/)).

# Arguments
- `f!`: callable `f!(integrator)` to execute
- `Δwt`: wall-clock interval in seconds
- `comm_ctx`: a `ClimaComms.AbstractCommsContext`

# Keyword Arguments
- `atinit`: if `true`, also trigger at initialization (default `false`)

[`Callbacks.initialize!`](@ref) and [`Callbacks.finalize!`](@ref) can be
extended for `typeof(f!)` to add setup/teardown behavior.
"""
function EveryXWallTimeSeconds(
    f!,
    Δwt,
    comm_ctx::ClimaComms.AbstractCommsContext;
    atinit = false,
)
    wt_next = 0.0

    function _initialize(c, u, t, integrator)
        wt = ClimaComms.allreduce(comm_ctx, time(), max)
        wt_next = wt + Δwt
        initialize!(c.affect!, integrator)
        if atinit
            c.affect!(integrator)
        end
    end

    function _finalize(c, u, t, integrator)
        finalize!(c.affect!, integrator)
    end

    function condition(u, t, integrator)
        wt = ClimaComms.allreduce(comm_ctx, time(), max)
        if wt >= wt_next
            while wt >= wt_next
                wt_next += Δwt
            end
            return true
        else
            return false
        end
    end

    DiscreteCallback(condition, f!; initialize = _initialize, finalize = _finalize)
end

"""
    EveryXSimulationTime(f!, Δt; atinit=false)

Trigger `f!(integrator)` every `Δt` simulation time units.

# Arguments
- `f!`: callable `f!(integrator)` to execute
- `Δt`: simulation time interval

# Keyword Arguments
- `atinit`: if `true`, also trigger at initialization (default `false`)

[`Callbacks.initialize!`](@ref) and [`Callbacks.finalize!`](@ref) can be
extended for `typeof(f!)` to add setup/teardown behavior.
"""
function EveryXSimulationTime(f!, Δt; atinit = false)
    t_next = zero(Δt)

    function _initialize(c, u, t, integrator)
        t_next = Δt
        initialize!(c.affect!, integrator)
        if atinit
            c.affect!(integrator)
        end
    end

    function _finalize(c, u, t, integrator)
        finalize!(c.affect!, integrator)
    end

    function condition(u, t, integrator)
        if t >= t_next
            while t >= t_next
                t_next += Δt
            end
            return true
        else
            return false
        end
    end
    DiscreteCallback(condition, f!; initialize = _initialize, finalize = _finalize)
end

"""
    EveryXSimulationSteps(f!, Δsteps; atinit=false)

Trigger `f!(integrator)` every `Δsteps` simulation steps.

# Arguments
- `f!`: callable `f!(integrator)` to execute
- `Δsteps`: number of steps between triggers

# Keyword Arguments
- `atinit`: if `true`, also trigger at initialization (default `false`)

[`Callbacks.initialize!`](@ref) and [`Callbacks.finalize!`](@ref) can be
extended for `typeof(f!)` to add setup/teardown behavior.
"""
function EveryXSimulationSteps(f!, Δsteps; atinit = false)
    steps = 0
    steps_next = 0

    function _initialize(c, u, t, integrator)
        steps = 0
        steps_next = Δsteps
        initialize!(c.affect!, integrator)
        if atinit
            c.affect!(integrator)
        end
    end

    function _finalize(c, u, t, integrator)
        finalize!(c.affect!, integrator)
    end

    function condition(u, t, integrator)
        steps += 1
        if steps >= steps_next
            steps_next += Δsteps
            return true
        else
            return false
        end
    end

    DiscreteCallback(condition, f!; initialize = _initialize, finalize = _finalize)
end

end # module
