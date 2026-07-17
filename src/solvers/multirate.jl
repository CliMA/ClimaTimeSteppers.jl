export Multirate


"""
    Multirate(fast, slow)

A multirate Runge-Kutta scheme that pairs a slow (outer) algorithm with a
fast (inner) algorithm. The problem must be a [`SplitODEProblem`](@ref) where
`f1` is the fast tendency and `f2` is the slow tendency.

The outer method's family sets the fast/slow exchange granularity. The
stage-exchange family evaluates the slow tendency at every outer stage; the
step-exchange family evaluates it only at whole-step states and freezes it while
the fast system integrates the full step.

# Arguments
- `fast`: inner algorithm (e.g. `LSRK54CarpenterKennedy()`)
- `slow`: outer algorithm, one of:
  - [`LowStorageRungeKutta2N`](@ref) (stage-exchange)
  - [`MultirateInfinitesimalStep`](@ref) (stage-exchange)
  - [`WickerSkamarockRungeKutta`](@ref) (stage-exchange)
  - [`LieSplitOuter`](@ref) (step-exchange)

Pass `fast_dt` as a keyword argument to [`init`](@ref) or [`solve`](@ref)
to set the inner timestep.

For the step-exchange family, `f1` is a full `ClimaODEFunction` (an
implicit-explicit inner sub-cycle), `f2` is `freeze!(G, G_lim, u, p, t)`, which
fills the frozen slow forcing pair, and the application derives `n_sub` and
passes `fast_dt = dt / n_sub`.

# Examples
```julia
using ClimaTimeSteppers
import ClimaTimeSteppers as CTS

# Stage-exchange: slow tendency re-evaluated at every outer stage.
prob = CTS.SplitODEProblem(f_fast, f_slow, u0, tspan, p)
alg  = Multirate(LSRK54CarpenterKennedy(), MIS3C())
sol  = CTS.solve(prob, alg; dt = 0.1, fast_dt = 0.01)

# Step-exchange: `f_fast` is a full `ClimaODEFunction`, `freeze!` fills the
# frozen slow forcing pair, and the inner sub-cycle is implicit-explicit.
prob = CTS.SplitODEProblem(f_fast, freeze!, u0, tspan, p)
alg  = Multirate(IMEXAlgorithm(ARS343(), NewtonsMethod()), LieSplitOuter())
sol  = CTS.solve(prob, alg; dt = 0.1, fast_dt = 0.1 / 4)
```
"""
struct Multirate{F, S} <: TimeSteppingAlgorithm
    fast::F
    slow::S
end


"""
    MultirateCache{OC, II}

Pre-allocated workspace for a [`Multirate`](@ref) method.

# Fields
- `outercache`: cache for the slow (outer) algorithm `alg.slow`, advancing the
  slow tendency `f2`.
- `innerinteg`: a full sub-integrator for the fast (inner) algorithm `alg.fast`,
  advancing the fast tendency `f1` (wrapped in an [`OffsetODEFunction`](@ref)
  carrying the slow forcing) over the inner timestep `fast_dt`.
"""
struct MultirateCache{OC, II}
    outercache::OC
    innerinteg::II
end

"""
    cts_remake(prob::ODEProblem; f)

Remake an ODE problem with a new function `f`.
"""
function cts_remake(prob::ODEProblem; f)
    return ODEProblem(f, prob.u0, prob.tspan, prob.p)
end

function init_cache(
    prob::ODEProblem,
    alg::Multirate;
    dt,
    fast_dt,
    kwargs...,
)

    @assert prob.f isa SplitFunction

    # subproblems
    outerprob = cts_remake(prob; f = prob.f.f1)
    outercache = init_cache(outerprob, alg.slow)

    innerfun = init_inner(prob, outercache, dt)
    innerprob = cts_remake(prob; f = innerfun)
    # The inner integrator's substeps are never saved, so disable its saving
    # callback to keep `solve!(innerinteg)` allocation-free.
    innerinteg = init(innerprob, alg.fast; dt = fast_dt, save = false, kwargs...)
    return MultirateCache(outercache, innerinteg)
end


function step_u!(int, cache::MultirateCache)
    outercache = cache.outercache
    tab = outercache.tableau

    u = int.u
    p = int.p
    dt = int.dt
    t = int.t

    innerinteg = cache.innerinteg
    fast_dt = innerinteg.dt

    N = n_stages(outercache)
    for stage in 1:N

        update_inner!(innerinteg, outercache, int.sol.prob.f.f2, u, p, t, dt, stage)

        # solve inner problem
        #  dv/dτ .= B[s]/(C[s+1] - C[s]) .* du .+ f_fast(v,τ) τ ∈ [τ0,τ1]

        # TODO: make this more generic
        # there are 2 strategies we can use here:
        #  a. use same fast_dt for all slow stages
        #     - problems for ARK (e.g. requires expensive LU factorization)
        #  b. use different fast_dt, cache expensive ops

        solve!(innerinteg)
        innerinteg.dt = fast_dt # reset
    end
end
