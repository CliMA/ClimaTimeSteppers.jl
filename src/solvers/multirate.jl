export Multirate


"""
    Multirate(fast, slow)

A multirate Runge-Kutta scheme that pairs a slow (outer) algorithm with a
fast (inner) algorithm. The problem must be a [`SplitODEProblem`](@ref) where
`f1` is the fast tendency and `f2` is the slow tendency.

# Arguments
- `fast`: inner algorithm (e.g. `LSRK54CarpenterKennedy()`)
- `slow`: outer algorithm — must be one of:
  - [`LowStorageRungeKutta2N`](@ref)
  - [`MultirateInfinitesimalStep`](@ref)
  - [`WickerSkamarockRungeKutta`](@ref)

Pass `fast_dt` as a keyword argument to [`init`](@ref) or [`solve`](@ref)
to set the inner timestep.

# Example
```julia
using ClimaTimeSteppers
import ClimaTimeSteppers as CTS

prob = CTS.SplitODEProblem(f_fast, f_slow, u0, tspan, p)
alg  = Multirate(LSRK54CarpenterKennedy(), MIS3C())
sol  = CTS.solve(prob, alg; dt = 0.1, fast_dt = 0.01)
```
"""
struct Multirate{F, S} <: TimeSteppingAlgorithm
    fast::F
    slow::S
end


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
    innerinteg = init(innerprob, alg.fast; dt = fast_dt, kwargs...)
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
