export Multirate

"""
    Multirate(fast, slow)

A multirate Runge--Kutta scheme, combining `fast` and `slow` algorithms

`slow` can be any algorithm providing methods for the following functions
  - `init_inner(prob, outercache, dt)`
  - `update_inner!(innerinteg, outercache, f_slow, u, p, t, dt, stage)`
Algorithms which currently support this are:
 - [`LowStorageRungeKutta2N`](@ref)
 - [`MultirateInfinitesimalStep`](@ref)
 - [`WickerSkamarockRungeKutta`](@ref)
"""
struct Multirate{F, S} <: DistributedODEAlgorithm
    fast::F
    slow::S
end


struct MultirateCache{OC, II}
    outercache::OC
    innerinteg::II
end

function cache(prob::DiffEqBase.AbstractODEProblem, alg::Multirate; dt, fast_dt, kwargs...)

    @assert prob.f isa DiffEqBase.SplitFunction

    # subproblems
    outerprob = DiffEqBase.remake(prob; f = prob.f.f2)
    outercache = cache(outerprob, alg.slow)

    innerfun = init_inner(prob, outercache, dt)
    innerprob = DiffEqBase.remake(prob; f = innerfun)
    innerinteg = DiffEqBase.init(innerprob, alg.fast; dt = fast_dt, kwargs...)
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

    N = nstages(outercache)
    for stage in 1:N

        update_inner!(innerinteg, outercache, int.sol.prob.f.f2, u, p, t, dt, stage)

        # solve inner problem
        #  dv/dτ .= B[s]/(C[s+1] - C[s]) .* du .+ f_fast(v,τ) τ ∈ [τ0,τ1]

        # TODO: make this more generic
        # there are 2 strategies we can use here:
        #  a. use same fast_dt for all slow stages, use `adjustfinal=true`
        #     - problems for ARK (e.g. requires expensive LU factorization)
        #  b. use different fast_dt, cache expensive ops

        innerinteg.adjustfinal = true
        DiffEqBase.solve!(innerinteg)
        innerinteg.dt = fast_dt # reset
    end
end
