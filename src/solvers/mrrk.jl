
export MultirateRungeKutta

"""
    MultirateRungeKutta(fast, slow)

A multirate Runge--Kutta scheme, combining `fast` and `slow` algorithms

Currently `slow` must be a LSRK method.
"""
struct MultirateRungeKutta{F,S} <: DistributedODEAlgorithm
    fast::F
    slow::S
end


struct MultirateRungeKuttaCache{OC,II}
    outercache::OC
    innerinteg::II
end

function cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::MultirateRungeKutta;
    dt, fast_dt, kwargs...)

    @assert prob.f isa DiffEqBase.SplitFunction

    # subproblems
    outerprob = DiffEqBase.remake(prob; f=prob.f.f2)
    outercache = cache(outerprob, alg.slow)

    innerfun = OffsetODEFunction(prob.f.f1, zero(dt), one(dt), zero(dt), outercache.du)
    innerprob = DiffEqBase.remake(prob; f=innerfun)
    innerinteg = DiffEqBase.init(innerprob, alg.fast; dt=fast_dt, kwargs...)
    return MultirateRungeKuttaCache(outercache, innerinteg)
end


function step_u!(int, cache::MultirateRungeKuttaCache)
    outercache = cache.outercache
    tab = outercache.tableau
    du = outercache.du

    u = int.u
    p = int.prob.p
    dt = int.dt
    t = int.t

    innerinteg = cache.innerinteg
    fast_dt = innerinteg.dt

    N = nstages(outercache)
    for stage in 1:N

        update_inner!(innerinteg, outercache, int.prob.f.f2, u, p, t, dt, stage)

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

function update_inner!(innerinteg, outercache::LowStorageRungeKutta2NIncCache,
        f_slow, u, p, t, dt, stage)

    f_offset = innerinteg.prob.f
    tab = outercache.tableau
    N = nstages(outercache)

    τ0 = t+tab.C[stage]*dt
    τ1 = stage == N ? t+dt : t+tab.C[stage+1]*dt
    f_offset.α = τ0
    innerinteg.t = zero(τ0)
    innerinteg.tstop = τ1-τ0

    #  du .= f(u, p, t + tab.C[stage]*dt) .+ tab.A[stage] .* du
    f_slow(f_offset.x, u, p, τ0, 1, tab.A[stage])

    C0 = tab.C[stage]
    C1 = stage == N ? one(tab.C[stage]) : tab.C[stage+1]
    f_offset.γ = tab.B[stage] / (C1-C0)
end