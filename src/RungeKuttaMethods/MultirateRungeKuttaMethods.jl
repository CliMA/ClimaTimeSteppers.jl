
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

    # Only supported choice for now
    # TODO: figure out generic RK interface
    @assert alg.slow isa LowStorageRungeKutta2N 

    # subproblems
    outerprob = DiffEqBase.remake(prob; f=prob.f.f2)
    outercache = cache(outerprob, alg.slow)

    innerfun = OffsetODEFunction(prob.f.f1, zero(dt), outercache.du)
    innerprob = DiffEqBase.remake(prob; f=innerfun)
    innerinteg = DiffEqBase.init(innerprob, alg.fast; dt=fast_dt, kwargs...)
    return MultirateRungeKuttaCache(outercache, innerinteg)
end


function step_u!(int, cache::MultirateRungeKuttaCache{OC}) where {OC <: LowStorageRungeKutta2NIncCache}
    outercache = cache.outercache
    tab = outercache.tableau
    du = outercache.du

    u = int.u
    p = int.prob.p
    dt = int.dt
    τ0 = t = int.t

    innerinteg = cache.innerinteg
    fast_dt = innerinteg.dt
    
    N = nstages(outercache)
    for stage in 1:N
        # update offset
        #  du .= f(u, p, t + tab.C[stage]*dt) .+ tab.A[stage] .* du        
        int.prob.f.f2(du, u, p, τ0, 1, tab.A[stage])

        # solve inner problem
        #  dv/dτ .= B[s]/(C[s+1] - C[s]) .* du .+ f_fast(v,τ) τ ∈ [τ0,τ1]
        τ1 = stage == N ? t+dt : t+tab.C[stage+1]*dt        
        #@show (stage, τ0, τ1)

        Δτ = τ1 - τ0
        innerinteg.prob.f.γ = tab.B[stage] * (dt / Δτ)

        # approximate number of steps
        nsubsteps = cld(Δτ, fast_dt)
        innerinteg.dt = Δτ/nsubsteps
        for i = 1:nsubsteps
            # @show (i, innerinteg.t)
            # don't call step! as we don't want to invoke callbacks
            step_u!(cache.innerinteg)
            innerinteg.t += innerinteg.dt
        end
        τ0 = τ1
    end
    innerinteg.dt = fast_dt # reset
end
