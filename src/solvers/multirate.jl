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
struct Multirate{F,S} <: DistributedODEAlgorithm
    fast::F
    slow::S
end


struct MultirateCache{OC,II,SD}
    outercache::OC
    innerinteg::II
    dt_cache::SD
end

function init_cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::Multirate;
    dt, fast_dt, kwargs...)

    @assert prob.f isa DiffEqBase.SplitFunction

    # subproblems
    outerprob = DiffEqBase.remake(prob; f=prob.f.f2)
    outercache = init_cache(outerprob, alg.slow)

    sub_dts = inner_dts(outercache, dt, fast_dt)
    unique_sub_dts = unique(sub_dts)

    innerfun = init_inner_fun(prob, outercache, dt)
    innerprob = DiffEqBase.remake(prob; f=innerfun)
    innerinteg = DiffEqBase.init(innerprob, alg.fast; dt=unique_sub_dts[1], adjustfinal=false, kwargs...)

    # build dt_cache
    unique_dt_caches = [
        i == 1 ? get_dt_cache(innerinteg.cache) : init_dt_cache(innerinteg.cache, innerinteg.prob, unique_sub_dts[i])
        for i = 1:length(unique_sub_dts)]

    dt_cache = map(sub_dts) do sub_dt
        i = findfirst(==(sub_dt), unique_sub_dts)
        unique_sub_dts[i] => unique_dt_caches[i]
    end

    return MultirateCache(outercache, innerinteg, dt_cache)
end

get_dt_cache(cache::Multirate) = cache.dt_cache
function init_dt_cache(cache::Multirate, prob, dt)
    outercache = cache.outercache
    innerinteg = cache.innerinteg

    fast_dt = innerinteg.dt # TODO: get the original fast_dt from somewhere

    sub_dts = inner_dts(outercache, dt, fast_dt)
    unique_sub_dts = unique(sub_dts)

    unique_dt_caches = [
        init_dt_cache(innerinteg.cache, innerinteg.prob, unique_sub_dts[i])
        for i = 1:length(unique_sub_dts)]

    dt_cache = map(sub_dts) do sub_dt
        i = findfirst(==(sub_dt), unique_sub_dts)
        unique_sub_dts[i] => unique_dt_caches[i]
    end
    return dt_cache
end
adjust_dt!(cache::Multirate, dt, dt_cache::Tuple) = cache.dt_cache


function step_u!(int, cache::MultirateCache)
    outercache = cache.outercache
    tab = outercache.tableau

    u = int.u
    p = int.prob.p
    dt = int.dt
    t = int.t

    innerinteg = cache.innerinteg
    fast_dt = innerinteg.dt

    for i in 1:nstages(outercache)
        sub_dt, sub_dt_cache = cache.dt_cache[i]
        adjust_dt!(innerinteg, sub_dt, sub_dt_cache)
        update_inner!(innerinteg, outercache, int.prob.f.f2, u, p, t, dt, i)
        DiffEqBase.solve!(innerinteg)
    end
end

# interface
"""
    nstages(outercache::AC)

The number of stages of the algorithm determined by cache type `AC`. This should
be defined for any algorithm cache type `AC` used as an outer solver.
"""
function nstages end


"""
    inner_dts(outercache::AC, dt, fast_dt)

The inner timesteps that will be used at each stage of the multirate procedure.

This should be defined for any algorithm cache type `AC` that will be used as an
outer solver, and should return a tuple of the length of the number of stages.
Each value will be approximately `fast_dt`, but rounded so that an integer
number of steps can be used at each outer stage (where `dt` is the slow time
step).
"""
function inner_dts end

"""
    init_inner_fun(prob, outercache::AC, dt)

Construct the inner `ODEFunction` that will be used with inner solver. This
should be defined for any algorithm cache type `AC` that will be used as an
outer solver.
"""
function init_inner_fun end

"""
    update_inner!(innerinteg, outercache::AC, f_slow, u, p, t, dt, i)

Update the inner integrator `innerinteg` for stage `i` of the outer algorithm.
This should be defined for any `outercache` type `AC`, and will typically modify:
- `innerinteg.prob.f`
- `innerinteg.u`
- `innerinteg.t`
- `innerinteg.tstop`
"""
function update_inner! end