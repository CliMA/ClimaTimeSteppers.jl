export LieSplitOuter, TrapezoidalSplitOuter

#####
##### Step-exchange multirate outer methods. See
##### docs/src/algorithm_formulations/mrrk.md ("Step-exchange (split-explicit) methods").
#####

"""
    StepExchangeOuter

Supertype for the step-exchange outer methods [`LieSplitOuter`](@ref) and
[`TrapezoidalSplitOuter`](@ref), used as the `slow` argument to
[`Multirate`](@ref).
"""
abstract type StepExchangeOuter <: TimeSteppingAlgorithm end

"""
    LieSplitOuter()

First-order step-exchange outer method: freeze the slow forcing at the step
start and sub-cycle the fast system once over the whole step.
"""
struct LieSplitOuter <: StepExchangeOuter end

"""
    TrapezoidalSplitOuter()

Second-order step-exchange outer method: average the slow forcing between the
step start and a predicted step end, then sub-cycle the fast system over the
whole step with the averaged forcing.
"""
struct TrapezoidalSplitOuter <: StepExchangeOuter end

"""
    DualOffsetODEFunction(f, G, G_lim)

Wrap a fused explicit/limited tendency `f(du_exp, du_lim, u, p, t)` and add a
frozen forcing pair: `G` to the unlimited output and `G_lim` to the limited
output. Generalizes [`OffsetODEFunction`](@ref) to the two-output
(explicit, limited) form used by the step-exchange inner sub-cycle, so the
inner limiter is applied to the limited forcing.

`G` and `G_lim` are aliased, not copied; the outer step mutates them in place
between sub-cycles.
"""
struct DualOffsetODEFunction{F, A}
    f::F
    G::A
    G_lim::A
end
function (o::DualOffsetODEFunction)(du_exp, du_lim, u, p, t)
    o.f(du_exp, du_lim, u, p, t)
    du_exp .+= o.G
    du_lim .+= o.G_lim
    return nothing
end

"""
    StepExchangeOuterCache{O, FR, FF, B, B2, DT}

Workspace for a step-exchange [`Multirate`](@ref) method.

# Fields
- `outer`: the [`StepExchangeOuter`](@ref) method.
- `freeze!`: `freeze!(G, G_lim, U, p, t)` fills the frozen forcing pair at
  state `U`.
- `fast_fn`: the fast `ClimaODEFunction`; `cache!` refreshes the full cache at
  whole-step states, `cache_imp!` refreshes the sub-cycle cache, and
  `constrain_state!` is applied once per outer step under its
  `update_constrain_state` handler.
- `G`, `G_lim`: frozen forcing pair, aliased into the inner sub-cycle's forcing.
- `G2`, `G2_lim`: second-pass forcing pair for `TrapezoidalSplitOuter`; `nothing`
  for `LieSplitOuter`.
- `U0`: outer-step-start state for `TrapezoidalSplitOuter`; `nothing` for
  `LieSplitOuter`.
- `fast_dt`: inner sub-step size.
"""
struct StepExchangeOuterCache{O <: StepExchangeOuter, FR, FF, B, B2, DT}
    outer::O
    freeze!::FR
    fast_fn::FF
    G::B
    G_lim::B
    G2::B2
    G2_lim::B2
    U0::B2
    fast_dt::DT
end

"""
    second_pass_buffers(outer, u0)

Allocate the second-pass forcing pair and step-start state used by the
`TrapezoidalSplitOuter` step path, returning `nothing` for `LieSplitOuter`,
which does not use them.
"""
second_pass_buffers(::LieSplitOuter, u0) = (nothing, nothing, nothing)
second_pass_buffers(::TrapezoidalSplitOuter, u0) =
    (zero(u0), zero(u0), zero(u0))

function init_cache(
    prob::ODEProblem,
    alg::Multirate{F, <:StepExchangeOuter};
    dt,
    fast_dt,
    kwargs...,
) where {F}
    @assert prob.f isa SplitFunction
    u0 = prob.u0
    G2, G2_lim, U0 = second_pass_buffers(alg.slow, u0)
    outercache = StepExchangeOuterCache(
        alg.slow,
        prob.f.f2,
        prob.f.f1,
        zero(u0),
        zero(u0),
        G2,
        G2_lim,
        U0,
        fast_dt,
    )
    innerfun = init_inner(prob, outercache)
    innerprob = cts_remake(prob; f = innerfun)
    innerinteg = init(innerprob, alg.fast; dt = fast_dt, save = false, kwargs...)
    return MultirateCache(outercache, innerinteg)
end

function init_inner(prob, outercache::StepExchangeOuterCache)
    fast_fn = outercache.fast_fn
    return ClimaODEFunction(;
        T_exp_T_lim! = DualOffsetODEFunction(
            fast_fn.T_exp_T_lim!,
            outercache.G,
            outercache.G_lim,
        ),
        T_imp! = fast_fn.T_imp!,
        T_post_imp! = fast_fn.T_post_imp!,
        cache! = fast_fn.cache_imp!,
        cache_imp! = fast_fn.cache_imp!,
        lim! = fast_fn.lim!,
        dss! = fast_fn.dss!,
        # `constrain_state!` fires once per outer step, in `step_split_outer!`.
        initialize_imp! = fast_fn.initialize_imp!,
    )
end

"""
    subcycle!(inner, cache_imp!, u_start, p, t, dt, fast_dt)

Sub-cycle the inner integrator from `u_start` over `[t, t + dt]` with the frozen
forcing. A tstop at `t + dt` aligns the final sub-step with the step end.
"""
function subcycle!(inner, cache_imp!, u_start, p, t, dt, fast_dt)
    cache_imp!(u_start, p, t)
    t_end = t + dt
    inner.u .= u_start
    inner.t = t
    set_dt!(inner, fast_dt)
    empty!(inner.tstops)
    add_tstop!(inner, t_end)
    while inner.t < t_end
        step!(inner)
    end
    return inner.u
end

function step_u!(int, cache::MultirateCache{<:StepExchangeOuterCache})
    step_split_outer!(int, cache, cache.outercache.outer)
end

function step_split_outer!(int, cache, outer::LieSplitOuter)
    (; outercache, innerinteg) = cache
    (; freeze!, fast_fn, G, G_lim, fast_dt) = outercache
    (; u, p, t, dt) = int

    freeze!(G, G_lim, u, p, t)
    subcycle!(innerinteg, fast_fn.cache_imp!, u, p, t, dt, fast_dt)
    u .= innerinteg.u
    needs_update!(fast_fn.update_constrain_state, EndOfStepSignal()) &&
        fast_fn.constrain_state!(u, p, t + dt)
    fast_fn.cache!(u, p, t + dt)
    return u
end

function step_split_outer!(int, cache, outer::TrapezoidalSplitOuter)
    (; outercache, innerinteg) = cache
    (; freeze!, fast_fn, G, G_lim, G2, G2_lim, U0, fast_dt) = outercache
    (; u, p, t, dt) = int
    cache_imp! = fast_fn.cache_imp!

    U0 .= u
    freeze!(G, G_lim, U0, p, t)
    subcycle!(innerinteg, cache_imp!, U0, p, t, dt, fast_dt)
    freeze!(G2, G2_lim, innerinteg.u, p, t + dt)
    @. G = (G + G2) / 2
    @. G_lim = (G_lim + G2_lim) / 2
    # The second-pass freeze evaluates the predicted end state; refresh the
    # full cache at the second-pass restart state before the second sub-cycle.
    fast_fn.cache!(U0, p, t)
    subcycle!(innerinteg, cache_imp!, U0, p, t, dt, fast_dt)
    u .= innerinteg.u
    needs_update!(fast_fn.update_constrain_state, EndOfStepSignal()) &&
        fast_fn.constrain_state!(u, p, t + dt)
    fast_fn.cache!(u, p, t + dt)
    return u
end
