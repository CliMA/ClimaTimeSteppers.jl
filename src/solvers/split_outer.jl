export LieSplitOuter, TrapezoidalSplitOuter

#####
##### Step-exchange multirate outer methods. See
##### docs/src/algorithm_formulations/mrrk.md ("Step-exchange (split-explicit) methods").
#####

"""
    StepExchangeOuter

Supertype of the step-exchange outer methods, which are passed as `slow` to [`Multirate`](@ref).

Subtypes:
- [`LieSplitOuter`](@ref): first order, one slow evaluation per step.
- [`TrapezoidalSplitOuter`](@ref): second order, two slow evaluations per step.
"""
abstract type StepExchangeOuter <: TimeSteppingAlgorithm end

"""
    LieSplitOuter(complement = nothing; fast_step! = nothing, sub_cycle_start! = nothing)

First-order step-exchange outer method for [`Multirate`](@ref).

Each step evaluates the slow forcing once, at the step start state, and sub-cycles the fast
system over the whole step with that forcing held fixed.

# Arguments
- `complement`: outer implicit complement `complement(u, p, t, dt)`, which advances `u` in
  place. It is called once per step, over `dt`, before the sub-cycle. By default, `nothing`
  (no call).

# Keyword Arguments
- `fast_step!`: state update `fast_step!(u, p, t, sub_dt)`, which advances `u` in place after
  each inner step. `t` is the end time and `sub_dt` the length of that inner step. By
  default, `nothing` (no call).
- `sub_cycle_start!`: hook `sub_cycle_start!(u, p, t)`, called with the state that the
  sub-cycle starts from, after `freeze!` and before the first inner step. By default,
  `nothing` (no call).

The requirements on `fast_step!` are listed in [Optional callables](@ref).

# Examples
```julia
alg = Multirate(IMEXAlgorithm(ARS343(), NewtonsMethod()), LieSplitOuter())
```

See also [`TrapezoidalSplitOuter`](@ref).
"""
struct LieSplitOuter{C, S, R} <: StepExchangeOuter
    complement::C
    fast_step!::S
    sub_cycle_start!::R
end
LieSplitOuter(complement = nothing; fast_step! = nothing, sub_cycle_start! = nothing) =
    LieSplitOuter(complement, fast_step!, sub_cycle_start!)

"""
    TrapezoidalSplitOuter(complement = nothing; fast_step! = nothing, sub_cycle_start! = nothing)

Second-order step-exchange outer method for [`Multirate`](@ref).

Each step evaluates the slow forcing at the step start state, sub-cycles to a predicted end
state, and evaluates the slow forcing again there. It then sub-cycles again from the step start
state with the average of the two evaluations.

# Arguments
- `complement`: outer implicit complement `complement(u, p, t, dt)`, which advances `u` in
  place. It is called over `dt / 2` before and after the sub-cycles (Strang splitting), so an
  `ITime` step must divide exactly by 2. By default, `nothing` (no call).

# Keyword Arguments
- `fast_step!`: state update `fast_step!(u, p, t, sub_dt)`, which advances `u` in place after
  each inner step of both sub-cycles. `t` is the end time and `sub_dt` the length of that
  inner step. Only the updates of the second sub-cycle remain in the step result. By default,
  `nothing` (no call).
- `sub_cycle_start!`: hook `sub_cycle_start!(u, p, t)`, called before each of the two
  sub-cycles with the step start state. By default, `nothing` (no call).

The requirements on `fast_step!` and the order of all calls are listed in
[Optional callables](@ref).

# Examples
```julia
alg = Multirate(IMEXAlgorithm(ARS343(), NewtonsMethod()), TrapezoidalSplitOuter())
```

See also [`LieSplitOuter`](@ref).
"""
struct TrapezoidalSplitOuter{C, S, R} <: StepExchangeOuter
    complement::C
    fast_step!::S
    sub_cycle_start!::R
end
TrapezoidalSplitOuter(
    complement = nothing; fast_step! = nothing, sub_cycle_start! = nothing,
) = TrapezoidalSplitOuter(complement, fast_step!, sub_cycle_start!)

"""
    DualOffsetODEFunction(f, G, G_lim)

Fused explicit and limited tendency `f` with a frozen forcing pair added to its outputs.

Calling it as `o(du_exp, du_lim, u, p, t)` evaluates `f(du_exp, du_lim, u, p, t)`, then adds
`G` to `du_exp` and `G_lim` to `du_lim`, so the inner limiter acts on the limited forcing.
`G` and `G_lim` are held by reference, and the outer step updates them in place between
sub-cycles. It generalizes [`OffsetODEFunction`](@ref) to two outputs.
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

Workspace of a step-exchange [`Multirate`](@ref) method.

# Fields
- `outer`: the [`StepExchangeOuter`](@ref) method, which holds the optional callables.
- `freeze!`: `freeze!(G, G_lim, U, p, t)`, which fills the frozen forcing pair at state `U`.
- `fast_fn`: the fast `ClimaODEFunction`. Its `cache!` runs at whole-step states, its
  `cache_imp!` inside the sub-cycle, and its `constrain_state!` once per step when
  `update_constrain_state` signals `EndOfStepSignal`.
- `G`, `G_lim`: frozen slow forcing pair, which the inner sub-cycle adds to its tendencies.
- `G2`, `G2_lim`: frozen slow forcing pair at the predicted end state of
  `TrapezoidalSplitOuter`, or `nothing`.
- `U0`: step start state of `TrapezoidalSplitOuter`, or `nothing`.
- `fast_dt`: length of an inner step.
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

Allocate the buffers `(G2, G2_lim, U0)` that the second sub-cycle of `outer` needs.

`TrapezoidalSplitOuter` gets three arrays shaped like `u0`, and `LieSplitOuter`, which has no
second sub-cycle, gets `(nothing, nothing, nothing)`. Called from `init_cache`.
"""
second_pass_buffers(::LieSplitOuter, u0) = (nothing, nothing, nothing)
second_pass_buffers(::TrapezoidalSplitOuter, u0) = (zero(u0), zero(u0), zero(u0))

function init_cache(
    prob::ODEProblem, alg::Multirate{F, <:StepExchangeOuter}; dt, fast_dt, kwargs...,
) where {F}
    @assert prob.f isa SplitFunction
    u0 = prob.u0
    G2, G2_lim, U0 = second_pass_buffers(alg.slow, u0)
    outercache = StepExchangeOuterCache(
        alg.slow, prob.f.f2, prob.f.f1,
        zero(u0), zero(u0), G2, G2_lim, U0,
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
            fast_fn.T_exp_T_lim!, outercache.G, outercache.G_lim,
        ),
        T_imp! = fast_fn.T_imp!, T_post_imp! = fast_fn.T_post_imp!,
        cache! = fast_fn.cache_imp!, cache_imp! = fast_fn.cache_imp!,
        lim! = fast_fn.lim!, dss! = fast_fn.dss!,
        # `step_split_outer!` applies `constrain_state!` once per outer step.
        initialize_imp! = fast_fn.initialize_imp!,
    )
end

"""
    subcycle!(inner, cache_imp!, fast_step!, u_start, p, t, dt, fast_dt)

Advance the inner integrator from `u_start` over `[t, t + dt]`, and return its state.

The inner integrator takes steps of `fast_dt`, and a stop at `t + dt` shortens the final step
to end there. When `fast_step!` is not `nothing`, each inner step is followed by
`fast_step!(inner.u, p, inner.t, inner.dt)` and then by `cache_imp!(inner.u, p, inner.t)`, so
that the cache matches the updated state. Called from `step_split_outer!`.
"""
function subcycle!(inner, cache_imp!, fast_step!, u_start, p, t, dt, fast_dt)
    cache_imp!(u_start, p, t)
    t_end = t + dt
    inner.u .= u_start
    inner.t = t
    set_dt!(inner, fast_dt)
    empty!(inner.tstops)
    add_tstop!(inner, t_end)
    while inner.t < t_end
        step!(inner)
        if !isnothing(fast_step!)
            fast_step!(inner.u, p, inner.t, inner.dt)
            cache_imp!(inner.u, p, inner.t)
        end
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
    (; complement, fast_step!, sub_cycle_start!) = outer

    isnothing(complement) || complement(u, p, t, dt)
    freeze!(G, G_lim, u, p, t)
    isnothing(sub_cycle_start!) || sub_cycle_start!(u, p, t)
    subcycle!(innerinteg, fast_fn.cache_imp!, fast_step!, u, p, t, dt, fast_dt)
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
    (; complement, fast_step!, sub_cycle_start!) = outer

    # `dt / 2` is evaluated only with a complement, because the division of an
    # `ITime` can fail.
    isnothing(complement) || complement(u, p, t, dt / 2)
    U0 .= u
    freeze!(G, G_lim, U0, p, t)
    isnothing(sub_cycle_start!) || sub_cycle_start!(U0, p, t)
    subcycle!(innerinteg, cache_imp!, fast_step!, U0, p, t, dt, fast_dt)
    freeze!(G2, G2_lim, innerinteg.u, p, t + dt)
    @. G = (G + G2) / 2
    @. G_lim = (G_lim + G2_lim) / 2
    fast_fn.cache!(U0, p, t)
    isnothing(sub_cycle_start!) || sub_cycle_start!(U0, p, t)
    subcycle!(innerinteg, cache_imp!, fast_step!, U0, p, t, dt, fast_dt)
    u .= innerinteg.u
    if !isnothing(complement)
        fast_fn.cache!(u, p, t + dt)
        complement(u, p, t + dt / 2, dt / 2)
    end
    needs_update!(fast_fn.update_constrain_state, EndOfStepSignal()) &&
        fast_fn.constrain_state!(u, p, t + dt)
    fast_fn.cache!(u, p, t + dt)
    return u
end
