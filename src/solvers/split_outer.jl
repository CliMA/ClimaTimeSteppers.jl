export LieSplitOuter, TrapezoidalSplitOuter

#####
##### Step-exchange multirate outer methods.
#####
##### The slow tendency is evaluated only at whole-step states (once for
##### `LieSplitOuter`, twice for `TrapezoidalSplitOuter`); the fast system
##### integrates the full step per pass with the slow forcing frozen. This is
##### the split-explicit composition used by atmospheric dynamical cores, in
##### contrast to the stage-exchange family ([`MultirateInfinitesimalStep`](@ref),
##### [`WickerSkamarockRungeKutta`](@ref)), which re-evaluates the slow tendency
##### at every outer stage.
#####

"""
    StepExchangeOuter

Supertype for the step-exchange outer methods [`LieSplitOuter`](@ref) and
[`TrapezoidalSplitOuter`](@ref), used as the `slow` argument to
[`Multirate`](@ref).
"""
abstract type StepExchangeOuter <: TimeSteppingAlgorithm end

"""
    LieSplitOuter(complement = nothing)

First-order step-exchange outer method: freeze the slow forcing at the step
start and sub-cycle the fast system once over the whole step.

`complement` is an optional outer implicit complement `(u, p, t, dt) -> nothing`
that advances `u` in place; the outer step calls it once over `dt` before the
sub-cycle. `nothing` disables it.
"""
struct LieSplitOuter{C} <: StepExchangeOuter
    complement::C
end
LieSplitOuter() = LieSplitOuter(nothing)

"""
    TrapezoidalSplitOuter(complement = nothing)

Second-order step-exchange outer method: average the slow forcing between the
step start and a predicted step end, then sub-cycle the fast system over the
whole step with the averaged forcing.

`complement` is an optional outer implicit complement `(u, p, t, dt) -> nothing`
that advances `u` in place; the outer step brackets the sub-cycle with two
half-step calls (Strang splitting). `nothing` disables it.
"""
struct TrapezoidalSplitOuter{C} <: StepExchangeOuter
    complement::C
end
TrapezoidalSplitOuter() = TrapezoidalSplitOuter(nothing)

"""
    DualOffsetODEFunction(f, G, G_lim)

Wrap a fused explicit/limited tendency `f(du_exp, du_lim, u, p, t)` and add a
frozen forcing pair: `G` to the unlimited output and `G_lim` to the limited
output. Generalizes [`OffsetODEFunction`](@ref) to the two-output
(explicit, limited) form used by the step-exchange inner sub-cycle, so the
limited forcing flows through the inner limiter path.

`G` and `G_lim` are mutated in place by the outer step between sub-cycles, so
the wrapper refers to them rather than copying them.
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
    sub_timestep(dt, n_sub)

Sub-step size `dt / n_sub`. The identity fallback divides plain numbers; a
refined time type (e.g. `ITime`) may add an exact-division method.
"""
sub_timestep(dt, n_sub) = dt / n_sub

"""
    refine_ns(t)

Express a time in the inner integrator's units. Identity by default; a refined
time type (e.g. `ITime`) may add a method.
"""
refine_ns(t) = t

"""
    StepExchangeOuterCache{O, FR, FF, B, DT}

Workspace for a step-exchange [`Multirate`](@ref) method.

# Fields
- `outer`: the [`StepExchangeOuter`](@ref) method, with the optional
  complement.
- `freeze!`: `freeze!(G, G_lim, U, p, t)` fills the frozen forcing pair at
  state `U`.
- `fast_fn`: the fast `ClimaODEFunction`, whose `cache!` refreshes the full
  cache once per outer step and whose `cache_imp!` refreshes the sub-cycle cache.
- `G`, `G_lim`: frozen forcing pair, aliased into the inner sub-cycle's forcing.
- `G2`, `G2_lim`: second-pass forcing pair for `TrapezoidalSplitOuter`.
- `U0`: outer-step-start state.
- `fast_dt`: inner sub-step size.
"""
struct StepExchangeOuterCache{O <: StepExchangeOuter, FR, FF, B, DT}
    outer::O
    freeze!::FR
    fast_fn::FF
    G::B
    G_lim::B
    G2::B
    G2_lim::B
    U0::B
    fast_dt::DT
end

# Step-exchange family: `prob.f.f1` is the fast `ClimaODEFunction` (IMEX inner)
# and `prob.f.f2` is the forcing-freeze operation. The outer cache stores the
# frozen forcing workspace and the outer method; the inner integrator sub-cycles
# the fast function wrapped in a `DualOffsetODEFunction`.
function init_cache(
    prob::ODEProblem,
    alg::Multirate{F, <:StepExchangeOuter};
    dt,
    fast_dt,
    kwargs...,
) where {F}
    @assert prob.f isa SplitFunction
    u0 = prob.u0
    outercache = StepExchangeOuterCache(
        alg.slow,
        prob.f.f2,
        prob.f.f1,
        zero(u0),
        zero(u0),
        zero(u0),
        zero(u0),
        zero(u0),
        fast_dt,
    )
    innerfun = init_inner(prob, outercache, dt)
    innerprob = cts_remake(prob; f = innerfun)
    innerinteg = init(innerprob, alg.fast; dt = fast_dt, save = false, kwargs...)
    return MultirateCache(outercache, innerinteg)
end

function init_inner(prob, outercache::StepExchangeOuterCache, dt)
    fast_fn = outercache.fast_fn
    return ClimaODEFunction(;
        T_exp_T_lim! = DualOffsetODEFunction(
            fast_fn.T_exp_T_lim!,
            outercache.G,
            outercache.G_lim,
        ),
        T_imp! = fast_fn.T_imp!,
        cache! = fast_fn.cache_imp!,
        cache_imp! = fast_fn.cache_imp!,
        lim! = fast_fn.lim!,
        dss! = fast_fn.dss!,
        initialize_imp! = fast_fn.initialize_imp!,
    )
end

# Sub-cycle the inner problem from `u_start` over `[t, t + dt]` with the
# currently-set frozen forcing. The tstop at `t + dt` lands the final sub-step
# exactly on the step end.
function subcycle!(inner, cache_imp!, u_start, p, t, dt, fast_dt)
    cache_imp!(u_start, p, t)
    t_end = refine_ns(t + dt)
    inner.u .= u_start
    inner.t = refine_ns(t)
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
    complement = outer.complement

    isnothing(complement) || complement(u, p, t, dt)
    freeze!(G, G_lim, u, p, t)
    subcycle!(innerinteg, fast_fn.cache_imp!, u, p, t, dt, fast_dt)
    u .= innerinteg.u
    fast_fn.cache!(u, p, t + dt)
    return u
end

function step_split_outer!(int, cache, outer::TrapezoidalSplitOuter)
    (; outercache, innerinteg) = cache
    (; freeze!, fast_fn, G, G_lim, G2, G2_lim, U0, fast_dt) = outercache
    (; u, p, t, dt) = int
    cache_imp! = fast_fn.cache_imp!
    complement = outer.complement
    half = sub_timestep(dt, 2)

    isnothing(complement) || complement(u, p, t, half)
    U0 .= u
    freeze!(G, G_lim, U0, p, t)
    subcycle!(innerinteg, cache_imp!, U0, p, t, dt, fast_dt)
    freeze!(G2, G2_lim, innerinteg.u, p, t + dt)
    @. G = (G + G2) / 2
    @. G_lim = (G_lim + G2_lim) / 2
    subcycle!(innerinteg, cache_imp!, U0, p, t, dt, fast_dt)
    u .= innerinteg.u
    isnothing(complement) || complement(u, p, t + half, half)
    fast_fn.cache!(u, p, t + dt)
    return u
end
