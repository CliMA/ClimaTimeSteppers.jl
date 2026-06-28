export WSRK2, WSRK3

"""
    WickerSkamarockRungeKutta <: TimeSteppingAlgorithm

Class of multirate algorithms developed in [WS1998](@cite) and [WS2002](@cite),
which can be used as slow methods in [`Multirate`](@ref).

These require two additional copies of the state vector ``u``.

Available implementations are:
- [`WSRK2`](@ref)
- [`WSRK3`](@ref)
"""
abstract type WickerSkamarockRungeKutta <: TimeSteppingAlgorithm end

"""
    WickerSkamarockRungeKuttaTableau{T}

Storage for the tableau of a [`WickerSkamarockRungeKutta`](@ref) algorithm.

# Fields
- `c`: time-scaling coefficient vector (length `s`).
"""
struct WickerSkamarockRungeKuttaTableau{T <: NTuple}
    c::T
end
n_stages(::WickerSkamarockRungeKuttaTableau{T}) where {T} = n_stages_ntuple(T)

"""
    WickerSkamarockRungeKuttaCache{T, A}

Pre-allocated workspace for a [`WickerSkamarockRungeKutta`](@ref) method.

# Fields
- `tableau`: the [`WickerSkamarockRungeKuttaTableau`](@ref).
- `U`: preallocated space for the stage state.
- `F`: preallocated space for the slow tendency at each stage.
"""
struct WickerSkamarockRungeKuttaCache{T <: WickerSkamarockRungeKuttaTableau, A}
    tableau::T
    U::A
    F::A
end
function init_cache(prob::ODEProblem, alg::WickerSkamarockRungeKutta; kwargs...)
    U = zero(prob.u0)
    F = zero(prob.u0)
    return WickerSkamarockRungeKuttaCache(tableau(alg, eltype(F)), U, F)
end

n_stages(cache::WickerSkamarockRungeKuttaCache) = n_stages(cache.tableau)


function init_inner(prob, outercache::WickerSkamarockRungeKuttaCache, dt)
    OffsetODEFunction(prob.f.f1, zero(dt), one(dt), one(dt), outercache.F)
end
function update_inner!(
    innerinteg,
    outercache::WickerSkamarockRungeKuttaCache,
    f_slow,
    u,
    p,
    t,
    dt,
    i,
)

    f_offset = innerinteg.sol.prob.f
    (; c) = outercache.tableau
    U = outercache.U
    N = n_stages(outercache)

    f_slow(f_offset.x, i == 1 ? u : U, p, t + c[i] * dt)

    if i < N
        U .= u
        innerinteg.u = U
    else
        innerinteg.u = u
    end

    # Re-arm the inner integrator for this stage's substep [t, t_star]. The
    # inner integrator was built from the full outer tspan, so its tstops queue
    # still holds the outer end time; it must be cleared (as MIS and LSRK do)
    # or `solve!` would drain past t_star to the outer tf and overshoot.
    innerinteg.t = t
    empty!(innerinteg.tstops)
    t_star = i == N ? t + dt : t + c[i + 1] * dt
    push!(innerinteg.tstops, t_star)
end


"""
    WSRK2()

The 2 stage, 2nd order RK2 scheme of [WS1998](@cite).
"""
struct WSRK2 <: WickerSkamarockRungeKutta end
tableau(::WSRK2, RT) = WickerSkamarockRungeKuttaTableau((RT(0), RT(1 // 2)))

"""
    WSRK3()

The 3 stage, 2nd order (3rd order for linear problems) RK3 scheme of [WS2002](@cite).
"""
struct WSRK3 <: WickerSkamarockRungeKutta end
tableau(::WSRK3, RT) = WickerSkamarockRungeKuttaTableau((RT(0), RT(1 // 3), RT(1 // 2)))
