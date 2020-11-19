export WSRK2, WSRK3

"""
    WickerSkamarockRungeKutta <: DistributedODEAlgorithm

Class of multirate algorithms developed in [WS1998](@cite) and [WS2002](@cite),
which can be used as slow methods in [`Multirate`](@ref).

These require two additional copies of the state vector ``u``.

Available implementations are:
- [`WSRK2`](@ref)
- [`WSRK3`](@ref)
"""
abstract type WickerSkamarockRungeKutta <: DistributedODEAlgorithm end

struct WickerSkamarockRungeKuttaTableau{Nstages, RT}
  "Time-scaling coefficients c"
  c::NTuple{Nstages, RT}
end

struct WickerSkamarockRungeKuttaCache{Nstages, RT, A}
  tableau::WickerSkamarockRungeKuttaTableau{Nstages, RT}
  U::A
  F::A
end
function init_cache(prob::DiffEqBase.ODEProblem, alg::WickerSkamarockRungeKutta; kwargs...)
  U = similar(prob.u0)
  F = similar(prob.u0)
  return WickerSkamarockRungeKuttaCache(tableau(alg, eltype(F)), U, F)
end

nstages(::WickerSkamarockRungeKuttaCache{Nstages}) where {Nstages} = Nstages

function inner_dts(outercache::WickerSkamarockRungeKuttaCache, dt, fast_dt)
  tab = outercache.tableau
  if length(tab.c) == 2 # WSRK2
    Δt = dt/2
  else # WSRK3
    Δt = dt/6
  end
  sub_dt = Δt / round(Δt / fast_dt)
  return map(c -> sub_dt, tab.c)
end

function init_inner_fun(prob, outercache::WickerSkamarockRungeKuttaCache, dt)
  OffsetODEFunction(prob.f.f1, zero(dt), one(dt), one(dt), outercache.F)
end
function update_inner!(innerinteg, outercache::WickerSkamarockRungeKuttaCache,
  f_slow, u, p, t, dt, i)

  f_offset = innerinteg.prob.f
  tab = outercache.tableau
  U = outercache.U
  N = nstages(outercache)

  f_slow(f_offset.x, i == 1 ? u : U, p, t + tab.c[i]*dt)

  if i < N
    U .= u
    innerinteg.u = U
  else
    innerinteg.u = u
  end

  innerinteg.t = t
  innerinteg.tstop = i == N ? t+dt : t+tab.c[i+1]*dt
end


"""
    WSRK2()

The 2 stage, 2nd order RK2 scheme of [WS1998](@cite).
"""
struct WSRK2 <: WickerSkamarockRungeKutta
end
tableau(::WSRK2, RT) = WickerSkamarockRungeKuttaTableau((RT(0), RT(1//2)))

"""
    WSRK3()

The 3 stage, 2nd order (3rd order for linear problems) RK3 scheme of [WS2002](@cite).
"""
struct WSRK3 <: WickerSkamarockRungeKutta
end
tableau(::WSRK3, RT) = WickerSkamarockRungeKuttaTableau((RT(0), RT(1//3), RT(1//2)))
