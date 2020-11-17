using TimeMachine, DiffEqBase, Plots, TimeMachine.Callbacks

const years = 1.0
const days = years/365
const hours = days/24

# Relaxation timescales
p = (τaa=6*hours,
    τoo=5*years,
    τao=60*days)

# How many s
Tatmos_ref(t) = 1.0+sinpi(2t)
Tocean_ref(t) = 0.0
function f(du,u,p,t)
  Tao = u[1]
  Toa = u[2]
  du[1] = -1/p.τaa*(Tao - Tatmos_ref(t)) - 1/p.τao*(Tao - Toa)
  du[2] = -1/p.τoo*(Toa - Tocean_ref(t)) + 1/p.τao*(Tao - Toa)
end

tspan = (0.0,2years)
prob = ODEProblem(f,[1.0,0.5],tspan, p)

struct RecordState
  u::Vector{Any}
  t::Vector{Any}
end
RecordState() = RecordState(Any[], Any[])

function (rs::RecordState)(integrator)
  push!(rs.u, deepcopy(integrator.u))
  push!(rs.t, integrator.t)
end
rs = RecordState()
sol = solve(prob, SSPRK33ShuOsher(), dt=0.1days, callback=EveryXSimulationSteps(rs, 1; atinit=true))
plot(rs.t, [u[i] for u in rs.u, i in 1:2])





function f_atmos(du,u,p,t)
  Tao = u[1]
  pp = p.p
  Toa = p.ocean[1] # ocean at time t
  atmos_tendency = -1/pp.τaa*(Tao - Tatmos_ref(t))
  boundary_tendency = -1/pp.τao*(Tao - Toa)
  du[1] = atmos_tendency + boundary_tendency
end

function f_ocean(du,u,p,t)
  Tao = p.atmos[1]
  # atmos at time t+dt
  # vs atmos at time t
  # vs integral of atmos flux / dt
  pp = p.p
  Toa = u[1]
  ocean_tendency = -1/pp.τoo*(Toa - Tocean_ref(t))
  boundary_tendency = + 1/pp.τao*(Tao - Toa)
  du[1] =  ocean_tendency + boundary_tendency
end

f(u) = a*u

struct CoupledFunction <: DiffEqBase.AbstractODEFunction{true}
  atmos
  ocean
end


struct CoupledAlgorithm <: TimeMachine.DistributedODEAlgorithm
  atmos
  ocean
end
struct CoupledCache
  ns_atmos
  ns_ocean
  atmosinteg
  oceaninteg
end

function TimeMachine.cache(prob, alg::CoupledAlgorithm; dt, ns_atmos, ns_ocean, kwargs...)
  atmosprob = ODEProblem(prob.f.atmos, prob.u0.atmos, prob.tspan, (p=p,ocean=prob.u0.ocean))
  atmosinteg = DiffEqBase.init(atmosprob, alg.atmos, dt=dt/ns_atmos, kwargs...)
  oceanprob = ODEProblem(prob.f.ocean, prob.u0.ocean, prob.tspan, (p=p,atmos=prob.u0.atmos))
  oceaninteg = DiffEqBase.init(oceanprob, alg.ocean, dt=dt/ns_ocean, kwargs...)
  return CoupledCache(ns_atmos, ns_ocean, atmosinteg, oceaninteg)
end

function TimeMachine.step_u!(int, cache::CoupledCache)
  # do n1 steps of atmosphere
  for i = 1:cache.ns_atmos
    step!(cache.atmosinteg)
  end
  # do n2 steps of ocean
  for i = 1:cache.ns_ocean
    step!(cache.oceaninteg)
  end
end

coupledprob = ODEProblem(CoupledFunction(f_atmos, f_ocean), (atmos=[1.0], ocean=[0.5]), tspan, p)
rs_coupled = RecordState()
sol = solve(coupledprob, CoupledAlgorithm(SSPRK33ShuOsher(),SSPRK33ShuOsher()),
  dt=5days, ns_atmos=10, ns_ocean=2, callback=EveryXSimulationSteps(rs_coupled, 1; atinit=true))
plot(rs_coupled.t, [getproperty(u, name)[1] for u in rs_coupled.u, name in (:atmos,:ocean)], yrange=(-1,2))

plot!(rs.t, [u[i] for u in rs.u, i in 1:2])