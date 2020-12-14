using DiffEqBase, TimeMachine, LinearAlgebra, Test, RecursiveArrayTools, TimeMachine.Callbacks

include("ode_tests_common.jl")

struct RecordState
  u::Vector{Any}
  t::Vector{Any}
end
RecordState() = RecordState(Any[], Any[])
  
function (rs::RecordState)(integrator)
push!(rs.u, deepcopy(integrator.u))
push!(rs.t, integrator.t)
end

# Problem Definition -- Mass on spring:
# d²x/dt² = -ωx => u1 = x, u_2 = dx/dt, du1/dt = u2, du2/dt = -ωu1
mutable struct SpringRHS{T}
  v::T
end

function (f!::SpringRHS)(du::ArrayPartition,u::ArrayPartition,p,args...)
  f!.v .= u.x[1]
  du.x[1] .= u.x[2]
  du.x[2] .= -p.ω .* f!.v
end

# function (f!::SpringRHS)(du,u,p,args...)
#   f!.v .= u[1]
#   du[1] .= u[2]
#   du[2] .= -p.ω .* f!.v
# end

function f_NL!(du,u,p,args...)
  # no NL component of this ode system
  du .= 0.0 .* u
end

function exactsolution(t,x0,v0)
  # x0 initial position, v0 initial velocity, ω spring constant
  v0./sqrt.(ω) .* sin.(sqrt.(ω) .* t) + x0 .* cos.(sqrt.(ω) .* t)
end

# Parameters and Initial Conditions
ω = [5.0] # spring constant vector for each spring
p = (ω=ω,)
x0 = [1.0] # initial position
v0 = [1.0] # initial velocity
AT = typeof(x0)
dims = size(x0)
tspan = (0.0,2.0π)
dt = 0.0001
using Plots
plot(tspan[1]:dt:tspan[2], exactsolution(tspan[1]:dt:tspan[2], x0, v0), lw=3, ls=:dash, label=["exact"])

# for (method, order) in explicit_methods
  method = LSRK54CarpenterKennedy() #LSRK144NiegemannDiehlBusch()
  u0 = ArrayPartition(copy(x0), copy(v0))
  @info method, u0
  prob = ODEProblem(SpringRHS(AT(undef, dims)), u0, tspan, p)
  rs = RecordState()
  sol = solve(prob, method; dt=dt, callback=EveryXSimulationSteps(rs, 1; atinit=true))
  plot!(rs.t, [u[1] for u in rs.u], lw=3, label=["$method"])
# end
current()

# for (method, order) in imex_methods
#   u0 = ArrayPartition(copy(x0), copy(v0))
#   @info method, u0
#   prob = SplitODEProblem(SpringRHS(AT(undef, dims)),f_NL!, u0, tspan, p)
#   rs = RecordState()
#   sol = solve(prob, method(DirectSolver); dt=dt, callback=EveryXSimulationSteps(rs, 1; atinit=true))
#   plot!(rs.t, [u[1] for u in rs.u], lw=3, label=["$method"])
# end