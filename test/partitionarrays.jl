using DiffEqBase, TimeMachine, LinearAlgebra, Test, RecursiveArrayTools, TimeMachine.Callbacks
using Plots

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

# LSRK & SSPRK ArrayPartition 
# *** Problem Definition -- Mass on spring ***
# d²x/dt² = -ωx => u1 = x, u_2 = dx/dt, du1/dt = u2, du2/dt = -ωu1
mutable struct SpringRHS{T}
  v::T
end

function (f!::SpringRHS)(du::ArrayPartition, u::ArrayPartition, p, t, α=1.0, β=0.0, args...)
  f!.v .= u.x[1]
  du.x[1] .= α .* u.x[2] .+ β .* du.x[1]
  du.x[2] .= α .* -p.ω .* f!.v .+ β .* du.x[2]
end

function spring_sol(u0, p, t)
  # x0 = u0.x[1] initial position, v0 = u0.x[2] initial velocity, ω spring constant
  u0.x[2]./sqrt.(p.ω) .* sin.(sqrt.(p.ω) .* t) + u0.x[1] .* cos.(sqrt.(p.ω) .* t)
end

# Parameters and Initial Conditions
ω = [5.0] # spring constant vector for each spring
p = (ω=ω,)
x0 = [1.0] # initial position
v0 = [1.0] # initial velocity
AT = typeof(x0)
dims = size(x0)
tspan = (0.0,2.0π)
dt = 0.001
plot(tspan[1]:dt:tspan[2]+dt, spring_sol(ArrayPartition(x0, v0), p, tspan[1]:dt:tspan[2]+dt),
    lw=3, label=["exact"], legend=:bottomright)

for (method, order) in explicit_methods
  u0 = ArrayPartition(copy(x0), copy(v0))
  @info method
  prob = ODEProblem(SpringRHS(AT(undef, dims)), u0, tspan, p)
  rs = RecordState()
  sol = solve(prob, method; dt=dt, callback=EveryXSimulationSteps(rs, 1; atinit=true))
  plot!(rs.t, [u[1] for u in rs.u], lw=3, ls=:dash, label=["$method"])
end
current()

# IMEX ArrayPartition
  # *** Problem Definition ***
  # autonomous and non-autonomous imex from problems.jl

  p = (α=4.0,)
  x0 = [0.5]
  y0 = [0.5]
  tspan = (0.0,1.0)
  dt = 0.01

  function f_im(du, u::ArrayPartition, p, t, α=true, β=false)
    @. du.x[1] = α * cos(t) * u.x[1] + β * du.x[1] # nonautonomous
    @. du.x[2] = α * u.x[2] + β * du.x[2] # autonmous
  end

  # Needed for direct solver - which takes arrays, not partition arrays
  function f_im(du, u, p, t, α=true, β=false)
    du[1] = α * cos(t) * u[1] + β * du[1]
    du[2] = α * u[2] + β * du[2]
  end

  function f_ex(du, u, p, t, α=true, β=false)
    @. du = α * cos(t)/p.α + β * du
  end

  function imex_sol(u0,p,t)
    x = @. (exp(sin(t)) * (1 + p.α * u0) - 1) / p.α # nonautonomous_sol
    y = @. (exp(t) + sin(t) - cos(t)) / 2p.α + exp(t) * u0 # autonomous sol
    return x, y
  end
  plot(tspan[1]:dt:tspan[2], [x for x in imex_sol(x0, p, tspan[1]:dt:tspan[2])],
      lw=3, label=["true nonautonomous" "true autonomous"], legend=:topleft)

  for (method, order) in imex_methods
    # method = ARK1ForwardBackwardEuler
    u0 = ArrayPartition(copy(x0), copy(y0))
    @info method, u0
    prob = SplitODEProblem(f_im, f_ex, u0, tspan, p)
    rs = RecordState()
    sol = solve(prob, method(DirectSolver); dt=dt, callback=EveryXSimulationSteps(rs, 1; atinit=true))
    plot!(rs.t, [u[i] for u in rs.u, i in 1:2], lw=3, ls=:dash, label=["$method nonautonomous" "$method autonomous"])
  end
current()

# Multirate ArrayPartition
  # From ode_tests_basic.jl "MRRK methods with 2 rates" test set
a = 100
b = 1
c = 1 / 100
Δ = sqrt(4 * a * c - b^2)
α1, α2 = 1 / 4, 3 / 4

function f_slow!(dQ, Q, p, t, α=1., β=0., args...)
    @. dQ = α * α2 * cos(t) * (a + b * Q + c * Q^2) + β * dQ
end

function f_fast!(dQ, Q, p, t, α=1., β=0., args...)
    @. dQ = α * α1 * cos(t) * (a + b * Q + c * Q^2) + β * dQ
end

function mrrk_exact_sol(q0, t0, t)
  k = @. 2 * atan((2 * c * q0 + b) / Δ) / Δ - sin(t0)
  solution = @. (Δ * tan((k + sin(t)) * Δ / 2) - b) / (2 * c)
  return Array(solution)
end

p = ()
x0 = range(-10.0, 10.0, length = 5)
tspan = (0.1,1.1)
dt = 2.0^(-5)
fast_dt = dt / 2
plot(tspan[1]:dt:tspan[2],
    [mrrk_exact_sol(x0[i], tspan[1], tspan[1]:dt:tspan[2]) for i in 1:length(x0)],
    lw=3, label = "")

  for (fast, order) in fast_mrrk_methods
    for (slow, order) in (slow_mrrk_methods..., mis_methods..., wickerskamarock_methods...)
  u0 = ArrayPartition((map(x -> [x], x0))...)
  @info fast, slow
  prob = SplitODEProblem(
    IncrementingODEFunction{true}(f_fast!),
    IncrementingODEFunction{true}(f_slow!),
    u0, tspan, p)
  method = Multirate(fast(), slow())
  rs = RecordState()
  sol = solve(prob, method; dt = dt, fast_dt = fast_dt, callback=EveryXSimulationSteps(rs, 1; atinit=true))
  plot!(rs.t, [u[i] for u in rs.u, i in 1:length(x0)], lw=3, ls=:dash, label="")
  end
end
current()
