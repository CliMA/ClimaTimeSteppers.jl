#=
using Revise; include("docs\\src\\test_problems\\diffusion_2d.jl")
=#
using ClimaCorePlots
import ClimaTimeSteppers as CTS
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
cts_dir = pkgdir(CTS)
include(joinpath(cts_dir, "test", "utils.jl"))
include(joinpath(cts_dir, "test", "problems.jl")) # Define the problem
test_problem = climacore_2Dheat_test_cts(Float64);
import OrdinaryDiffEq as ODE
alg = CTS.IMEXARKAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 2))
dt = 0.01
integrator = DiffEqBase.init(test_problem.split_prob, alg; dt)
ODE.solve!(integrator)
T_ss = integrator.u.u
import Plots
Plots.plot(T_ss)
Plots.plot!(; title="Numerical solution at t=$(integrator.t)")
Plots.savefig("sol.png")
