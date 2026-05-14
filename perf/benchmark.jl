# Example of benchmarking a single integrator step for a simple test problem

import BenchmarkTools
import CUDA

include(joinpath(dirname(@__DIR__), "test", "problems.jl"))

problem = ark_analytic_sys_test_cts(Float64).split_prob
algorithm = IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 2))
integrator = init(problem, algorithm; dt = 0.01)
ClimaTimeSteppers.benchmark_step(integrator)
