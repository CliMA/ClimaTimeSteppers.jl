#=
Benchmark a single IMEX step on CPU or GPU.

Measures wall-clock time, allocations, and counts of each callback
(lim!, dss!, cache!, cache_imp!) using CTS.benchmark_step.

Usage:
    julia --project=perf perf/benchmark.jl
    julia --project=perf perf/benchmark.jl --problem ark_sys
=#
using ArgParse, DiffEqBase, ClimaTimeSteppers
using ClimaComms
using CUDA, BenchmarkTools, OrderedCollections, StatsBase, PrettyTables
import ClimaTimeSteppers as CTS

# ── Command-line interface ──────────────────────────────────────────────── #

function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--problem"
        help = "Problem name: 'ark_sys' or 'diffusion2d'"
        arg_type = String
        default = "diffusion2d"
    end
    return ArgParse.parse_args(ARGS, s)
end

parsed_args = parse_commandline()

# ── Problem setup ────────────────────────────────────────────────────────── #

cts_root = dirname(@__DIR__)
include(joinpath(cts_root, "test", "problems.jl"))

function make_integrator(problem)
    algorithm = CTS.IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 2))
    integrator = DiffEqBase.init(problem, algorithm; dt = 0.01)
    integrator.cache = CTS.init_cache(problem, algorithm)
    return integrator
end
make_integrator(itc::IntegratorTestCase) = make_integrator(itc.split_prob)

# Select the test problem
test_case = if parsed_args["problem"] == "diffusion2d"
    finitediff_2Dheat_test_cts(Float64)
elseif parsed_args["problem"] == "ark_sys"
    ark_analytic_sys_test_cts(Float64)
else
    error("Unknown problem '$(parsed_args["problem"])'. Valid: diffusion2d, ark_sys")
end

integrator = make_integrator(test_case)

# ── Benchmark ────────────────────────────────────────────────────────────── #

device = ClimaComms.device()
CTS.benchmark_step(integrator, device)
