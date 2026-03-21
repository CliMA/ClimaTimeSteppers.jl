#=
JET optimization analysis for ClimaTimeSteppers step functions.

Runs JET.@test_opt on step_u! and __step! to detect type-instabilities and
unresolvable dispatches. Also measures allocations as a regression check.

Usage:
    julia --project=perf perf/jet.jl --problem ark_sys
    julia --project=perf perf/jet.jl --problem diffusion2d
=#
using ArgParse, JET, Test, BenchmarkTools, ClimaTimeSteppers
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

# Minimal no-op callback functors (must be concrete types for JET analysis)
struct Foo end
(::Foo)(integrator) = nothing
struct Bar end
(::Bar)(integrator) = nothing

"""
    discrete_cb(cb!, n)

Create a `DiscreteCallback` with a concrete-typed `cb!` functor.
When `n == 1`, the condition always fires; otherwise it fires randomly.
This exercises both code paths during JET analysis.
"""
function discrete_cb(cb!, n)
    cond = if n == 1
        (u, t, integrator) -> isnothing(cb!(integrator))
    else
        (u, t, integrator) -> isnothing(cb!(integrator)) || rand() ≤ 0.5
    end
    CTS.DiscreteCallback(cond, cb!)
end

function make_integrator(problem)
    algorithm = CTS.IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 2))
    callbacks = CTS.CallbackSet(
        discrete_cb(Foo(), 0), discrete_cb(Bar(), 0),
        discrete_cb(Foo(), 1), discrete_cb(Bar(), 1),
    )
    integrator = CTS.init(problem, algorithm; dt = 0.01, callback = callbacks)
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

# ── JET analysis ─────────────────────────────────────────────────────────── #

@testset "JET / allocations" begin
    # Warm-up (compile all code paths)
    CTS.step_u!(integrator, integrator.cache)
    step_allocs = @allocated CTS.step_u!(integrator, integrator.cache)
    @show step_allocs

    # Check for type instabilities and unresolvable dispatches
    JET.@test_opt CTS.step_u!(integrator, integrator.cache)

    CTS.__step!(integrator)
    JET.@test_opt CTS.__step!(integrator)
end
