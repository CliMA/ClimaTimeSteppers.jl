#=
Flame graph profiler for ClimaTimeSteppers step_u!.

Profiles 100k steps of the 1D heat equation with ARS343 + Newton, then
either saves a flame.html artifact (on Buildkite) or opens an interactive
viewer locally.

Usage:
    julia --project=perf perf/flame.jl --job_id diffusion_1D
=#
using ArgParse, Test, BenchmarkTools, DiffEqBase, ClimaTimeSteppers
import ClimaTimeSteppers as CTS
import Profile
import ProfileCanvas

# ── Command-line interface ──────────────────────────────────────────────── #

function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--job_id"
        help = "Identifier for the output directory (used on Buildkite)"
        arg_type = String
    end
    return ArgParse.parse_args(ARGS, s)
end

parsed_args = parse_commandline()

# ── Problem setup ────────────────────────────────────────────────────────── #

cts_root = dirname(@__DIR__)
include(joinpath(cts_root, "test", "problems.jl"))

test_case = finitediff_1Dheat_test_cts(Float64)
algorithm = CTS.IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 2))
integrator = DiffEqBase.init(test_case.split_prob, algorithm; dt = 0.01)
cache = CTS.init_cache(test_case.split_prob, algorithm)

# ── Profiling ────────────────────────────────────────────────────────────── #

function do_work!(integrator, cache)
    for _ in 1:100_000
        CTS.step_u!(integrator, cache)
    end
end

do_work!(integrator, cache)  # warm-up / compilation

# Verify zero allocations after compilation
n_allocs = @allocated do_work!(integrator, cache)
@testset "Flame profiler allocations" begin
    @test n_allocs == 0
end

# Collect profile data
Profile.clear_malloc_data()
Profile.clear()
Profile.@profile do_work!(integrator, cache)

# ── Output ───────────────────────────────────────────────────────────────── #

if haskey(ENV, "BUILDKITE_COMMIT") || haskey(ENV, "BUILDKITE_BRANCH")
    # On CI: write an HTML artifact for Buildkite to upload
    output_dir = joinpath(cts_root, parsed_args["job_id"])
    mkpath(output_dir)
    ProfileCanvas.html_file(joinpath(output_dir, "flame.html"))
else
    # Locally: open the interactive flame-graph viewer
    ProfileCanvas.view(Profile.fetch())
end
