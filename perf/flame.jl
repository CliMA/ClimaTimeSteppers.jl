using ArgParse, Test, BenchmarkTools, DiffEqBase, ClimaTimeSteppers
import ClimaTimeSteppers as CTS
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--problem"
        help = "Problem type [`ode_fun`, `fe`]"
        arg_type = String
        default = "ode_fun"
        "--job_id"
        help = "Job ID"
        arg_type = String
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return (s, parsed_args)
end
(s, parsed_args) = parse_commandline()
cts = joinpath(dirname(@__DIR__));
include(joinpath(cts, "test", "problems.jl"))
function config_integrators(problem)
    algorithm = ARS343(NewtonsMethod(; linsolve = linsolve_direct, max_iters = 2))
    dt = 0.01
    integrator = DiffEqBase.init(problem, algorithm; dt)
    not_generated_integrator = DiffEqBase.init(problem, algorithm; dt)
    integrator.cache = CTS.cache(problem, algorithm)
    not_generated_integrator.cache = CTS.not_generated_cache(problem, algorithm)
    return (; integrator_generated=integrator, not_generated_integrator)
end
function do_work!(integrator)
    for _ in 1:100_000
        CTS.not_generated_step_u!(integrator, integrator.cache)
    end
end
problem_str = parsed_args["problem"]
prob = if problem_str=="ode_fun"
    split_linear_prob_wfact_split
elseif problem_str=="fe"
    split_linear_prob_wfact_split_fe
else
    error("Bad option")
end
(; not_generated_integrator) = config_integrators(prob)
integrator = not_generated_integrator

do_work!(integrator) # compile first
import Profile
Profile.clear_malloc_data()
Profile.clear()
prof = Profile.@profile begin
    do_work!(integrator)
end

import ProfileCanvas
include("profile_canvas_patch.jl")
if haskey(ENV, "BUILDKITE_COMMIT") || haskey(ENV, "BUILDKITE_BRANCH")
    output_dir = joinpath(cts, parsed_args["job_id"])
    mkpath(output_dir)
    html_file(joinpath(output_dir, "flame.html"))
else
    ProfileCanvas.view(Profile.fetch())
end
