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
function do_work!(integrator, not_generated_cache)
    for _ in 1:100_000
        CTS.not_generated_step_u!(integrator, not_generated_cache)
    end
end
problem_str = parsed_args["problem"]
prob = if problem_str=="ode_fun"
    split_linear_prob_wfact_split()
elseif problem_str=="fe"
    split_linear_prob_wfact_split_fe()
else
    error("Bad option")
end
algorithm = ARS343(NewtonsMethod(; max_iters = 2))
dt = 0.01
integrator = DiffEqBase.init(prob, algorithm; dt)
not_generated_cache = CTS.not_generated_cache(prob, algorithm)
do_work!(integrator, not_generated_cache) # compile first
import Profile
Profile.clear_malloc_data()
Profile.clear()
prof = Profile.@profile begin
    do_work!(integrator, not_generated_cache)
end

import ProfileCanvas

if haskey(ENV, "BUILDKITE_COMMIT") || haskey(ENV, "BUILDKITE_BRANCH")
    output_dir = joinpath(cts, parsed_args["job_id"])
    mkpath(output_dir)
    ProfileCanvas.html_file(joinpath(output_dir, "flame.html"))
else
    ProfileCanvas.view(Profile.fetch())
end
