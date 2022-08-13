using ArgParse, Test, BenchmarkTools, DiffEqBase, ClimaTimeSteppers
import ClimaTimeSteppers as CTS
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--problem"
        help = "Problem type [`ode_fun`, `fe`]"
        arg_type = String
        default = "ode_fun"
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

if haskey(ENV, "BUILDKITE_COMMIT") || haskey(ENV, "BUILDKITE_BRANCH")

    import ChromeProfileFormat
    output_path = cts
    cpufile = problem_str * ".cpuprofile"
    ChromeProfileFormat.save_cpuprofile(joinpath(output_path, cpufile))

    if !isempty(get(ENV, "BUILDKITE", ""))
        import URIs

        print_link_url(url) = print("\033]1339;url='$(url)'\a\n")

        profiler_url(uri) = URIs.URI(
            "https://profiler.firefox.com/from-url/$(URIs.escapeuri(uri))",
        )

        # copy the file to the clima-ci bucket
        buildkite_pipeline = ENV["BUILDKITE_PIPELINE_SLUG"]
        buildkite_buildnum = ENV["BUILDKITE_BUILD_NUMBER"]
        buildkite_step = ENV["BUILDKITE_STEP_KEY"]

        profile_uri = "$buildkite_pipeline/build/$buildkite_buildnum/$buildkite_step/$cpufile"
        gs_profile_uri = "gs://clima-ci/$profile_uri"
        dl_profile_uri = "https://storage.googleapis.com/clima-ci/$profile_uri"

        # sync to bucket
        run(`gsutil cp $(joinpath(output_path, cpufile)) $gs_profile_uri`)

        # print link
        println("+++ Profiler link for '$profile_uri': ")
        print_link_url(profiler_url(dl_profile_uri))
    end
else
    import PProf
    PProf.pprof()
    # http://localhost:57599/ui/flamegraph?tf
end
