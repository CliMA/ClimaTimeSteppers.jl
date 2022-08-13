using ArgParse, JET, Test, BenchmarkTools, DiffEqBase, ClimaTimeSteppers
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
prob = if parsed_args["problem"]=="ode_fun"
    split_linear_prob_wfact_split
elseif parsed_args["problem"]=="fe"
    split_linear_prob_wfact_split_fe
else
    error("Bad option")
end
(; not_generated_integrator) = config_integrators(prob)
integrator = not_generated_integrator

JET.@test_opt CTS.not_generated_step_u!(integrator, integrator.cache)

