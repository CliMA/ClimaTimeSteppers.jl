using BenchmarkTools, ClimaTimeSteppers, DiffEqBase
include("problems.jl")
include("utils.jl")
algorithm = ARS343(NewtonsMethod(; linsolve = linsolve_direct, max_iters = 2))
test_step_func(integrator, step_func) = step_func(integrator, integrator.cache)
for problem in (split_linear_prob_wfact_split, split_linear_prob_wfact_split_fe)
    for (name, cache_func, step_func) in (
        ("generated: ", ClimaTimeSteppers.cache, ClimaTimeSteppers.step_u!),
        ("not generated: ", ClimaTimeSteppers.not_generated_cache, ClimaTimeSteppers.not_generated_step_u!),
    )
        integrator = DiffEqBase.init(problem, algorithm; dt = 0.01)
        integrator.cache = cache_func(problem, algorithm)
        @info name
        @btime test_step_func($integrator, $step_func)
    end
end
