using Test, BenchmarkTools, DiffEqBase, ClimaTimeSteppers

include("problems.jl")

@testset "Generated vs. Not Generated" begin
    algorithm = ARS343(NewtonsMethod(; max_iters = 2))
    dt = 0.01
    for problem in (
        split_linear_prob_wfact_split(),
        split_linear_prob_wfact_split_fe(),
    )
        integrator = DiffEqBase.init(deepcopy(problem), algorithm; dt)
        not_generated_integrator = deepcopy(integrator)

        cache = ClimaTimeSteppers.cache(problem, algorithm)
        not_generated_cache =
            ClimaTimeSteppers.not_generated_cache(problem, algorithm)

        ClimaTimeSteppers.step_u!(integrator, cache)
        ClimaTimeSteppers.not_generated_step_u!(
            not_generated_integrator,
            not_generated_cache,
        )
        @test !(integrator.u === not_generated_integrator.u)
        @test integrator.u == not_generated_integrator.u

        benchmark = @benchmark ClimaTimeSteppers.step_u!($integrator, $cache)
        not_generated_benchmark =
            @benchmark ClimaTimeSteppers.not_generated_step_u!(
                $not_generated_integrator,
                $not_generated_cache,
            )
        @info "Generated step_u! benchmark: $benchmark"
        @info "Not generated step_u! benchmark: $not_generated_benchmark"
    end
end
