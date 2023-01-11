using ClimaTimeSteppers
using Test
using InteractiveUtils: subtypes

ENV["GKSwstype"] = "nul" # avoid displaying plots

include(joinpath(@__DIR__, "..", "plotting_utils.jl"))
include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl"))

all_subtypes(::Type{T}) where {T} = isabstracttype(T) ? vcat(all_subtypes.(subtypes(T))...) : [T]

@testset "IMEX Algorithm Convergence" begin
    title = "IMEX Algorithms"
    algorithm_names = map(T -> T(), all_subtypes(ClimaTimeSteppers.IMEXAlgorithmName))
    test_imex_algorithms(title, algorithm_names, ark_analytic_nonlin_test_cts(Float64), 200)
    test_imex_algorithms(title, algorithm_names, ark_analytic_sys_test_cts(Float64), 400)
    test_imex_algorithms(title, algorithm_names, ark_analytic_test_cts(Float64), 40000; super_convergence = (ARS121(),))
    test_imex_algorithms(
        title,
        algorithm_names,
        onewaycouple_mri_test_cts(Float64),
        10000;
        num_steps_scaling_factor = 5,
    )
    test_imex_algorithms(
        title,
        algorithm_names,
        climacore_1Dheat_test_cts(Float64),
        200;
        num_steps_scaling_factor = 4,
        numerical_reference_algorithm_name = ARS343(),
    )
    test_imex_algorithms(
        title,
        algorithm_names,
        climacore_2Dheat_test_cts(Float64),
        200;
        num_steps_scaling_factor = 4,
        numerical_reference_algorithm_name = ARS343(),
    )
end

@testset "Unconstrained vs SSPConstrained (no limiters)" begin
    algorithm_names = map(T -> T(), all_subtypes(ClimaTimeSteppers.IMEXSSPRKAlgorithmName))
    for (test_case, num_steps) in (
        (ark_analytic_nonlin_test_cts(Float64), 200),
        (ark_analytic_sys_test_cts(Float64), 400),
        (ark_analytic_test_cts(Float64), 40000),
        (onewaycouple_mri_test_cts(Float64), 10000),
        (climacore_1Dheat_test_cts(Float64), 200),
        (climacore_2Dheat_test_cts(Float64), 200),
    )
        prob = test_case.split_prob
        dt = test_case.t_end / num_steps
        newtons_method = NewtonsMethod(; max_iters = test_case.linear_implicit ? 1 : 2)
        for algorithm_name in algorithm_names
            algorithm = IMEXAlgorithm(algorithm_name, newtons_method)
            reference_algorithm = IMEXAlgorithm(algorithm_name, newtons_method, Unconstrained())
            solution = solve(deepcopy(prob), algorithm; dt).u[end]
            reference_solution = solve(deepcopy(prob), reference_algorithm; dt).u[end]
            @test norm(solution .- reference_solution) / norm(reference_solution) < 30 * eps(Float64)
        end
    end
end
