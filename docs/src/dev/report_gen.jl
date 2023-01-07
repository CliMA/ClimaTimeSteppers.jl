using ClimaTimeSteppers
using Test
cts_dir = pkgdir(ClimaTimeSteppers)

ENV["GKSwstype"] = "nul" # avoid displaying plots

include(joinpath(@__DIR__, "..", "plotting_utils.jl"))
include(joinpath(cts_dir, "test", "convergence_orders.jl"))
include(joinpath(cts_dir, "test", "utils.jl"))
include(joinpath(cts_dir, "test", "problems.jl"))

@testset "IMEXAlgorithm{ARK}" begin
    get_algorithm =
        (algorithm_name, test_case) ->
            IMEXAlgorithm(algorithm_name, NewtonsMethod(; max_iters = test_case.linear_implicit ? 1 : 2); mode = ARK())
    algorithm_names = map(type -> type(), [subtypes(IMEXARKAlgorithmName)..., subtypes(IMEXSSPRKAlgorithmName)...])
    test_algorithms("IMEX ARK", get_algorithm, algorithm_names, ark_analytic_nonlin_test_cts(Float64), 200)
    test_algorithms("IMEX ARK", get_algorithm, algorithm_names, ark_analytic_sys_test_cts(Float64), 400)
    test_algorithms(
        "IMEX ARK",
        get_algorithm,
        algorithm_names,
        ark_analytic_test_cts(Float64),
        40000;
        super_convergence = (ARS121(),),
    )
    test_algorithms(
        "IMEX ARK",
        get_algorithm,
        algorithm_names,
        onewaycouple_mri_test_cts(Float64),
        10000;
        num_steps_scaling_factor = 5,
    )
    test_algorithms(
        "IMEX ARK",
        get_algorithm,
        algorithm_names,
        climacore_1Dheat_test_cts(Float64),
        200;
        num_steps_scaling_factor = 4,
        numerical_reference_algorithm_name = ARS343(),
    )
    test_algorithms(
        "IMEX ARK",
        get_algorithm,
        algorithm_names,
        climacore_2Dheat_test_cts(Float64),
        200;
        num_steps_scaling_factor = 4,
        numerical_reference_algorithm_name = ARS343(),
    )
end

@testset "IMEXAlgorithm{SSPRK}" begin
    get_algorithm =
        (algorithm_name, test_case) ->
            IMEXAlgorithm(algorithm_name, NewtonsMethod(; max_iters = test_case.linear_implicit ? 1 : 2))
    algorithm_names = map(type -> type(), subtypes(IMEXSSPRKAlgorithmName))
    test_algorithms("IMEX SSPRK", get_algorithm, algorithm_names, ark_analytic_nonlin_test_cts(Float64), 200)
    test_algorithms("IMEX SSPRK", get_algorithm, algorithm_names, ark_analytic_sys_test_cts(Float64), 400)
    test_algorithms("IMEX SSPRK", get_algorithm, algorithm_names, ark_analytic_test_cts(Float64), 40000)
    test_algorithms(
        "IMEX SSPRK",
        get_algorithm,
        algorithm_names,
        onewaycouple_mri_test_cts(Float64),
        10000;
        num_steps_scaling_factor = 5,
    )
    test_algorithms(
        "IMEX SSPRK",
        get_algorithm,
        algorithm_names,
        climacore_1Dheat_test_cts(Float64),
        200;
        num_steps_scaling_factor = 4,
        numerical_reference_algorithm_name = ARS343(),
    )
    test_algorithms(
        "IMEX SSPRK",
        get_algorithm,
        algorithm_names,
        climacore_2Dheat_test_cts(Float64),
        200;
        num_steps_scaling_factor = 4,
        numerical_reference_algorithm_name = ARS343(),
    )
end
