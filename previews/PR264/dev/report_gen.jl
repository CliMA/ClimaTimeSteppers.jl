#=
julia --project=docs
using Revise; include(joinpath("docs", "src", "dev", "report_gen.jl"))
=#
using ClimaTimeSteppers
using Test
using InteractiveUtils: subtypes

ENV["GKSwstype"] = "nul" # avoid displaying plots

include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl"))
include(joinpath(@__DIR__, "..", "plotting_utils.jl"))

all_subtypes(::Type{T}) where {T} = isabstracttype(T) ? vcat(all_subtypes.(subtypes(T))...) : [T]

let # Convergence
    title = "All Algorithms"
    algorithm_names = map(T -> T(), all_subtypes(ClimaTimeSteppers.AbstractAlgorithmName))

    verify_convergence(title, algorithm_names, ark_analytic_nonlin_test_cts(Float64), 300)
    verify_convergence(title, algorithm_names, ark_analytic_sys_test_cts(Float64), 350)
    verify_convergence(title, algorithm_names, onewaycouple_mri_test_cts(Float64), 2000)
    verify_convergence(
        title,
        algorithm_names,
        ark_analytic_test_cts(Float64),
        16650;
        num_test_points = 6,
        num_steps_scaling_factor = 23,
        super_convergence = (ARS121(),),
    )

    verify_convergence(
        title,
        algorithm_names,
        climacore_1Dheat_test_cts(Float64),
        40;
        numerical_reference_algorithm_name = ARK548L2SA2(),
        numerical_reference_num_steps = 500000,
    )
    verify_convergence(
        title,
        algorithm_names,
        climacore_2Dheat_test_cts(Float64),
        40;
        numerical_reference_algorithm_name = ARK548L2SA2(),
        numerical_reference_num_steps = 500000,
    )

    verify_convergence(
        title,
        algorithm_names,
        climacore_1Dheat_test_implicit_cts(Float64),
        60;
        num_test_points = 4,
        num_steps_scaling_factor = 8,
        numerical_reference_algorithm_name = ARK548L2SA2(),
        numerical_reference_num_steps = 500000,
    )
end
