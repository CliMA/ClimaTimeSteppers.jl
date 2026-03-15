#=
Convergence testing infrastructure for documentation report generation.

This file re-uses the core convergence utilities from the test suite and adds
a documentation-specific wrapper that collects results into a Dict for
plotting by summarize_convergence.jl.
=#
using ClimaTimeSteppers

using Distributions: quantile, TDist
using LinearAlgebra: norm

# Import the shared convergence infrastructure from the test suite
include(joinpath(pkgdir(ClimaTimeSteppers), "test", "utils", "convergence_utils.jl"))

"""
    test_convergence!(convergence_results, algorithm_name, test_case, default_num_steps; kwargs...)

Documentation-specific wrapper around the core convergence testing logic.
Computes convergence orders and stores detailed results (sampled errors,
solutions at default dt, etc.) in `convergence_results` for downstream
plotting by `summarize_convergence.jl`.
"""
function test_convergence!(
    convergence_results,
    algorithm_name,
    test_case,
    default_num_steps;
    super_convergence_algorithm_names = (),
    num_steps_scaling_factor = 4,
    high_order_sample_shifts = 1,
    numerical_reference_algorithm_name = nothing,
    numerical_reference_num_steps = num_steps_scaling_factor^3 * default_num_steps,
    broken_tests = (),
    error_on_failure = true,
    verbose = false,
)
    (; test_name, t_end, linear_implicit, analytic_sol) = test_case
    prob = test_case.split_prob

    default_dt = t_end / default_num_steps
    ref_sol = if isnothing(numerical_reference_algorithm_name)
        analytic_sol
    else
        ref_alg_str = string(nameof(typeof(numerical_reference_algorithm_name)))
        ref_alg = algorithm(numerical_reference_algorithm_name, linear_implicit)
        ref_dt = t_end / numerical_reference_num_steps
        verbose &&
            @info "Generating reference solution for $test_name with $ref_alg_str and dt = $ref_dt"
        solve(deepcopy(prob), ref_alg; dt = ref_dt, save_everystep = true)
    end
    numerical_reference_info = if isnothing(numerical_reference_algorithm_name)
        nothing
    else
        ref_average_rms_error = rms(rms_error.(ref_sol.u, ref_sol.t, (analytic_sol,)))
        (ref_alg_str, ref_dt, ref_average_rms_error)
    end

    alg_str = string(nameof(typeof(algorithm_name)))
    alg = algorithm(algorithm_name, linear_implicit)
    verbose && @info "Testing convergence of $alg_str for $test_name"

    predicted_order = predicted_convergence_order(algorithm_name, prob.f)
    predicted_super_convergence = algorithm_name in super_convergence_algorithm_names
    num_steps_powers =
        (-1:0.5:1) .- high_order_sample_shifts * max(0, predicted_order - 3) / 2
    sampled_num_steps = default_num_steps .* num_steps_scaling_factor .^ num_steps_powers
    sampled_dts = t_end ./ round.(Int, sampled_num_steps)
    average_rms_errors = map(sampled_dts) do dt
        sol = solve(deepcopy(prob), alg; dt = dt, save_everystep = true)
        rms(rms_error.(sol.u, sol.t, (ref_sol,)))
    end
    verbose && @info "Sampled timesteps = $sampled_dts"
    verbose && @info "Average RMS errors = $average_rms_errors"

    # Compute a 99% confidence interval for the convergence order
    order, order_uncertainty = convergence_order(sampled_dts, average_rms_errors, 0.99)
    verbose && @info "Convergence order = $order ± $order_uncertainty"
    actual_predicted_order = predicted_order + Bool(predicted_super_convergence)
    convergence_test_error = if isnan(order)
        "Timestepper does not converge for $alg_str ($test_name)"
    elseif abs(order - actual_predicted_order) > order_uncertainty
        "Predicted order outside error bars for $alg_str ($test_name)"
    elseif order_uncertainty > actual_predicted_order / 10
        "Order uncertainty too large for $alg_str ($test_name)"
    else
        nothing
    end
    if isnothing(convergence_test_error)
        @assert !(algorithm_name in broken_tests)
    elseif error_on_failure && !(algorithm_name in broken_tests)
        error(convergence_test_error)
    else
        @warn convergence_test_error
    end

    default_dt_sol = solve(deepcopy(prob), alg; dt = default_dt, save_everystep = true)
    default_dt_times = default_dt_sol.t
    default_dt_solutions = rms.(default_dt_sol.u)
    default_dt_errors = rms_error.(default_dt_sol.u, default_dt_sol.t, (ref_sol,))

    convergence_results[test_name] = Dict()
    convergence_results[test_name]["default_dt"] = default_dt
    convergence_results[test_name]["numerical_reference_info"] = numerical_reference_info
    convergence_results[test_name]["all_alg_results"] = Dict()
    convergence_results[test_name]["all_alg_results"][alg_str] = Dict()
    alg_results = convergence_results[test_name]["all_alg_results"][alg_str]
    alg_results["predicted_order"] = predicted_order
    alg_results["predicted_super_convergence"] = predicted_super_convergence
    alg_results["sampled_dts"] = sampled_dts
    alg_results["average_rms_errors"] = average_rms_errors
    alg_results["default_dt_times"] = default_dt_times
    alg_results["default_dt_solutions"] = default_dt_solutions
    alg_results["default_dt_errors"] = default_dt_errors
    return convergence_results
end
