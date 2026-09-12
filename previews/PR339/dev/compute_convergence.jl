import JLD2
import ClimaCorePlots
import Plots
using Distributions: quantile, TDist
using Printf: @sprintf
using LaTeXStrings: latexstring
import DiffEqCallbacks
import ClimaTimeSteppers as CTS

function get_algorithm_names()
    all_subtypes(::Type{T}) where {T} = isabstracttype(T) ? vcat(all_subtypes.(subtypes(T))...) : [T]
    algorithm_names = map(T -> T(), all_subtypes(ClimaTimeSteppers.AbstractAlgorithmName))
    return filter(name -> !(name isa ARK437L2SA1 || name isa ARK548L2SA2), algorithm_names) # too high order
end

function get_imex_ssprk_algorithm_names()
    all_subtypes(::Type{T}) where {T} = isabstracttype(T) ? vcat(all_subtypes.(subtypes(T))...) : [T]
    algorithm_names = map(T -> T(), all_subtypes(ClimaTimeSteppers.IMEXSSPRKAlgorithmName))
    return algorithm_names
end

function make_saving_callback(cb, u, t, integrator)
    savevalType = typeof(cb(u, t, integrator))
    return DiffEqCallbacks.SavingCallback(cb, DiffEqCallbacks.SavedValues(typeof(t), savevalType))
end

"""
    imex_convergence_orders(algorithm_name)

Return a tuple containing the expected convergence order of the tableau when
using only an implicit tendency, the order when using only an explicit tendency,
and the order when using both tendencies.
"""
function imex_convergence_orders end
imex_convergence_orders(::ARS111) = (1, 1, 1)
imex_convergence_orders(::ARS121) = (1, 1, 1)
imex_convergence_orders(::ARS122) = (2, 2, 2)
imex_convergence_orders(::ARS222) = (2, 2, 2)
imex_convergence_orders(::ARS232) = (2, 3, 2)
imex_convergence_orders(::ARS233) = (3, 3, 3)
imex_convergence_orders(::ARS343) = (3, 4, 3)
imex_convergence_orders(::ARS443) = (3, 3, 3)
imex_convergence_orders(::Union{IMKG232a, IMKG232b}) = (2, 2, 2)
imex_convergence_orders(::Union{IMKG242a, IMKG242b}) = (2, 4, 2)
imex_convergence_orders(::IMKG243a) = (2, 4, 2)
imex_convergence_orders(::Union{IMKG252a, IMKG252b}) = (2, 2, 2)
imex_convergence_orders(::Union{IMKG253a, IMKG253b}) = (2, 2, 2)
imex_convergence_orders(::Union{IMKG254a, IMKG254b, IMKG254c}) = (2, 2, 2)
imex_convergence_orders(::IMKG342a) = (3, 4, 3)
imex_convergence_orders(::IMKG343a) = (3, 4, 3)
imex_convergence_orders(::SSP222) = (2, 2, 2)
imex_convergence_orders(::SSP322) = (2, 2, 2)
imex_convergence_orders(::SSP332) = (2, 3, 2)
imex_convergence_orders(::SSP333) = (3, 3, 3)
imex_convergence_orders(::SSP433) = (3, 3, 3)
imex_convergence_orders(::DBM453) = (3, 3, 3)
imex_convergence_orders(::HOMMEM1) = (2, 3, 2)
imex_convergence_orders(::ARK2GKC) = (2, 2, 2)
imex_convergence_orders(::ARK437L2SA1) = (4, 4, 4)
imex_convergence_orders(::ARK548L2SA2) = (5, 5, 5)
imex_convergence_orders(::SSP22Heuns) = (2, 2, 2)
imex_convergence_orders(::SSP33ShuOsher) = (3, 3, 3)
imex_convergence_orders(::RK4) = (4, 4, 4)
# SSPKnoth is not really an IMEX method
imex_convergence_orders(::SSPKnoth) = (2, 2, 2)

# Compute a confidence interval for the convergence order, returning the
# estimated convergence order and its uncertainty.
function convergence_order(dts, errs, confidence)
    log_dts = log10.(dts)
    log_errs = log10.(errs)
    n_dts = length(dts)

    # slope and vertical intercept of linear regression (for each log_dt and
    # log_err, log_err ≈ log_dt * order + log_err_of_dt1)
    order, log_err_of_dt1 = hcat(log_dts, ones(n_dts)) \ log_errs

    # number of degrees of freedom of linear regression (number of data
    # points minus number of fitted parameters)
    n_dof = n_dts - 2

    # critical value of Student's t-distribution for two-sided confidence
    # interval
    confidence_interval_t_value = -quantile(TDist(n_dof), (1 - confidence) / 2)

    # standard deviation of linear regression
    regression_standard_deviation = sqrt(sum((log_errs .- (log_dts .* order .+ log_err_of_dt1)) .^ 2) / n_dof)

    # standard deviation of slope
    order_standard_deviation = regression_standard_deviation / sqrt(sum((log_dts .- sum(log_dts) / n_dts) .^ 2))

    # "uncertainty" in slope (half of width of confidence interval)
    order_uncertainty = confidence_interval_t_value * order_standard_deviation

    return order, order_uncertainty
end

"""
    predicted_convergence_order(algorithm_name, ode_function)

Return the predicted convergence order of the algorithm for the given ODE
function (assuming that the algorithm converges).
"""
function predicted_convergence_order(algorithm_name::AbstractAlgorithmName, ode_function::AbstractClimaODEFunction)
    (imp_order, exp_order, combined_order) = imex_convergence_orders(algorithm_name)
    has_imp = !isnothing(ode_function.T_imp!)
    has_exp = CTS.has_T_exp(ode_function)
    has_imp && !has_exp && return imp_order
    !has_imp && has_exp && return exp_order
    has_imp && has_exp && return combined_order
    return 0
end

import Logging
function export_convergence_results(alg_name, test_problem, num_steps; kwargs...)
    @info "=========================== Exporting convergence for $(test_problem.test_name)..."
    out_dict = Dict()
    (; test_name) = test_problem
    out_dict[string(test_name)] = Dict()
    out_dict[string(test_name)][string(alg_name)] = Dict()
    out_dict[string(test_name)]["args"] = (alg_name, test_problem, num_steps)
    out_dict[string(test_name)]["kwargs"] = kwargs
    compute_convergence!(out_dict, alg_name, test_problem, num_steps; kwargs...)
    Logging.with_logger(Logging.NullLogger()) do
        JLD2.save_object("convergence_$(alg_name)_$(test_problem.test_name).jld2", out_dict)
    end
end


function compute_convergence!(
    out_dict,
    alg_name,
    test_case,
    num_steps;
    num_steps_scaling_factor = 10,
    order_confidence_percent = 99,
    super_convergence = (),
    numerical_reference_algorithm_name = nothing,
    numerical_reference_num_steps = num_steps_scaling_factor^3 * num_steps,
    full_history_algorithm_name = nothing,
    average_function = array -> norm(array) / sqrt(length(array)),
    average_function_str = "RMS",
    only_endpoints = false,
    verbose = false,
)
    (; test_name, t_end, linear_implicit, analytic_sol) = test_case
    prob = test_case.split_prob
    FT = typeof(t_end)
    default_dt = t_end / num_steps
    key1 = string(test_name)
    key2 = string(alg_name)

    algorithm(algorithm_name::ClimaTimeSteppers.ERKAlgorithmName) = ExplicitAlgorithm(algorithm_name)
    algorithm(algorithm_name::ClimaTimeSteppers.SSPKnoth) =
        ClimaTimeSteppers.RosenbrockAlgorithm(ClimaTimeSteppers.tableau(ClimaTimeSteppers.SSPKnoth()))
    algorithm(algorithm_name::ClimaTimeSteppers.IMEXARKAlgorithmName) =
        IMEXAlgorithm(algorithm_name, NewtonsMethod(; max_iters = linear_implicit ? 1 : 2))

    ref_sol = if isnothing(numerical_reference_algorithm_name)
        analytic_sol
    else
        ref_alg = algorithm(numerical_reference_algorithm_name)
        ref_alg_str = string(nameof(typeof(numerical_reference_algorithm_name)))
        ref_dt = t_end / numerical_reference_num_steps
        verbose &&
            @info "Generating numerical reference solution for $test_name with $ref_alg_str (dt = $ref_dt)..."
        sol = solve(deepcopy(prob), ref_alg; dt = ref_dt, save_everystep = !only_endpoints)
        out_dict[string(test_name)]["numerical_ref_sol"] = sol
    end

    cur_avg_err(u, t, integrator) = average_function(abs.(u .- ref_sol(t)))
    cur_avg_sol_and_err(u, t, integrator) = (average_function(u), average_function(abs.(u .- ref_sol(t))))

    float_str(x) = @sprintf "%.4f" x
    pow_str(x) = "10^{$(@sprintf "%.1f" log10(x))}"
    function si_str(x)
        if isnan(x) || x in (0, Inf, -Inf)
            return string(x)
        end
        exponent = floor(Int, log10(x))
        mantissa = x / 10.0^exponent
        return "$(float_str(mantissa)) \\times 10^{$exponent}"
    end

    net_avg_sol_str = "\\textrm{$average_function_str}\\_\\textrm{solution}"
    net_avg_err_str = "\\textrm{$average_function_str}\\_\\textrm{error}"
    cur_avg_sol_str = "\\textrm{current}\\_$net_avg_sol_str"
    cur_avg_err_str = "\\textrm{current}\\_$net_avg_err_str"

    linestyles = (:solid, :dash, :dot, :dashdot, :dashdotdot)
    marker_kwargs = (; markershape = :circle, markeralpha = 0.5, markerstrokewidth = 0)
    plot_kwargs = (;
        legendposition = :outerright,
        legendtitlefontpointsize = 8,
        palette = :glasbey_bw_minc_20_maxl_70_n256,
        size = (1000, 2000), # size in px
        leftmargin = 60Plots.px,
        rightmargin = 0Plots.px,
        topmargin = 0Plots.px,
        bottommargin = 30Plots.px,
    )

    plot1_dts = t_end ./ round.(Int, num_steps .* num_steps_scaling_factor .^ (-1:0.5:1))
    plot1 = Plots.plot(;
        title = "Convergence Orders",
        xaxis = (latexstring("dt"), :log10),
        yaxis = (latexstring(net_avg_err_str), :log10),
        legendtitle = "Convergence Order ($order_confidence_percent% CI)",
        plot_kwargs...,
    )

    plot2b_min = typemax(FT)
    plot2b_max = typemin(FT)
    plot2a = Plots.plot(;
        title = latexstring("Solutions with \$dt = $(pow_str(default_dt))\$"),
        xaxis = (latexstring("t"),),
        yaxis = (latexstring(cur_avg_sol_str),),
        legendtitle = latexstring(net_avg_sol_str),
        plot_kwargs...,
    )
    plot2b = Plots.plot(;
        title = latexstring("Errors with \$dt = $(pow_str(default_dt))\$"),
        xaxis = (latexstring("t"),),
        yaxis = (latexstring(cur_avg_err_str), :log10),
        legendtitle = latexstring(net_avg_err_str),
        plot_kwargs...,
    )

    scb_cur_avg_err = make_saving_callback(cur_avg_err, prob.u0, t_end, nothing)
    scb_cur_avg_sol_and_err = make_saving_callback(cur_avg_sol_and_err, prob.u0, t_end, nothing)

    cur_avg_errs_dict = Dict()
    # for algorithm_name in algorithm_names
    algorithm_name = alg_name
    alg = algorithm(algorithm_name)
    alg_str = string(nameof(typeof(algorithm_name)))
    predicted_order = predicted_convergence_order(algorithm_name, prob.f)
    linestyle = linestyles[(predicted_order - 1) % length(linestyles) + 1]

    verbose && @info "Running $test_name with $alg_str..."
    @info "Using plot1_dts=$plot1_dts"
    plot1_net_avg_errs = map(plot1_dts) do plot1_dt
        cur_avg_errs =
            solve(deepcopy(prob), alg; dt = plot1_dt, save_everystep = !only_endpoints, callback = scb_cur_avg_err).u
        cur_avg_errs_dict[plot1_dt] = cur_avg_errs
        verbose && @info "RMS_error(dt = $plot1_dt) = $(average_function(cur_avg_errs))"
        return average_function(cur_avg_errs)
    end
    out_dict[key1][key2]["cur_avg_errs_dict"] = cur_avg_errs_dict
    order, order_uncertainty = convergence_order(plot1_dts, plot1_net_avg_errs, order_confidence_percent / 100)
    order_str = "$(float_str(order)) \\pm $(float_str(order_uncertainty))"
    if algorithm_name in super_convergence
        predicted_order += 1
        plot1_label = "$alg_str: \$$order_str\\ \\ \\ \\textbf{\\textit{SC}}\$"
    else
        plot1_label = "$alg_str: \$$order_str\$"
    end
    verbose && @info "Order = $order ± $order_uncertainty"
    if abs(order - predicted_order) > order_uncertainty
        @warn "Predicted order outside error bars for $alg_str ($test_name)"
    end
    if order_uncertainty > predicted_order / 10
        @warn "Order uncertainty too large for $alg_str ($test_name)"
    end

    # Remove all 0s from plot2_cur_avg_errs because they cannot be plotted on a
    # logarithmic scale. Record the extrema of plot2_cur_avg_errs to set ylim.
    plot2_values = solve(
        deepcopy(prob),
        alg;
        dt = default_dt,
        save_everystep = !only_endpoints,
        callback = scb_cur_avg_sol_and_err,
    )
    if any(isnan, plot2_values)
        @show test_name
        @show default_dt
        @show count(isnan, plot2_values.u[end])
        @show length(plot2_values.u[end])
        @show count(isnan, plot2_values.u[1])
        @show length(plot2_values.u[1])
        out_path = joinpath("output", "convergence_failure")
        mkpath(out_path)
        fname(i) = replace("$(key1)_$(key2)_step_$(i).png", " " => "_", "(" => "", ")" => "")
        for (i, u) in enumerate(plot2_values.u)
            if !all(isnan, u.u)
                f = joinpath(out_path, fname(i))
                Plots.png(Plots.plot(u.u), joinpath(out_path, fname(i)))
            end
        end
        # error("NaN found in plot2_values in problem $(test_name)")
    end
    out_dict[key1][key2]["plot2_values"] = plot2_values


    if !isnothing(full_history_algorithm_name)
        history_alg = algorithm(full_history_algorithm_name)
        history_alg_name = string(nameof(typeof(full_history_algorithm_name)))
        history_solve_results = solve(
            deepcopy(prob),
            history_alg;
            dt = default_dt,
            save_everystep = !only_endpoints,
            callback = make_saving_callback((u, t, integrator) -> u .- ref_sol(t), prob.u0, t_end, nothing),
        )
        out_dict[key1][key2]["history_solve_results"] = history_solve_results
    end
    return out_dict
end

function test_unconstrained_vs_ssp_without_limiters(alg_name, test_case, num_steps)
    prob = test_case.split_prob
    dt = test_case.t_end / num_steps
    newtons_method = NewtonsMethod(; max_iters = test_case.linear_implicit ? 1 : 2)
    algorithm = IMEXAlgorithm(alg_name, newtons_method)
    reference_algorithm = IMEXAlgorithm(alg_name, newtons_method, Unconstrained())
    solution = solve(deepcopy(prob), algorithm; dt).u[end]
    reference_solution = solve(deepcopy(prob), reference_algorithm; dt).u[end]
    if norm(solution .- reference_solution) / norm(reference_solution) > 30 * eps(Float64)
        alg_str = string(nameof(typeof(alg_name)))
        @warn "Unconstrained and SSP versions of $alg_str \
               give different results for $(test_case.test_name)"
    end
end
