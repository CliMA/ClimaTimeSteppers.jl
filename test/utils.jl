import Plots, Printf, Distributions
using LaTeXStrings
import ClimaTimeSteppers as CTS
using Test

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
    confidence_interval_t_value = -Distributions.quantile(Distributions.TDist(n_dof), (1 - confidence) / 2)

    # standard deviation of linear regression
    regression_standard_deviation = sqrt(sum((log_errs .- (log_dts .* order .+ log_err_of_dt1)) .^ 2) / n_dof)

    # standard deviation of slope
    order_standard_deviation = regression_standard_deviation / sqrt(sum((log_dts .- sum(log_dts) / n_dts) .^ 2))

    # "uncertainty" in slope (half of width of confidence interval)
    order_uncertainty = confidence_interval_t_value * order_standard_deviation

    return order, order_uncertainty
end

average_function_name() = "RMS"
average_function(array) = norm(array) / sqrt(length(array))

"""
    test_algs(
        tableaus_name,
        algs,
        test_case,
        num_steps;
        save_every_n_steps = max(1, Int(fld(num_steps, 500))),
        no_increment_algs = (),
    )

Check that all of the specified ODE algorithms have the predicted convergence
order, and that the increment formulations give the same results as the tendency
formulations. Generate plots that show the algorithms' solutions for the
specified test case, the errors of these solutions, and the convergence of these
errors with respect to `dt`.

# Arguments
- `algs_name::String`: the name of the collection of algorithms
- `tableaus`: an array of tableaus
- `test_case::IntegratorTestCase`: the test case to use
- `num_steps::Int`: the numerical solutions for the solution and error plots
      are computed with `dt = t_end / num_steps`, while the solutions for
      the convergence plots are computed with `dt`, `dt / sqrt(10)`, and
      `dt * sqrt(10)`
- `save_every_n_steps::Int`: the solution and error plots show only show the
      values at every `n`-th step; the default value is such that 500-999 steps
      are plotted (unless there are fewer than 500 steps, in which case every
      step is plotted)
"""
function test_tableaus(
    tableaus_name,
    tableaus,
    test_case,
    num_steps;
    num_steps_scaling_factor = 10,
    order_confidence_percent = 99,
    super_convergence_tableaus = (),
    numerical_reference_tableau = nothing,
    numerical_reference_num_steps = num_steps_scaling_factor^3 * num_steps,
    full_history_tableau = nothing,
)
    (; test_name, linear_implicit, t_end, analytic_sol) = test_case
    prob = test_case.split_prob
    FT = typeof(t_end)
    default_dt = t_end / num_steps

    max_iters = linear_implicit ? 1 : 2 # TODO: is 2 enough?
    algorithm(tableau) = CTS.IMEXARKAlgorithm(tableau(), NewtonsMethod(; max_iters))

    ref_sol = if isnothing(numerical_reference_tableau)
        analytic_sol
    else
        ref_alg = algorithm(numerical_reference_tableau)
        ref_dt = t_end / numerical_reference_num_steps
        solve(deepcopy(prob), ref_alg; dt = ref_dt, save_everystep = true)
    end

    cur_avg_err(u, t) = average_function(abs.(u .- ref_sol(t)))
    cur_avg_sol_and_err(u, t) = (average_function(u), average_function(abs.(u .- ref_sol(t))))

    float_str(x) = Printf.@sprintf "%.4f" x
    pow_str(x) = "10^{$(Printf.@sprintf "%.1f" log10(x))}"
    function si_str(x)
        exponent = floor(Int, log10(x))
        mantissa = x / 10.0^exponent
        return "$(float_str(mantissa)) \\times 10^{$exponent}"
    end

    net_avg_sol_str = "\\textrm{$(average_function_name())}\\_\\textrm{solution}"
    net_avg_err_str = "\\textrm{$(average_function_name())}\\_\\textrm{error}"
    cur_avg_sol_str = "\\textrm{current}\\_$net_avg_sol_str"
    cur_avg_err_str = "\\textrm{current}\\_$net_avg_err_str"

    linestyles = (:solid, :dash, :dot, :dashdot, :dashdotdot)
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

    for tableau in tableaus
        alg = algorithm(tableau)
        alg_name = string(nameof(tableau))
        predicted_order = CTS.theoretical_convergence_order(tableau(), prob.f)
        linestyle = linestyles[(predicted_order - 1) % length(linestyles) + 1]

        plot1_net_avg_errs = map(plot1_dts) do plot1_dt
            cur_avg_errs =
                solve(
                    deepcopy(prob),
                    alg;
                    dt = plot1_dt,
                    save_everystep = true,
                    save_func = cur_avg_err,
                    kwargshandle = DiffEqBase.KeywordArgSilent,
                ).u
            return average_function(cur_avg_errs)
        end
        order, order_uncertainty = convergence_order(plot1_dts, plot1_net_avg_errs, order_confidence_percent / 100)
        order_str = "$(float_str(order)) \\pm $(float_str(order_uncertainty))"
        if tableau in super_convergence_tableaus
            predicted_order += 1
            plot1_label = "$alg_name: \$$order_str\\ \\ \\ \\textbf{\\textit{SC}}\$"
        else
            plot1_label = "$alg_name: \$$order_str\$"
        end
        @test abs(order - predicted_order) <= order_uncertainty
        @test order_uncertainty <= predicted_order / 10
        Plots.plot!(
            plot1,
            plot1_dts,
            plot1_net_avg_errs;
            label = latexstring(plot1_label),
            markershape = :circle,
            markeralpha = 0.5,
            markerstrokewidth = 0,
            linestyle,
        )

        # Remove all 0s from plot2_cur_avg_errs because they cannot be plotted on a
        # logarithmic scale. Record the extrema of plot2_cur_avg_errs to set ylim.
        plot2_values = solve(
            deepcopy(prob),
            alg;
            dt = default_dt,
            save_everystep = true,
            save_func = cur_avg_sol_and_err,
            kwargshandle = DiffEqBase.KeywordArgSilent,
        )
        plot2_ts = plot2_values.t
        plot2_cur_avg_sols = first.(plot2_values.u)
        plot2_cur_avg_errs = last.(plot2_values.u)
        plot2b_min = min(plot2b_min, minimum(x -> x == 0 ? typemax(FT) : x, plot2_cur_avg_errs))
        plot2b_max = max(plot2b_max, maximum(plot2_cur_avg_errs))
        plot2_cur_avg_errs .= max.(plot2_cur_avg_errs, eps(FT(0)))
        plot2_net_avg_sol = average_function(plot2_cur_avg_sols)
        plot2_net_avg_err = average_function(plot2_cur_avg_errs)
        Plots.plot!(
            plot2a,
            plot2_ts,
            plot2_cur_avg_sols;
            label = latexstring("$alg_name: \$$(si_str(plot2_net_avg_sol))\$"),
            linestyle,
        )
        Plots.plot!(
            plot2b,
            plot2_ts,
            plot2_cur_avg_errs;
            label = latexstring("$alg_name: \$$(si_str(plot2_net_avg_err))\$"),
            linestyle,
        )
    end

    Plots.plot!(plot2b; ylim = (plot2b_min / 2, plot2b_max * 2))

    plots = (plot1, plot2a, plot2b)

    if !isnothing(full_history_tableau)
        history_alg = algorithm(full_history_tableau)
        history_alg_name = string(nameof(full_history_tableau))
        history_solve_results = solve(
            deepcopy(prob),
            history_alg;
            dt = default_dt,
            save_everystep = true,
            save_func = (u, t) -> u .- ref_sol(t),
            kwargshandle = DiffEqBase.KeywordArgSilent,
        )
        history_array = hcat(history_solve_results.u...)
        history_plot_title = "Errors for $history_alg_name with \$dt = $(pow_str(default_dt))\$"
        history_plot = Plots.heatmap(
            history_solve_results.t,
            1:size(history_array, 1),
            history_array;
            title = latexstring(history_plot_title),
            xaxis = (latexstring("t"),),
            yaxis = ("index",),
            plot_kwargs...,
        )
        plots = (plots..., history_plot)
    end

    avg_def_str(val_str, var_str) = "\\textrm{$(average_function_name())}(\\{$val_str\\,\\forall\\,$var_str\\})"
    ref_sol_def_str =
        isnothing(numerical_reference_tableau) ? "Y_{analytic}(t)[\\textrm{index}]" : "Y_{ref}(t)[\\textrm{index}]"
    cur_avg_sol_def_str = avg_def_str("Y(t)[\\textrm{index}]", "\\textrm{index}")
    cur_avg_err_def_str = avg_def_str("|Y(t)[\\textrm{index}] - $ref_sol_def_str|", "\\textrm{index}")
    net_avg_sol_def_str = avg_def_str("$cur_avg_sol_str(t)", "t")
    net_avg_err_def_str = avg_def_str("$cur_avg_err_str(t)", "t")
    footnote = """
        Terminology:
            \$$cur_avg_sol_str(t) = $cur_avg_sol_def_str\$
            \$$cur_avg_err_str(t) = $cur_avg_err_def_str\$
            \$$net_avg_sol_str = $net_avg_sol_def_str\$
            \$$net_avg_err_str = $net_avg_err_def_str\$"""
    if !isnothing(numerical_reference_tableau)
        ref_alg_name = string(nameof(numerical_reference_tableau))
        ref_cur_avg_errs = map(ref_sol.u, ref_sol.t) do u, t
            average_function(abs.(u .- analytic_sol(t)))
        end
        ref_net_avg_err = average_function(ref_cur_avg_errs)
        ref_cur_avg_err_def_str =
            avg_def_str("|Y_{ref}(t)[\\textrm{index}] - Y_{analytic}(t)[\\textrm{index}]|", "\\textrm{index}")
        ref_net_avg_err_def_str = avg_def_str(ref_cur_avg_err_def_str, "t")
        footnote = "$footnote\n\n\nNote: The \"reference solution\" \$Y_{ref}\$ was \
                    computed using $ref_alg_name with\n\$dt = $(pow_str(ref_dt)),\\ \
                    \\textrm{and}\\ $ref_net_avg_err_def_str = \
                    $(si_str(ref_net_avg_err))\$"
    end
    footnote_plot = Plots.plot(;
        title = latexstring(footnote),
        titlelocation = :left,
        titlefontsize = 12,
        axis = nothing,
        framestyle = :none,
        margin = 0Plots.px,
        bottommargin = 10Plots.px,
    )
    plots = (plots..., footnote_plot)

    n_plots = length(plots)
    plot = Plots.plot(
        plots...;
        plot_title = "Analysis of $tableaus_name Methods for \"$test_name\"",
        layout = Plots.grid(n_plots, 1; heights = [repeat([1 / (n_plots - 1)], n_plots - 1)..., 0]),
        fontfamily = "Computer Modern",
    )

    mkpath("output")
    file_suffix = lowercase(replace(test_name * ' ' * tableaus_name, " " => "_"))
    Plots.savefig(plot, joinpath("output", "convergence_$file_suffix.png"))
end
