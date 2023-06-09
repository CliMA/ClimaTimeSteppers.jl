import Plots, Markdown
using ClimaCorePlots
using Distributions: quantile, TDist
using Printf: @sprintf
using LaTeXStrings: latexstring
using PrettyTables: pretty_table, ft_printf

"""
    predicted_convergence_order(algorithm_name, ode_function)

Return the predicted convergence order of the algorithm for the given ODE
function (assuming that the algorithm converges).
"""
function predicted_convergence_order(algorithm_name::AbstractAlgorithmName, ode_function::ClimaODEFunction)
    (imp_order, exp_order, combined_order) = imex_convergence_orders(algorithm_name)
    has_imp = !isnothing(ode_function.T_imp!)
    has_exp = !isnothing(ode_function.T_exp!) || !isnothing(ode_function.T_lim!)
    has_imp && !has_exp && return imp_order
    !has_imp && has_exp && return exp_order
    has_imp && has_exp && return combined_order
    return 0
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

function verify_convergence(
    title,
    algorithm_names,
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

    algorithm(algorithm_name::ClimaTimeSteppers.ERKAlgorithmName) = ExplicitAlgorithm(algorithm_name)
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
        solve(deepcopy(prob), ref_alg; dt = ref_dt, save_everystep = !only_endpoints)
    end

    cur_avg_err(u, t) = average_function(abs.(u .- ref_sol(t)))
    cur_avg_sol_and_err(u, t) = (average_function(u), average_function(abs.(u .- ref_sol(t))))

    float_str(x) = @sprintf "%.4f" x
    pow_str(x) = "10^{$(@sprintf "%.1f" log10(x))}"
    function si_str(x)
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

    for algorithm_name in algorithm_names
        alg = algorithm(algorithm_name)
        alg_str = string(nameof(typeof(algorithm_name)))
        predicted_order = predicted_convergence_order(algorithm_name, prob.f)
        linestyle = linestyles[(predicted_order - 1) % length(linestyles) + 1]

        verbose && @info "Running $test_name with $alg_str..."
        plot1_net_avg_errs = map(plot1_dts) do plot1_dt
            cur_avg_errs =
                solve(
                    deepcopy(prob),
                    alg;
                    dt = plot1_dt,
                    save_everystep = !only_endpoints,
                    save_func = cur_avg_err,
                    kwargshandle = DiffEqBase.KeywordArgSilent,
                ).u
            verbose && @info "RMS_error(dt = $plot1_dt) = $(average_function(cur_avg_errs))"
            return average_function(cur_avg_errs)
        end
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
        Plots.plot!(plot1, plot1_dts, plot1_net_avg_errs; label = latexstring(plot1_label), linestyle, marker_kwargs...)

        # Remove all 0s from plot2_cur_avg_errs because they cannot be plotted on a
        # logarithmic scale. Record the extrema of plot2_cur_avg_errs to set ylim.
        plot2_values = solve(
            deepcopy(prob),
            alg;
            dt = default_dt,
            save_everystep = !only_endpoints,
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
            label = latexstring("$alg_str: \$$(si_str(plot2_net_avg_sol))\$"),
            linestyle,
            (only_endpoints ? marker_kwargs : (;))...,
        )
        Plots.plot!(
            plot2b,
            plot2_ts,
            plot2_cur_avg_errs;
            label = latexstring("$alg_str: \$$(si_str(plot2_net_avg_err))\$"),
            linestyle,
            (only_endpoints ? marker_kwargs : (;))...,
        )
    end

    Plots.plot!(plot2b; ylim = (plot2b_min / 2, plot2b_max * 2))

    plots = (plot1, plot2a, plot2b)

    if !isnothing(full_history_algorithm_name)
        history_alg = algorithm(full_history_algorithm_name)
        history_alg_name = string(nameof(typeof(full_history_algorithm_name)))
        history_solve_results = solve(
            deepcopy(prob),
            history_alg;
            dt = default_dt,
            save_everystep = !only_endpoints,
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

    avg_def_str(val_str, var_str) = "\\textrm{$average_function_str}(\\{$val_str\\,\\forall\\,$var_str\\})"
    ref_sol_def_str =
        isnothing(numerical_reference_algorithm_name) ? "Y_{analytic}(t)[\\textrm{index}]" :
        "Y_{ref}(t)[\\textrm{index}]"
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
    if !isnothing(numerical_reference_algorithm_name)
        ref_cur_avg_errs = map(ref_sol.u, ref_sol.t) do u, t
            average_function(abs.(u .- analytic_sol(t)))
        end
        ref_net_avg_err = average_function(ref_cur_avg_errs)
        ref_cur_avg_err_def_str =
            avg_def_str("|Y_{ref}(t)[\\textrm{index}] - Y_{analytic}(t)[\\textrm{index}]|", "\\textrm{index}")
        ref_net_avg_err_def_str = avg_def_str(ref_cur_avg_err_def_str, "t")
        footnote = "$footnote\n\n\nNote: The \"reference solution\" \$Y_{ref}\$ was \
                    computed using $ref_alg_str with\n\$dt = $(pow_str(ref_dt)),\\ \
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
        plot_title = "Analysis of $title for \"$test_name\"",
        layout = Plots.grid(n_plots, 1; heights = [repeat([1 / (n_plots - 1)], n_plots - 1)..., 0]),
        fontfamily = "Computer Modern",
    )

    mkpath("output")
    file_suffix = lowercase(replace(test_name * ' ' * title, " " => "_"))
    Plots.savefig(plot, joinpath("output", "convergence_$file_suffix.png"))
end

# Generates the Table 1 from
# "Optimization-based limiters for the spectral element method" by Guba et al.,
# and also plots the values used to generate the table.
function limiter_summary(::Type{FT}, algorithm_names, test_case_type, num_steps) where {FT}
    to_title(name) = titlecase(replace(string(name), '_' => ' '))
    table_rows = []
    for algorithm_name in algorithm_names
        alg_str = string(nameof(typeof(algorithm_name)))
        plots = []
        plot_kwargs = (;
            clims = (0, 1),
            color = :diverging_rainbow_bgymr_45_85_c67_n256,
            colorbar = false,
            guide = "",
            margin = 10Plots.px,
        )
        for use_limiter in (false, true), use_hyperdiffusion in (false, true)
            test_case = test_case_type(FT; use_limiter, use_hyperdiffusion)
            prob = test_case.split_prob
            dt = test_case.t_end / num_steps
            algorithm =
                algorithm_name isa ClimaTimeSteppers.IMEXARKAlgorithmName ?
                IMEXAlgorithm(algorithm_name, NewtonsMethod()) : ExplicitAlgorithm(algorithm_name)
            solution = solve(deepcopy(prob), algorithm; dt).u
            initial_q = solution[1].ρq ./ solution[1].ρ
            final_q = solution[end].ρq ./ solution[end].ρ
            names = propertynames(initial_q)

            if isempty(plots)
                for name in names
                    push!(plots, Plots.plot(initial_q.:($name); plot_kwargs..., title = to_title(name)))
                end
            end
            for name in names
                push!(plots, Plots.plot(final_q.:($name); plot_kwargs..., title = ""))
            end

            for name in names
                ϕ₀ = initial_q.:($name)
                ϕ = final_q.:($name)
                Δϕ₀ = maximum(ϕ₀) - minimum(ϕ₀)
                ϕ_error = ϕ .- ϕ₀
                table_row = [
                    alg_str;;
                    string(use_limiter);;
                    string(use_hyperdiffusion);;
                    to_title(name);;
                    max(0, -(minimum(ϕ) - minimum(ϕ₀)) / Δϕ₀);;
                    max(0, (maximum(ϕ) - maximum(ϕ₀)) / Δϕ₀);;
                    map(p -> norm(ϕ_error, p) / norm(ϕ₀, p), (1, 2, Inf))...
                ]
                push!(table_rows, table_row)
            end
        end
        colorbar_plot = Plots.scatter(
            [0];
            plot_kwargs...,
            colorbar = true,
            framestyle = :none,
            legend_position = :none,
            margin = 0Plots.px,
            markeralpha = 0,
            zcolor = [0],
        )
        plot = Plots.plot(
            plots...,
            colorbar_plot;
            layout = (Plots.@layout [Plots.grid(5, 3) a{0.1w}]),
            plot_title = "Tracer specific humidity for $alg_str (Initial, \
                          Final, Final w/ Hyperdiffusion, Final w/ Limiter, \
                          Final w/ Hyperdiffusion & Limiter)",
            size = (1600, 2000),
        )
        Plots.savefig(plot, joinpath("output", "limiter_summary_$alg_str.png"))
    end
    table = pretty_table(
        vcat(table_rows...);
        header = [
            "Algorithm",
            "Limiter",
            "Hyperdiffusion",
            "Tracer Name",
            "Max Undershoot",
            "Max Overshoot",
            "1-Norm Error",
            "2-Norm Error",
            "∞-Norm Error",
        ],
        body_hlines = collect(3:3:(length(table_rows) - 1)),
        formatters = ft_printf("%.4e"),
    )
    println(table)
end
