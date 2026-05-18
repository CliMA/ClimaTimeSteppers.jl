import Plots, Markdown
using ClimaCorePlots
using Distributions: quantile, TDist
using Printf: @sprintf
using LaTeXStrings: latexstring
using PrettyTables: pretty_table, ft_printf
import ClimaTimeSteppers as CTS
import DiffEqCallbacks

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

function make_saving_callback(cb, u, t, integrator)
    savevalType = typeof(cb(u, t, integrator))
    return DiffEqCallbacks.SavingCallback(cb, DiffEqCallbacks.SavedValues(typeof(t), savevalType))
end

# Generates the Table 1 from
# "Optimization-based limiters for the spectral element method" by Guba et al.,
# and also plots the values used to generate the table.
function limiter_summary(sol_dicts, algorithm_names)
    to_title(name) = titlecase(replace(string(name), '_' => ' '))
    table_rows = []
    mkpath("output")
    for alg_name in algorithm_names
        alg_str = string(nameof(typeof(alg_name)))
        plots = []
        plot_kwargs = (;
            clims = (0, 1),
            color = :diverging_rainbow_bgymr_45_85_c67_n256,
            colorbar = false,
            guide = "",
            margin = 10Plots.px,
        )
        for use_limiter in (false, true), use_hyperdiffusion in (false, true)
            solution = sol_dicts[alg_name, use_hyperdiffusion, use_limiter]
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
