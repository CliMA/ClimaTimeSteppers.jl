#=
julia --project=docs
using Revise; include(joinpath("docs", "src", "dev", "summarize_convergence.jl"))
=#

using ClimaTimeSteppers
import JLD2
using Test
using Printf: @sprintf
ENV["GKSwstype"] = "nul" # avoid displaying plots

using InteractiveUtils: subtypes

include(joinpath(@__DIR__, "compute_convergence.jl"))
include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl"))

all_subtypes(::Type{T}) where {T} = isabstracttype(T) ? vcat(all_subtypes.(subtypes(T))...) : [T]

function _rprint_dict_structure(io::IO, x::T, pc, xname) where {T <: NamedTuple}
    for pn in propertynames(x)
        pc_full = (pc..., ".", pn)
        xi = getproperty(x, pn)
        _rprint_dict_structure(io, xi; pc = pc_full, xname)
    end
end;

function _rprint_dict_structure(io::IO, x::Dict; pc, xname)
    for k in keys(x)
        pc_full = (pc..., "[$k]")
        xi = getindex(x, k)
        _rprint_dict_structure(io, xi; pc = pc_full, xname)
    end
end;

function _rprint_dict_structure(io::IO, xi; pc, xname)
    println(io, "$(xname * string(join(pc)))")
end

_rprint_dict_structure(io::IO, x::T, xname) where {T <: Union{NamedTuple, Dict}} =
    _rprint_dict_structure(io, x; pc = (), xname)
_rprint_dict_structure(x::T, xname) where {T <: Union{NamedTuple, Dict}} = _rprint_dict_structure(stdout, x, xname)

"""
    @rprint_dict_structure(::T) where {T <: Union{NamedTuple, Dict}}

Recursively print keys of a (potentially nested) dict.
"""
macro rprint_dict_structure(x)
    return :(_rprint_dict_structure(stdout, $(esc(x)), $(string(x))))
end

function algorithm_names_by_availability(out_dict, test_name, algorithm_names_all, plot1_dts)
    # @rprint_dict_structure out_dict # for debugging
    algorithm_names = []
    for algorithm_name in algorithm_names_all
        key2 = string(algorithm_name)
        keep_alg = true
        if !haskey(out_dict, key2)
            keep_alg = false
            @warn "out_dict has no key $key2"
        else
            if !haskey(out_dict[key2], "cur_avg_errs_dict")
                keep_alg = false
                @warn "out_dict[$key2] has no key cur_avg_errs_dict"
            else
                for plot1_dt in plot1_dts
                    if !haskey(out_dict[key2]["cur_avg_errs_dict"], plot1_dt)
                        keep_alg = false
                        @warn "out_dict[$key2][cur_avg_errs_dict] has no key plot1_dt"
                    end
                end
            end
        end
        keep_alg && push!(algorithm_names, algorithm_name)
    end
    return algorithm_names
end

function summarize_convergence(
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

    title = "All Algorithms"
    algorithm_names_all = get_algorithm_names()

    (; test_name, t_end, linear_implicit, analytic_sol) = test_case

    keep_alg = true
    plot1_dts = t_end ./ round.(Int, num_steps .* num_steps_scaling_factor .^ (-1:0.5:1))
    algorithm_names = algorithm_names_by_availability(out_dict, test_name, algorithm_names_all, plot1_dts)
    @show algorithm_names

    # out_dict = Dict()
    # out_dict[key2] = Dict()

    prob = test_case.split_prob
    FT = typeof(t_end)
    default_dt = t_end / num_steps

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
        out_dict["numerical_ref_sol"] # solve(deepcopy(prob), ref_alg; dt = ref_dt, save_everystep = !only_endpoints)
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

    for algorithm_name in algorithm_names
        cur_avg_errs_dict = out_dict[string(algorithm_name)]["cur_avg_errs_dict"]
        alg = algorithm(algorithm_name)
        alg_str = string(nameof(typeof(algorithm_name)))
        predicted_order = predicted_convergence_order(algorithm_name, prob.f)
        linestyle = linestyles[(predicted_order - 1) % length(linestyles) + 1]

        verbose && @info "Running $test_name with $alg_str..."
        plot1_net_avg_errs = map(plot1_dts) do plot1_dt
            cur_avg_errs = cur_avg_errs_dict[plot1_dt]
            # solve(
            #     deepcopy(prob),
            #     alg;
            #     dt = plot1_dt,
            #     save_everystep = !only_endpoints,
            #     callback = scb_cur_avg_err,
            # ).u
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
        plot2_values = out_dict[string(algorithm_name)]["plot2_values"]
        # plot2_values = solve(
        #     deepcopy(prob),
        #     alg;
        #     dt = default_dt,
        #     save_everystep = !only_endpoints,
        #     callback = scb_cur_avg_sol_and_err,
        # )
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
        history_solve_results = out_dict[history_alg_name]["history_solve_results"]
        # history_solve_results = solve(
        #     deepcopy(prob),
        #     history_alg;
        #     dt = default_dt,
        #     save_everystep = !only_endpoints,
        #     callback = make_saving_callback((u, t, integrator) -> u .- ref_sol(t), prob.u0, t_end, nothing),
        # )
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
    heights₀ = repeat([1 / (n_plots - 1)], n_plots - 1)
    footnote_height = eps()
    heights = [heights₀ .- footnote_height / n_plots..., footnote_height]
    layout = Plots.grid(n_plots, 1; heights)
    plot = Plots.plot(
        plots...;
        plot_title = "Analysis of $title for \"$test_name\"",
        layout = layout,
        fontfamily = "Computer Modern",
    )

    mkpath("output")
    file_suffix = lowercase(replace(test_name * ' ' * title, " " => "_"))
    Plots.savefig(plot, joinpath("output", "convergence_$file_suffix.png"))
end


function merge_tests(v::Vector)
    test_names = collect(Set(map(x -> first(keys(x)), v)))
    return Dict(map(test_names) do tn
        matched_dicts = filter(x -> first(keys(x)) == tn, v)
        inner_dicts = map(x -> first(values(x)), matched_dicts)
        inner_merged = merge(inner_dicts...)
        Pair(tn, inner_merged)
    end...)
end

let # Convergence
    # NOTE: Some imperfections in the convergence order for SSPKnoth are to be
    # expected because we are not using the exact Jacobian
    vector_of_dicts = []
    files = readdir()
    filter!(x -> endswith(x, ".jld2"), files)
    filter!(x -> startswith(basename(x), "convergence_"), files)
    for f in files
        push!(vector_of_dicts, JLD2.load_object(f))
    end

    out_dict = merge_tests(vector_of_dicts)
    for test_name in keys(out_dict)
        out_dict_test = out_dict[test_name]
        args = out_dict_test["args"]
        kwargs = out_dict_test["kwargs"]
        summarize_convergence(out_dict_test, args...; kwargs...)
    end
end
