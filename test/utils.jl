import Plots, Printf
import ClimaTimeSteppers as CTS
using Test

has_increment_formulation(::CTS.AbstractIMEXARKTableau) = true
has_increment_formulation(::CTS.NewAbstractIMEXARKTableau) = false

"""
    test_algs(
        algs_name,
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
function test_algs(
    algs_name,
    tableaus,
    test_case,
    num_steps;
    save_every_n_steps = Int(cld(num_steps, 500)),
    super_convergence = nothing
)
    (; test_name, linear_implicit, t_end, analytic_sol) = test_case
    FT = typeof(t_end)
    linestyles = (:solid, :dash, :dot, :dashdot, :dashdotdot)
    plot_kwargs = (;
        size = (1000, 600),
        margin = 4Plots.mm,
        titlelocation = :left,
        legend_position = :outerright,
        palette = :glasbey_bw_minc_20_maxl_70_n256,
    )

    plot1_dt = t_end / num_steps
    plot1_saveat = 0:(plot1_dt * save_every_n_steps):t_end
    plot1a = Plots.plot(;
        title = "Solution Norms of $algs_name Methods for `$test_name` \
                 (with dt = 10^$(Printf.@sprintf "%.1f" log10(plot1_dt)))",
        xlabel = "t",
        ylabel = "Solution Norm: ||Y_computed||",
        plot_kwargs...,
    )
    plot1b = Plots.plot(;
        title = "Solution Errors of $algs_name Methods for `$test_name` \
                 (with dt = 10^$(Printf.@sprintf "%.1f" log10(plot1_dt)))",
        xlabel = "t",
        ylabel = "Error Norm: ||Y_computed - Y_analytic||",
        yscale = :log10,
        plot_kwargs...,
    )
    plot1b_ymin = typemax(FT) # dynamically set ylim because some errors are 0
    plot1b_ymax = typemin(FT)

    t_end_string = t_end % 1 == 0 ? string(Int(t_end)) : Printf.@sprintf("%.2f", t_end)
    plot2_dts = [plot1_dt / sqrt(10), plot1_dt, plot1_dt * sqrt(10)]
    plot2 = Plots.plot(;
        title = "Convergence Orders of $algs_name Methods for `$test_name` \
                 (at t = $t_end_string)",
        xlabel = "dt",
        ylabel = "Error Norm: ||Y_computed - Y_analytic||",
        xscale = :log10,
        yscale = :log10,
        plot_kwargs...,
    )

    analytic_sols = map(analytic_sol, plot1_saveat)
    analytic_end_sol = [analytic_sols[end]]

    for tab in tableaus
        if tab() isa CTS.AbstractIMEXARKTableau
            max_iters = linear_implicit ? 1 : 2 # TODO: is 2 enough?
            alg = CTS.IMEXARKAlgorithm(tab(), NewtonsMethod(; max_iters))
            tendency_prob = test_case.split_prob
            increment_prob = test_case.split_increment_prob
        elseif tab() isa CTS.NewAbstractIMEXARKTableau
            max_iters = linear_implicit ? 1 : 2 # TODO: is 2 enough?
            alg = CTS.NewIMEXARKAlgorithm(tab(), NewtonsMethod(; max_iters))
            tendency_prob = test_case.split_prob
            increment_prob = test_case.split_increment_prob
        else
            alg = tab()
            tendency_prob = test_case.prob
            increment_prob = test_case.increment_prob
        end
        predicted_order = if super_convergence==tab
            CTS.theoretical_convergence_order(tab())+1
        else
            CTS.theoretical_convergence_order(tab())
        end
        linestyle = linestyles[(predicted_order - 1) % length(linestyles) + 1]
        alg_name = string(nameof(tab))

        # Use tstops to fix saving issues due to machine precision (e.g. if the
        # integrator needs to save at t but it stops at t - eps(), it will skip
        # over saving at t, unless tstops forces it to round t - eps() to t).
        solve_args =
            (; dt = plot1_dt, saveat = plot1_saveat, tstops = plot1_saveat)
        tendency_sols = solve(deepcopy(tendency_prob), alg; solve_args...).u
        tendency_norms = @. norm(tendency_sols)
        tendency_errs = @. norm(tendency_sols - analytic_sols)
        min_err = minimum(x -> x == 0 ? typemax(FT) : x, tendency_errs)
        plot1b_ymin = min(plot1b_ymin, min_err)
        plot1b_ymax = max(plot1b_ymax, maximum(tendency_errs))
        tendency_errs .=
            max.(tendency_errs, eps(FT(0))) # plotting 0 breaks the log scale
        Plots.plot!(plot1a, plot1_saveat, tendency_norms; label = alg_name, linestyle)
        Plots.plot!(plot1b, plot1_saveat, tendency_errs; label = alg_name, linestyle)

        if has_increment_formulation(tab())
            increment_sols =
                solve(deepcopy(increment_prob), alg; solve_args...).u
            increment_errs = @. norm(increment_sols - tendency_sols)
            @test maximum(increment_errs) < 1000 * eps(FT) broken =
                alg_name == "HOMMEM1" # TODO: why is this one broken?
        end

        tendency_end_sols =
            map(dt -> solve(deepcopy(tendency_prob), alg; dt).u[end], plot2_dts)
        tendency_end_errs = @. norm(tendency_end_sols - analytic_end_sol)
        _, computed_order = hcat(ones(length(plot2_dts)), log10.(plot2_dts)) \
            log10.(tendency_end_errs)
        @test computed_order ≈ predicted_order rtol = 0.1
        label = "$alg_name ($(Printf.@sprintf "%.3f" computed_order))"
        Plots.plot!(plot2, plot2_dts, tendency_end_errs; label, linestyle)
    end
    Plots.plot!(plot1b; ylim = (plot1b_ymin / 2, plot1b_ymax * 2))

    mkpath("output")
    file_suffix = "$(test_name)_$(lowercase(replace(algs_name, " " => "_")))"
    Plots.savefig(plot1a, joinpath("output", "solutions_$(file_suffix).png"))
    Plots.savefig(plot1b, joinpath("output", "errors_$(file_suffix).png"))
    Plots.savefig(plot2, joinpath("output", "orders_$(file_suffix).png"))
end
