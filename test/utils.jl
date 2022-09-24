"""
    convergence_rates(problem, solution, method, dts; kwargs...)

Compute the errors rates of `method` on `problem` by comparing to `solution`
on the set of `dt` values in `dts`. Extra `kwargs` are passed to `solve`

`solution` should be a function with a method `solution(u0, p, t)`.
"""
function convergence_errors(prob, sol, method, dts; kwargs...)
  errs = map(dts) do dt
       # copy the problem so we don't mutate u0
      prob_copy = deepcopy(prob)
      u = solve(prob_copy, method; dt=dt, kwargs...)
      norm(u .- sol(prob.u0, prob.p, prob.tspan[end]))
  end
  return errs
end


"""
  convergence_order(problem, solution, method, dts; kwargs...)

Estimate the order of the rate of convergence of `method` on `problem` by comparing to
`solution` the set of `dt` values in `dts`.

`solution` should be a function with a method `solution(u0, p, t)`.
"""
function convergence_order(prob, sol, method, dts; kwargs...)
  errs = convergence_errors(prob, sol, method, dts; kwargs...)
  # find slope coefficient in log scale
  _,order_est = hcat(ones(length(dts)), log2.(dts)) \ log2.(errs)
  return order_est
end



"""
    DirectSolver

A linear solver which forms the full matrix of a linear operator and its LU factorization.
"""
struct DirectSolver end

DirectSolver(args...) = DirectSolver()

function (::DirectSolver)(x,A,b,matrix_updated; kwargs...)
  n = length(x)
  M = mapslices(y -> mul!(similar(y), A, y), Matrix{eltype(x)}(I,n,n), dims=1)
  x .= M \ b
end

function test_algs(
    algs_name,
    algs_to_order,
    test_case,
    num_dt_splits;
    num_saveat_splits = min(num_dt_splits, 8),
    no_increment_algs = (),
)
    (; test_name, linear_implicit, t_end, probs, split_probs, analytic_sol) =
        test_case
    FT = typeof(t_end)
    linestyles = (:solid, :dash, :dot, :dashdot, :dashdotdot)
    plot_kwargs = (;
        size = (1000, 600),
        margin = 3Plots.mm,
        titlelocation = :left,
        legend_position = :outerright,
        palette = :glasbey_bw_minc_20_maxl_70_n256,
    )
    
    plot1_dt = t_end / 2^num_dt_splits
    plot1_saveat = [FT(0), t_end / 2^num_saveat_splits]
    # Ensure that the saveat times are an EXACT subset of the integrator times.
    while plot1_saveat[end] < t_end
        push!(plot1_saveat, min(plot1_saveat[end] + plot1_saveat[2], t_end))
    end
    plot1a = plot(;
        title = "Solution Norms of $algs_name Methods for `$test_name` \
                 (with dt = 10^$(@sprintf "%.1f" log10(plot1_dt)))",
        xlabel = "t",
        ylabel = "Solution Norm: ||Y_computed||",
        plot_kwargs...,
    )
    plot1b = plot(;
        title = "Solution Errors of $algs_name Methods for `$test_name` \
                 (with dt = 10^$(@sprintf "%.1f" log10(plot1_dt)))",
        xlabel = "t",
        ylabel = "Error Norm: ||Y_computed - Y_analytic||",
        yscale = :log10,
        plot_kwargs...,
    )
    plot1b_ymin = typemax(FT) # dynamically set ylim because some errors are 0
    plot1b_ymax = typemin(FT)

    t_end_string = t_end % 1 == 0 ? string(Int(t_end)) : @sprintf("%.2f", t_end)
    plot2_dts = [plot1_dt / 4, plot1_dt, plot1_dt * 4]
    plot2 = plot(;
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
    sorted_algs_to_order = sort(collect(algs_to_order); by = x -> string(x[1]))
    for (alg_name, predicted_order) in sorted_algs_to_order
        if alg_name <: IMEXARKAlgorithm
            max_iters = linear_implicit ? 1 : 2
            alg =
                alg_name(NewtonsMethod(; linsolve = linsolve_direct, max_iters))
            (tendency_prob, increment_prob) = split_probs
        elseif alg_name <: RosenbrockAlgorithm
            alg = alg_name(;
                linsolve = linsolve_direct,
                multiply! = multiply_direct!,
                set_Δtγ! = set_Δtγ_direct!,
            )
            (tendency_prob, increment_prob) = probs
        end
        linestyle = linestyles[(predicted_order - 1) % length(linestyles) + 1]

        solve_args = (; dt = plot1_dt, saveat = plot1_saveat)
        tendency_sols =
            solve(deepcopy(tendency_prob), alg; solve_args...).u
        tendency_norms = @. norm(tendency_sols)
        tendency_errors = @. norm(tendency_sols - analytic_sols)
        min_error = minimum(x -> x == 0 ? typemax(FT) : x, tendency_errors)
        plot1b_ymin = min(plot1b_ymin, min_error)
        plot1b_ymax = max(plot1b_ymax, maximum(tendency_errors))
        tendency_errors .=
            max.(tendency_errors, eps(FT(0))) # plotting 0 breaks the log scale
        label = alg_name
        plot!(plot1a, plot1_saveat, tendency_norms; label, linestyle)
        plot!(plot1b, plot1_saveat, tendency_errors; label, linestyle)
        if !(alg_name in no_increment_algs)
            increment_sols =
                solve(deepcopy(increment_prob), alg; solve_args...).u
            increment_errors = @. norm(increment_sols - tendency_sols)
            @test maximum(increment_errors) < 10000 * eps(FT) broken =
                alg_name == HOMMEM1 # TODO: why is this broken???
        end

        tendency_end_sols = map(
            dt -> solve(deepcopy(tendency_prob), alg; dt).u[end],
            plot2_dts,
        )
        tendency_end_errors = @. norm(tendency_end_sols - analytic_end_sol)
        _, computed_order = hcat(ones(length(plot2_dts)), log10.(plot2_dts)) \
            log10.(tendency_end_errors)
        @test computed_order ≈ predicted_order rtol = 0.1
        label = "$alg_name ($(@sprintf "%.3f" computed_order))"
        plot!(plot2, plot2_dts, tendency_end_errors; label, linestyle)
    end
    plot!(plot1b; ylim = (plot1b_ymin / 2, plot1b_ymax * 2))

    mkpath("output")
    file_suffix = "$(test_name)_$(lowercase(replace(algs_name, " " => "_")))"
    savefig(plot1a, joinpath("output", "solutions_$(file_suffix).png"))
    savefig(plot1b, joinpath("output", "errors_$(file_suffix).png"))
    savefig(plot2, joinpath("output", "orders_$(file_suffix).png"))
end

