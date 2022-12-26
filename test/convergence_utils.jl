import ODEConvergenceTester as OCT
import OrdinaryDiffEq as ODE

"""
    DirectSolver

A linear solver which forms the full matrix of a linear operator and its LU factorization.
"""
struct DirectSolver end

DirectSolver(args...) = DirectSolver()

function (::DirectSolver)(x, A, b, matrix_updated; kwargs...)
    n = length(x)
    M = mapslices(y -> mul!(similar(y), A, y), Matrix{eltype(x)}(I, n, n), dims = 1)
    x .= M \ b
end

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
        u = solve(prob_copy, method; dt = dt, saveat = (prob.tspan[2],), kwargs...)
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
    _, order_est = hcat(ones(length(dts)), log2.(dts)) \ log2.(errs)
    return order_est
end

function default_expected_order(alg, tab)
    return if tab isa CTS.AbstractIMEXARKTableau
        CTS.theoretical_convergence_order(tab)
    else
        ODE.alg_order(alg)
    end
end

function test_convergence_order!(test_case, tab, results = Dict(); refinement_range)
    prob, alg = problem_algo(test_case, tab)
    expected_order = default_expected_order(alg, tab())
    cr = OCT.refinement_study(
        prob,
        alg;
        verbose = false,
        expected_order,
        refinement_range, # ::UnitRange, 2:4 is more fine than 1:3
    )
    computed_order = maximum(cr.computed_order)
    results[tab, test_case.test_name] = (; expected_order, computed_order)
    return nothing
end

distance(theoretic, computed) = abs(computed - theoretic) / theoretic
pass_conv(theoretic, computed) = distance(theoretic, computed) * 100 < 10
fail_conv(theoretic, computed) = !pass_conv(computed, theoretic) && !super_conv(theoretic, computed)
super_conv(theoretic, computed) = (computed - theoretic) / theoretic * 100 > 10

#= Calls `test_convergence_order!` for each combination of test case
and algorithm, returns a `Dict` of the results. =#
function convergence_order_results(tabs, test_cases)
    results = Dict()
    for test_case in test_cases
        @info "------------------ Test case: $(test_case.test_name)"
        for tab in tabs
            # @info "Running refinement study on $(nameof(tab))"
            test_convergence_order!(test_case, tab, results; refinement_range = 5:9)
        end
    end
    return results
end

function tabulate_convergence_orders(test_cases, tabs, results)
    columns = map(test_cases) do test_case
        map(tab -> results[tab, test_case.test_name], tabs)
    end
    expected_order = map(tab -> default_expected_order(nothing, tab()), tabs)
    tab_names = map(tab -> "$tab ($(default_expected_order(nothing, tab())))", tabs)
    data = hcat(columns...)
    summary(result) = result.computed_order
    data_summary = map(d -> summary(d), data)

    table_data = hcat(tab_names, data_summary)
    precentage_fail = sum(fail_conv.(getindex.(data, 1), getindex.(data, 2))) / length(data) * 100
    @info "Percentage of failed convergence order tests: $precentage_fail"
    fail_conv_hl = PrettyTables.Highlighter(
        (data, i, j) -> j ≠ 1 && fail_conv(expected_order[i], data[i, j]),
        PrettyTables.crayon"red bold",
    )
    super_conv_hl = PrettyTables.Highlighter(
        (data, i, j) -> j ≠ 1 && super_conv(expected_order[i], data[i, j]),
        PrettyTables.crayon"yellow bold",
    )
    tab_column_hl = PrettyTables.Highlighter((data, i, j) -> j == 1, PrettyTables.crayon"green bold")
    test_case_names = map(test_case -> test_case.test_name, test_cases)

    header = (["Tableau (theoretic)", test_case_names...],
    # ["", ["" for tc in test_case_names]...],
    )

    PrettyTables.pretty_table(
        table_data;
        header_crayon = PrettyTables.crayon"green bold",
        highlighters = (tab_column_hl, fail_conv_hl, super_conv_hl),
        title = "Computed convergence orders, red=fail, yellow=super-convergence",
        header,
        alignment = :c,
        crop = :none,
    )
end
