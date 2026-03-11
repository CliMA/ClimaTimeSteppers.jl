#=
Shared infrastructure for solver convergence tests.
Includes problem definitions, algorithm order mappings, and convergence utilities.
=#
using DiffEqBase, ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import SciMLBase
import PrettyTables

# Load problem definitions (shared across all solver tests)
include(joinpath(@__DIR__, "..", "problems.jl"))

# ============================================================================ #
# Algorithm order definitions
# ============================================================================ #

# Explicit RK
SciMLBase.alg_order(::SSP22Heuns) = 2
SciMLBase.alg_order(::SSP33ShuOsher) = 3
SciMLBase.alg_order(::RK4) = 4

# IMEX ARK — 1st order
for m in (ARS111, ARS121)
    @eval SciMLBase.alg_order(::$m) = 1
end

# IMEX ARK — 2nd order
for m in (
    ARS122,
    ARS232,
    ARS222,
    IMKG232a,
    IMKG232b,
    IMKG242a,
    IMKG242b,
    IMKG243a,
    IMKG252a,
    IMKG252b,
    IMKG253a,
    IMKG253b,
    IMKG254a,
    IMKG254b,
    IMKG254c,
    HOMMEM1,
    SSP222,
    SSP322,
    SSP332,
    SSPKnoth,
    ARK2GKC,
)
    @eval SciMLBase.alg_order(::$m) = 2
end

# IMEX ARK — 3rd order
for m in (ARS233, ARS343, ARS443, IMKG342a, IMKG343a, DBM453, SSP333, SSP433)
    @eval SciMLBase.alg_order(::$m) = 3
end

# IMEX ARK — 4th order
SciMLBase.alg_order(::ARK437L2SA1) = 4

# IMEX ARK — 5th order
SciMLBase.alg_order(::ARK548L2SA2) = 5

# ============================================================================ #
# Algorithm constructors
# ============================================================================ #

problem(test_case, tab::CTS.IMEXARKAlgorithmName) = test_case.split_prob
problem(test_case, tab) = test_case.prob

algorithm(tab::CTS.IMEXARKAlgorithmName) =
    CTS.IMEXAlgorithm(tab, NewtonsMethod(; max_iters = 2))
algorithm(tab::CTS.RosenbrockAlgorithmName) =
    CTS.RosenbrockAlgorithm(ClimaTimeSteppers.tableau(tab))
algorithm(tab) = tab

# ============================================================================ #
# Convergence testing infrastructure
# ============================================================================ #

"""
    convergence_errors(prob, sol, method, dts; kwargs...)

Compute the errors of `method` on `problem` by comparing to `solution`
on the set of `dt` values in `dts`.
"""
function convergence_errors(prob, sol, method, dts; kwargs...)
    hide_warning = (; kwargshandle = DiffEqBase.KeywordArgSilent)
    errs = map(dts) do dt
        prob_copy = deepcopy(prob)
        u = solve(
            prob_copy,
            method;
            dt = dt,
            save_everystep = false,
            saveat = (prob.tspan[2],),
            kwargs...,
            hide_warning...,
        )
        if applicable(sol, prob.u0, prob.p, prob.tspan[end])
            expected = sol(prob.u0, prob.p, prob.tspan[end])
        else
            expected = sol(prob.tspan[end])
        end
        norm(u.u[end] .- expected) / sqrt(length(expected))
    end
    return errs
end

"""
    convergence_order(prob, sol, method, dts; kwargs...)

Estimate the order of convergence of `method` on `problem`.
"""
function convergence_order(prob, sol, method, dts; kwargs...)
    errs = convergence_errors(prob, sol, method, dts; kwargs...)
    _, order_est = hcat(ones(length(dts)), log2.(dts)) \ log2.(errs)
    return order_est
end

default_expected_order(alg, tab::CTS.AbstractAlgorithmName) = SciMLBase.alg_order(tab)

function test_convergence_order!(test_case, tab, results = Dict())
    prob = problem(test_case, tab)
    alg = algorithm(tab)
    expected_order = default_expected_order(alg, tab)

    num_steps_scaling_factor = 4
    num_steps_powers =
        (-1:0.5:1) .- test_case.high_order_sample_shifts * max(0, expected_order - 3) / 2
    sampled_num_steps =
        test_case.default_num_steps .* num_steps_scaling_factor .^ num_steps_powers
    dts = test_case.t_end ./ round.(Int, sampled_num_steps)

    computed_order = convergence_order(prob, test_case.analytic_sol, alg, dts)
    results[test_case.test_name, typeof(alg)] = (; expected_order, computed_order)
    return nothing
end

pass_conv(theoretic, computed) =
    abs(computed - theoretic) / theoretic * 100 < 10 && computed > 0
fail_conv(theoretic, computed) =
    !pass_conv(theoretic, computed) && !super_conv(theoretic, computed)
super_conv(theoretic, computed) = (computed - theoretic) / theoretic * 100 > 10

function convergence_order_results(tabs, test_cases)
    results = Dict()
    for test_case in test_cases
        @info "------------------ Test case: $(test_case.test_name)"
        for tab in tabs
            test_convergence_order!(test_case, tab, results)
        end
    end
    return results
end

"""
    verify_convergence_orders(results, expected_orders, algs; prob_names, rtol=0.15)

Assert that computed convergence orders match expected orders within `rtol`
(relative tolerance). Super-convergence is allowed.
"""
function verify_convergence_orders(results, expected_orders, algs; prob_names, rtol = 0.15)
    for (i, alg) in enumerate(algs)
        expected = expected_orders[i]
        for name in prob_names
            (; computed_order) = results[name, typeof(alg)]
            pass = computed_order > expected * (1 - rtol) || computed_order > expected
            if !pass
                @warn "Convergence order shortfall" alg = nameof(typeof(alg)) name expected computed_order
            end
            @test pass
        end
    end
end

function tabulate_convergence_orders(
    prob_names,
    algs,
    results,
    expected_orders;
    tabs = nothing,
)
    data = hcat(map(prob_names) do name
        map(alg -> results[name, typeof(alg)], algs)
    end...)
    alg_names = if tabs ≠ nothing
        @. string(nameof(typeof(tabs)))
    else
        @. string(typeof(algs))
    end
    summary(result) = last(result)
    data_summary = map(d -> summary(d), data)

    table_data = hcat(alg_names, data_summary)
    percentage_fail =
        sum(fail_conv.(getindex.(data, 1), getindex.(data, 2))) / length(data) * 100
    @info "Percentage of failed convergence order tests: $percentage_fail"
    fail_conv_hl = PrettyTables.TextHighlighter(
        (data, i, j) -> j ≠ 1 && fail_conv(expected_orders[i], data[i, j]),
        PrettyTables.crayon"red bold",
    )
    super_conv_hl = PrettyTables.TextHighlighter(
        (data, i, j) -> j ≠ 1 && super_conv(expected_orders[i], data[i, j]),
        PrettyTables.crayon"yellow bold",
    )
    tab_column_hl = PrettyTables.TextHighlighter(
        (data, i, j) -> j == 1,
        PrettyTables.crayon"green bold",
    )

    column_labels = ["Tableau (theoretic)", prob_names...]

    PrettyTables.pretty_table(
        table_data;
        highlighters = [tab_column_hl, fail_conv_hl, super_conv_hl],
        title = "Computed convergence orders, red=fail, yellow=super-convergence",
        column_labels,
        alignment = :c,
        fit_table_in_display_vertically = false,
        fit_table_in_display_horizontally = false,
    )
end
