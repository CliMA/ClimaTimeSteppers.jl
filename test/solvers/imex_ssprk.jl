#=
julia --project=test
using Revise; include("test/solvers/imex_ssprk.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import PrettyTables
import SciMLBase

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

function tabulate_convergence_orders_imex_ssp()
    tabs = [SSP222, SSP322, SSP332, SSP333, SSP433]
    tabs = map(t -> t(), tabs)
    test_cases = all_test_cases(Float64)
    results = convergence_order_results(tabs, test_cases)
    algs = algorithm.(tabs)
    prob_names = map(t -> t.test_name, test_cases)
    expected_orders = SciMLBase.alg_order.(tabs)

    tabulate_convergence_orders(prob_names, algs, results, expected_orders; tabs)

    reliable_probs = filter(n -> !endswith(n, "(FD)"), prob_names)
    verify_convergence_orders(results, expected_orders, algs; prob_names = reliable_probs)
    return results
end
tabulate_convergence_orders_imex_ssp()
