#=
julia --project=test
using Revise; include("test/solvers/imex_ark.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import PrettyTables
import SciMLBase

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

function tabulate_convergence_orders_imex_ark()
    tabs = [
        ARS111,
        ARS121,
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
        ARS233,
        ARS343,
        ARS443,
        IMKG342a,
        IMKG343a,
        DBM453,
        ARK2GKC,
        ARK437L2SA1,
        ARK548L2SA2,
    ]
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
tabulate_convergence_orders_imex_ark()
