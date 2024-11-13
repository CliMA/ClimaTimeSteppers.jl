#=
julia --project=test
using Revise; include("test/tabulate_convergence_orders_imex_ssp.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import PrettyTables
import SciMLBase

include(joinpath(@__DIR__, "convergence_orders.jl"))
include(joinpath(@__DIR__, "convergence_utils.jl"))
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "problems.jl"))

function tabulate_convergence_orders_imex_ssp()
    tabs = [SSP222, SSP322, SSP332, SSP333, SSP433]
    tabs = map(t -> t(), tabs)
    test_cases = all_test_cases(Float64)
    results = convergence_order_results(tabs, test_cases)
    algs = algorithm.(tabs)
    prob_names = map(t -> t.test_name, test_cases)
    expected_orders = SciMLBase.alg_order.(tabs)

    tabulate_convergence_orders(prob_names, algs, results, expected_orders; tabs)
    return results
end
tabulate_convergence_orders_imex_ssp()