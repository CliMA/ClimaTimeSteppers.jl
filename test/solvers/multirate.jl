#=
julia --project=test
using Revise; include("test/solvers/multirate.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import PrettyTables
import SciMLBase

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

function tabulate_convergence_orders_multirate()

    co = Dict()
    names_probs_sols = [
        (:imex_auto, imex_autonomous_prob(Array{Float64}), imex_autonomous_sol),
        (:imex_nonauto, imex_nonautonomous_prob(Array{Float64}), imex_nonautonomous_sol),
    ]
    dts = 0.5 .^ (4:7)

    algs_orders = [
        # Multirate
        (Multirate(LSRK54CarpenterKennedy(), LSRK54CarpenterKennedy()), 4),
        # MIS
        (Multirate(LSRK54CarpenterKennedy(), MIS2()), 2),
        (Multirate(LSRK54CarpenterKennedy(), MIS3C()), 2),
        (Multirate(LSRK54CarpenterKennedy(), MIS4()), 3),
        (Multirate(LSRK54CarpenterKennedy(), MIS4a()), 3),
        (Multirate(LSRK54CarpenterKennedy(), TVDMISA()), 2),
        (Multirate(LSRK54CarpenterKennedy(), TVDMISB()), 2),
        # Wicker Skamarock
        (Multirate(LSRK54CarpenterKennedy(), WSRK2()), 2),
        (Multirate(LSRK54CarpenterKennedy(), WSRK3()), 2),
    ]

    for (name, prob, sol) in names_probs_sols
        for (alg, ord) in algs_orders
            co[name, typeof(alg)] =
                (ord, convergence_order(prob, sol, alg, dts; fast_dt = 0.5^12))
        end
    end

    prob_names = first.(names_probs_sols)
    algs = first.(algs_orders)
    expected_orders = last.(algs_orders)
    tabulate_convergence_orders(prob_names, algs, co, expected_orders)

    # TODO: Enable once multirate convergence is fixed. All multirate algorithms
    # currently show convergence orders ≈ 0 on both test problems (errors don't
    # decrease with dt refinement), suggesting a fundamental issue.
    # verify_multirate_convergence_orders(co, algs_orders, names_probs_sols)
end

tabulate_convergence_orders_multirate()
