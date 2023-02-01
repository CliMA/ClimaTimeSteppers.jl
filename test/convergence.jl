using ClimaTimeSteppers, LinearAlgebra, Test

include(joinpath(@__DIR__, "convergence_orders.jl"))
include(joinpath(@__DIR__, "convergence_utils.jl"))
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "problems.jl"))

@testset "LSRK and SSP convergence" begin
    dts = 0.5 .^ (4:7)

    for (prob, sol, tscale) in [
        (linear_prob(), linear_sol, 1)
        (sincos_prob(), sincos_sol, 1)
    ]

        @test convergence_order(prob, sol, LSRKEulerMethod(), dts .* tscale) ≈ 1 atol = 0.1
        @test convergence_order(prob, sol, LSRK54CarpenterKennedy(), dts .* tscale) ≈ 4 atol = 0.05
        @test convergence_order(prob, sol, LSRK144NiegemannDiehlBusch(), dts .* tscale) ≈ 4 atol = 0.05

        @test convergence_order(prob, sol, SSPRK22Heuns(), dts .* tscale) ≈ 2 atol = 0.05
        @test convergence_order(prob, sol, SSPRK22Ralstons(), dts .* tscale) ≈ 2 atol = 0.05
        @test convergence_order(prob, sol, SSPRK33ShuOsher(), dts .* tscale) ≈ 3 atol = 0.05
        @test convergence_order(prob, sol, SSPRK34SpiteriRuuth(), dts .* tscale) ≈ 3 atol = 0.05
    end

    for (prob, sol, tscale) in [(linear_prob_wfactt(), linear_sol, 1)]
        @test convergence_order(prob, sol, SSPKnoth(linsolve = linsolve_direct), dts .* tscale) ≈ 2 atol = 0.05

    end


    # ForwardEulerODEFunction
    for (prob, sol, tscale) in [
        (linear_prob_fe(), linear_sol, 1)
        (sincos_prob_fe(), sincos_sol, 1)
    ]
        @test convergence_order(prob, sol, SSPRK22Heuns(), dts .* tscale) ≈ 2 atol = 0.05
        @test convergence_order(prob, sol, SSPRK22Ralstons(), dts .* tscale) ≈ 2 atol = 0.05
        @test convergence_order(prob, sol, SSPRK33ShuOsher(), dts .* tscale) ≈ 3 atol = 0.05
        @test convergence_order(prob, sol, SSPRK34SpiteriRuuth(), dts .* tscale) ≈ 3 atol = 0.05

    end
end

#=
using Revise; include("test/convergence.jl")
results = tabulate_convergence_orders_imex_ark();
=#
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
    ]
    tabs = map(t -> t(), tabs)
    test_cases = all_test_cases(Float64)
    results = convergence_order_results(tabs, test_cases)
    algs = algorithm.(tabs)
    prob_names = map(t -> t.test_name, test_cases)
    expected_orders = ODE.alg_order.(tabs)
    tabulate_convergence_orders(prob_names, algs, results, expected_orders; tabs)
    return results
end
tabulate_convergence_orders_imex_ark()

function tabulate_convergence_orders_ark()
    # IMEX
    co = Dict()
    names_probs_sols = [
        (:auto, imex_autonomous_prob(Array{Float64}), imex_autonomous_sol),
        (:nonauto, imex_nonautonomous_prob(Array{Float64}), imex_nonautonomous_sol),
    ]
    algs_orders = [
        (ARK1ForwardBackwardEuler(DirectSolver), 1),
        (ARK2ImplicitExplicitMidpoint(DirectSolver), 2),
        (ARK2GiraldoKellyConstantinescu(DirectSolver), 2),
        (ARK2GiraldoKellyConstantinescu(DirectSolver; paperversion = true), 2),
        (ARK437L2SA1KennedyCarpenter(DirectSolver), 4),
        (ARK548L2SA2KennedyCarpenter(DirectSolver), 5),
    ]
    dts = 0.5 .^ (4:7)
    for (name, prob, sol) in names_probs_sols
        for (alg, ord) in algs_orders
            co[name, typeof(alg)] = (ord, convergence_order(prob, sol, alg, dts))
        end
    end
    prob_names = first.(names_probs_sols)
    algs = first.(algs_orders)
    expected_orders = last.(algs_orders)
    tabulate_convergence_orders(prob_names, algs, co, expected_orders)
end

tabulate_convergence_orders_ark()

function tabulate_convergence_orders_imex_ssp()
    tabs = [SSP222, SSP322, SSP332, SSP333, SSP433]
    tabs = map(t -> t(), tabs)
    test_cases = all_test_cases(Float64)
    results = convergence_order_results(tabs, test_cases)
    algs = algorithm.(tabs)
    prob_names = map(t -> t.test_name, test_cases)
    expected_orders = ODE.alg_order.(tabs)

    tabulate_convergence_orders(prob_names, algs, results, expected_orders; tabs)
    return results
end
tabulate_convergence_orders_imex_ssp()


function tabulate_convergence_orders_multirate()

    co = Dict()
    names_probs_sols = [
        (:imex_auto, imex_autonomous_prob(Array{Float64}), imex_autonomous_sol),
        (:imex_nonauto, imex_nonautonomous_prob(Array{Float64}), imex_nonautonomous_sol),
        # (:kpr_multirate, kpr_multirate_prob(), kpr_sol),
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
            co[name, typeof(alg)] = (ord, convergence_order(prob, sol, alg, dts; fast_dt = 0.5^12))
        end
    end

    prob_names = first.(names_probs_sols)
    algs = first.(algs_orders)
    expected_orders = last.(algs_orders)
    tabulate_convergence_orders(prob_names, algs, co, expected_orders)
end

tabulate_convergence_orders_multirate()
