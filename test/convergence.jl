using ClimaTimeSteppers, LinearAlgebra, Printf, Plots, Test

include("utils.jl")

dt_fracs = 0.5 .^ (4:7)

for (prob, sol) in [
    (linear_prob, linear_sol)
    (sincos_prob, sincos_sol)
]
    dts = dt_fracs .* prob.tspan[end]

    @test convergence_order(prob, sol, LSRKEulerMethod(), dts)              ≈ 1 atol=0.1
    @test convergence_order(prob, sol, LSRK54CarpenterKennedy(), dts)       ≈ 4 atol=0.05
    @test convergence_order(prob, sol, LSRK144NiegemannDiehlBusch(),  dts)  ≈ 4 atol=0.05

    @test convergence_order(prob, sol, SSPRK22Heuns(), dts)                 ≈ 2 atol=0.05
    @test convergence_order(prob, sol, SSPRK22Ralstons(), dts)              ≈ 2 atol=0.05
    @test convergence_order(prob, sol, SSPRK33ShuOsher(), dts)              ≈ 3 atol=0.05
    @test convergence_order(prob, sol, SSPRK34SpiteriRuuth(), dts)          ≈ 3 atol=0.05
end


# ForwardEulerODEFunction
for (prob, sol) in [
    (linear_prob_fe, linear_sol)
    (sincos_prob_fe, sincos_sol)
]
    dts = dt_fracs .* prob.tspan[end]

    @test convergence_order(prob, sol, SSPRK22Heuns(), dts)                 ≈ 2 atol=0.05
    @test convergence_order(prob, sol, SSPRK22Ralstons(), dts)              ≈ 2 atol=0.05
    @test convergence_order(prob, sol, SSPRK33ShuOsher(), dts)              ≈ 3 atol=0.05
    @test convergence_order(prob, sol, SSPRK34SpiteriRuuth(), dts)          ≈ 3 atol=0.05

end

@testset "Test Algorithms and Generate Plots" begin
    imex_ark_dict = let
        dict = Dict{Type, Int}()
        algs1 = (ARS111, ARS121)
        algs2 = (ARS122, ARS232, ARS222, IMKG232a, IMKG232b, IMKG242a, IMKG242b)
        algs2 = (algs2..., IMKG252a, IMKG252b, IMKG253a, IMKG253b, IMKG254a)
        algs2 = (algs2..., IMKG254b, IMKG254c, HOMMEM1)
        algs3 = (ARS233, ARS343, ARS443, IMKG342a, IMKG343a, DBM453)
        for (algs, order) in ((algs1, 1), (algs2, 2), (algs3, 3))
            for alg in algs
                dict[alg] = order
            end
        end
        dict
    end

    test_algs("IMEX ARK", imex_ark_dict, ark_analytic_sys, 7)

    test_algs("IMEX ARK", imex_ark_dict, ark_analytic_nonlin, 10)

    # For some bizarre reason, ARS121 converges with order 2 for ark_analytic.
    imex_ark_dict_ark_analytic = copy(imex_ark_dict)
    imex_ark_dict_ark_analytic[ARS121] = 2
    test_algs("IMEX ARK", imex_ark_dict_ark_analytic, ark_analytic, 15)

    rosenbrock_dict = let
        dict = Dict{Type, Int}()
        algs2 = (Rosenbrock23, SSPKnoth)
        algs3 = (ROS3w, ROS3Pw, ROS34PW1a, ROS34PW1b, ROS34PW2)
        algs4 = (RODASP2, ROS34PW3)
        for (algs, order) in ((algs2, 2), (algs3, 3), (algs4, 4))
            for alg in algs
                dict[alg] = order
            end
        end
        dict
    end
    no_increment_algs = (ROS3w, ROS3Pw, ROS34PW1a, ROS34PW1b)

    # 6 also works, but we'll keep it at 7 to match the IMEX ARK plots.
    test_algs(
        "Rosenbrock",
        rosenbrock_dict,
        ark_analytic_sys,
        7;
        no_increment_algs,
    )

    # RODASP2 needs a larger dt than the other methods, so it's skipped here.
    rosenbrock_dict′ = copy(rosenbrock_dict)
    delete!(rosenbrock_dict′, RODASP2)
    test_algs(
        "Rosenbrock",
        rosenbrock_dict′,
        ark_analytic_nonlin,
        10;
        no_increment_algs,
    )
    
    # The 3rd order algorithms need a larger dt than the 2nd order ones, and
    # neither of the 4th order algorithms converge within an rtol of 0.1 of
    # the predicted order for any value of dt.
    algs_name = "Rosenbrock 2nd Order"
    rosenbrock2_dict = Dict((Rosenbrock23, SSPKnoth) .=> 2)
    test_algs(algs_name, rosenbrock2_dict, ark_analytic, 15)
    algs_name = "Rosenbrock 3rd Order"
    rosenbrock3_dict =
        Dict((ROS3w, ROS3Pw, ROS34PW1a, ROS34PW1b, ROS34PW2) .=> 3)
    test_algs(algs_name, rosenbrock3_dict, ark_analytic, 12; no_increment_algs)
end

#=
if ArrayType == Array
for (prob, sol) in [
    imex_autonomous_prob => imex_autonomous_sol,
    #imex_nonautonomous_prob => imex_nonautonomous_sol,
]
# IMEX
    @test convergence_order(prob, sol, ARK1ForwardBackwardEuler(linsolve=DirectSolver), dts)       ≈ 1 atol=0.1
    @test convergence_order(prob, sol, ARK2ImplicitExplicitMidpoint(linsolve=DirectSolver), dts)   ≈ 2 atol=0.05
    @test convergence_order(prob, sol, ARK2GiraldoKellyConstantinescu(linsolve=DirectSolver), dts) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, ARK2GiraldoKellyConstantinescu(linsolve=DirectSolver; paperversion=true), dts) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, ARK437L2SA1KennedyCarpenter(linsolve=DirectSolver), dts)    ≈ 4 atol=0.05
    @test convergence_order(prob, sol, ARK548L2SA2KennedyCarpenter(linsolve=DirectSolver), dts)    ≈ 5 atol=0.05
end
end
for (prob, sol) in [
    imex_autonomous_prob => imex_autonomous_sol,
    imex_nonautonomous_prob => imex_nonautonomous_sol,
    # kpr_multirate_prob => kpr_sol,
]
    # Multirate
    @test convergence_order(prob, sol, Multirate(LSRK54CarpenterKennedy(),LSRK54CarpenterKennedy()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 4 atol=0.05
    # MIS
    @test convergence_order(prob, sol, Multirate(LSRK54CarpenterKennedy(),MIS2()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, Multirate(LSRK54CarpenterKennedy(),MIS3C()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, Multirate(LSRK54CarpenterKennedy(),MIS4()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 3 atol=0.05
    @test convergence_order(prob, sol, Multirate(LSRK54CarpenterKennedy(),MIS4a()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 3 atol=0.05
    @test convergence_order(prob, sol, Multirate(LSRK54CarpenterKennedy(),TVDMISA()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, Multirate(LSRK54CarpenterKennedy(),TVDMISB()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 2 atol=0.05

    # Wicker Skamarock
    @test convergence_order(prob, sol, Multirate(LSRK54CarpenterKennedy(),WSRK2()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, Multirate(LSRK54CarpenterKennedy(),WSRK3()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 2 atol=0.05


    end
=#