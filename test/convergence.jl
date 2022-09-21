using DiffEqBase, ClimaTimeSteppers, LinearAlgebra, Test

include("utils.jl")

dts = 0.5 .^ (4:7)

for (prob, sol, tscale) in [
    (linear_prob, linear_sol, 1)
    (sincos_prob, sincos_sol, 1)
]

    @test convergence_order(prob, sol, LSRKEulerMethod(), dts.*tscale)     ≈ 1 atol=0.1
    @test convergence_order(prob, sol, LSRK54CarpenterKennedy(), dts.*tscale)       ≈ 4 atol=0.05
    @test convergence_order(prob, sol, LSRK144NiegemannDiehlBusch(),  dts.*tscale)   ≈ 4 atol=0.05

    @test convergence_order(prob, sol, SSPRK22Heuns(), dts.*tscale)                 ≈ 2 atol=0.05
    @test convergence_order(prob, sol, SSPRK22Ralstons(), dts.*tscale)              ≈ 2 atol=0.05
    @test convergence_order(prob, sol, SSPRK33ShuOsher(), dts.*tscale)              ≈ 3 atol=0.05
    @test convergence_order(prob, sol, SSPRK34SpiteriRuuth(), dts.*tscale)          ≈ 3 atol=0.05
end

for (prob, sol, tscale) in [
    (linear_prob_wfactt, linear_sol, 1)
]
    @test convergence_order(prob, sol, SSPKnoth(linsolve=linsolve_direct), dts.*tscale) ≈ 2 atol=0.05

end


# ForwardEulerODEFunction
for (prob, sol, tscale) in [
    (linear_prob_fe, linear_sol, 1)
    (sincos_prob_fe, sincos_sol, 1)
]
    @test convergence_order(prob, sol, SSPRK22Heuns(), dts.*tscale)                 ≈ 2 atol=0.05
    @test convergence_order(prob, sol, SSPRK22Ralstons(), dts.*tscale)              ≈ 2 atol=0.05
    @test convergence_order(prob, sol, SSPRK33ShuOsher(), dts.*tscale)              ≈ 3 atol=0.05
    @test convergence_order(prob, sol, SSPRK34SpiteriRuuth(), dts.*tscale)          ≈ 3 atol=0.05

end

@testset "IMEX ARK Methods" begin
    algs1 = (ARS111, ARS121)
    algs2 = (ARS122, ARS232, ARS222, IMKG232a, IMKG232b, IMKG242a, IMKG242b)
    algs2 = (algs2..., IMKG252a, IMKG252b, IMKG253a, IMKG253b, IMKG254a)
    algs2 = (algs2..., IMKG254b, IMKG254c, HOMMEM1)
    algs3 = (ARS233, ARS343, ARS443, IMKG342a, IMKG343a, DBM453)
    for (algorithm_names, order) in ((algs1, 1), (algs2, 2), (algs3, 3))
        for algorithm_name in algorithm_names
            for (problem, solution) in (
                (split_linear_prob_wfact_split, linear_sol),
                (split_linear_prob_wfact_split_fe, linear_sol),
            )
                @show algorithm_name
                algorithm = algorithm_name(
                    NewtonsMethod(; linsolve = linsolve_direct, max_iters = 1),
                ) # the problem is linear, so more iters have no effect
                @test isapprox(
                    convergence_order(problem, solution, algorithm, dts),
                    order;
                    rtol = 0.01,
                )
            end
        end
    end
end

@testset "Rosenbrock-W Methods" begin
    algs2 = (Rosenbrock23, SSPKnoth, RODASP2, ROS3w, ROS3Pw)
    algs3 = (ROS34PW1a, ROS34PW1b, ROS34PW2, ROS34PW3)
    for (algorithm_names, order) in ((algs2, 2), (algs3, 3))
        for algorithm_name in algorithm_names
            for (problem, solution) in (
                (linear_prob_inexact_wfact, linear_sol),
            )
                algorithm = algorithm_name(; linsolve = linsolve_direct)
                @test isapprox(
                    convergence_order(problem, solution, algorithm, dts),
                    order;
                    rtol = 0.01,
                )
            end
        end
    end
end

@testset "Rosenbrock-W Methods Limiters Formulation" begin
    # only methods with invertible matrices â can be tested here
    algs2 = (Rosenbrock23, SSPKnoth, RODASP2)
    algs3 = (ROS34PW2, ROS34PW3)
    for (algorithm_names, order) in ((algs2, 2), (algs3, 3))
        for algorithm_name in algorithm_names
            for (problem, solution) in (
                (linear_prob_inexact_wfact_fe, linear_sol),
            )
                algorithm = algorithm_name(;
                    linsolve = linsolve_direct,
                    multiply! = multiply_direct!,
                    set_Δtγ! = set_Δtγ_direct!,
                )
                @test isapprox(
                    convergence_order(problem, solution, algorithm, dts),
                    order;
                    rtol = 0.01,
                )
            end
        end
    end
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