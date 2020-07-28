using DiffEqBase, TimeMachine, LinearAlgebra, Test

include("problems.jl")
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


for (prob, sol) in [
    imex_autonomous_prob => imex_autonomous_sol,
    #imex_nonautonomous_prob => imex_nonautonomous_sol,
]
    # IMEX
    @test convergence_order(prob, sol, ARK1ForwardBackwardEuler(DirectSolver), dts)       ≈ 1 atol=0.1
    @test convergence_order(prob, sol, ARK2ImplicitExplicitMidpoint(DirectSolver), dts)   ≈ 2 atol=0.05
    @test convergence_order(prob, sol, ARK2GiraldoKellyConstantinescu(DirectSolver), dts) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, ARK2GiraldoKellyConstantinescu(DirectSolver; paperversion=true), dts) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, ARK437L2SA1KennedyCarpenter(DirectSolver), dts)    ≈ 4 atol=0.05
    @test convergence_order(prob, sol, ARK548L2SA2KennedyCarpenter(DirectSolver), dts)    ≈ 5 atol=0.05
end

for (prob, sol) in [
    imex_autonomous_prob => imex_autonomous_sol,
    imex_nonautonomous_prob => imex_nonautonomous_sol,
    # kpr_multirate_prob => kpr_sol,
]
    # Multirate
    @test convergence_order(prob, sol, MultirateRungeKutta(LSRK54CarpenterKennedy(),LSRK54CarpenterKennedy()), dts;
        fast_dt = 0.5^12, adjustfinal=true) ≈ 4 atol=0.05

end
