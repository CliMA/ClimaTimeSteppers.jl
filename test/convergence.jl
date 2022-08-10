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

@test convergence_order(split_linear_prob_wfact_split1, linear_sol, ARS111(linsolve=linsolve_direct), dts) ≈ 1 atol=0.05
@test convergence_order(split_linear_prob_wfact_split1, linear_sol, ARS121(linsolve=linsolve_direct), dts) ≈ 1 atol=0.05
@test convergence_order(split_linear_prob_wfact_split1, linear_sol, ARS232(linsolve=linsolve_direct), dts) ≈ 2 atol=0.05
@test convergence_order(split_linear_prob_wfact_split1, linear_sol, ARS343(linsolve=linsolve_direct), dts) ≈ 3 atol=0.05

@test convergence_order(split_linear_prob_wfact_split2, linear_sol, ARS111(linsolve=linsolve_direct), dts) ≈ 1 atol=0.05
@test convergence_order(split_linear_prob_wfact_split2, linear_sol, ARS121(linsolve=linsolve_direct), dts) ≈ 1 atol=0.05
@test convergence_order(split_linear_prob_wfact_split2, linear_sol, ARS232(linsolve=linsolve_direct), dts) ≈ 2 atol=0.05
@test convergence_order(split_linear_prob_wfact_split2, linear_sol, ARS343(linsolve=linsolve_direct), dts) ≈ 3 atol=0.05
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