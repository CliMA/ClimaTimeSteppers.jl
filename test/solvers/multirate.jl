#=
julia --project=test
using Revise; include("test/solvers/multirate.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

@testset "Multirate convergence" begin
    dts = 0.5 .^ (4:7)

    names_probs_sols = [
        (:imex_auto, imex_autonomous_prob(Array{Float64}), imex_autonomous_sol),
        (:imex_nonauto, imex_nonautonomous_prob(Array{Float64}), imex_nonautonomous_sol),
        # State-coupled slow tendency: exposes inner/outer stage-coupling bugs
        # (e.g. an inner integrator that overshoots its substep) that the two
        # problems above cannot, since their slow tendency ignores the state.
        # Pairing this problem with WSRK2 below is the key regression guard
        # for the WSRK tstop fix.
        (:imex_statedep, imex_statedep_slow_prob(Array{Float64}), imex_statedep_slow_sol),
    ]

    algs_orders = [
        (Multirate(LSRK54CarpenterKennedy(), WSRK2()), 2),
        (Multirate(LSRK54CarpenterKennedy(), WSRK3()), 2),
        (Multirate(LSRK54CarpenterKennedy(), LSRK54CarpenterKennedy()), 4),
        (Multirate(LSRK54CarpenterKennedy(), MIS2()), 2),
        (Multirate(LSRK54CarpenterKennedy(), MIS3C()), 2),
        (Multirate(LSRK54CarpenterKennedy(), MIS4()), 3),
        (Multirate(LSRK54CarpenterKennedy(), MIS4a()), 3),
        (Multirate(LSRK54CarpenterKennedy(), TVDMISA()), 2),
        (Multirate(LSRK54CarpenterKennedy(), TVDMISB()), 2),
    ]
    for (name, prob, sol) in names_probs_sols
        for (alg, expected_order) in algs_orders
            computed_order = convergence_order(prob, sol, alg, dts; fast_dt = 0.5^12)
            @test abs(computed_order - expected_order) / expected_order < 0.15
        end
    end
end
