using DiffEqBase, TimeMachine, LinearAlgebra, Test

include("problems.jl")


function convergence_errors(prob, sol, method, dts; kwargs...)
    errs = map(dts) do dt
         # copy the problem so we don't mutate u0
        prob_copy = deepcopy(prob)
        u = solve(prob_copy, method; dt=dt, kwargs...)
        norm(u .- sol(prob.u0, prob.p, prob.tspan[end]))
    end
    return errs
end


"""
    convergence_order(problem, solution, method, dts; kwargs...)

Estimate the order of the rate of convergence of `method` on `problem` by comparing to
 `solution` the set of `dt` values in `dts`.

`solution` should be a function with a method `solution(u0, p, t)`.
"""
function convergence_order(prob, sol, method, dts; kwargs...)
    errs = convergence_errors(prob, sol, method, dts; kwargs...)
    # find slope coefficient in log scale
    _,order_est = hcat(ones(length(dts)), log2.(dts)) \ log2.(errs)
    return order_est
end

struct DirectSolver end

DirectSolver(args...) = DirectSolver()

function (::DirectSolver)(x,A,b,matrix_updated; kwargs...)
    n = length(x)
    M = mapslices(y -> mul!(similar(y), A, y), Matrix{eltype(x)}(I,n,n), dims=1)
    x .= M \ b 
end


dts = 0.5 .^ (4:7)


convergence_errors(kpr_multirate_prob, kpr_sol, MultirateRungeKutta(LSRK54CarpenterKennedy(),LSRK54CarpenterKennedy()), dts;
 fast_dt = 0.5^12)

#=

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
    imex_linconst_prob => imex_linconst_sol,
]

    @test convergence_order(prob, sol, ARK1ForwardBackwardEuler(DirectSolver), dts)       ≈ 1 atol=0.05
    @test convergence_order(prob, sol, ARK2ImplicitExplicitMidpoint(DirectSolver), dts)   ≈ 2 atol=0.05
    @test convergence_order(prob, sol, ARK2GiraldoKellyConstantinescu(DirectSolver), dts) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, ARK2GiraldoKellyConstantinescu(DirectSolver; paperversion=true), dts) ≈ 2 atol=0.05
    @test convergence_order(prob, sol, ARK437L2SA1KennedyCarpenter(DirectSolver), dts)    ≈ 4 atol=0.05
    @test convergence_order(prob, sol, ARK548L2SA2KennedyCarpenter(DirectSolver), dts)    ≈ 5 atol=0.05

end
=#