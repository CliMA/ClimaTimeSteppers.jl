using BenchmarkTools
import OrdinaryDiffEq # use import avoid namespace conflicts

@testset "Compare with OrdinaryDiffEq" begin
    (; t_end, probs, analytic_sol) = ark_analytic_sys
    FT = typeof(t_end)
    tendency_prob = probs[1]
    dt = t_end / 2^7

    cts_alg = Rosenbrock23(; linsolve = linsolve_direct)
    ode_alg = OrdinaryDiffEq.Rosenbrock23(; linsolve = linsolve_direct)

    cts_tendency_end_sol = solve(deepcopy(tendency_prob), cts_alg; dt).u[end]
    ode_tendency_end_sol =
        solve(deepcopy(tendency_prob), ode_alg; dt, adaptive = false).u[end]
    @test norm(cts_tendency_end_sol .- ode_tendency_end_sol) < eps(FT)

    @info "Benchmark Results for ClimaTimeSteppers.Rosenbrock23:"
    cts_trial = @benchmark solve($(deepcopy(tendency_prob)), $cts_alg, dt = $dt)
    display(cts_trial)

    @info "Benchmark Results for OrdinaryDiffEq.Rosenbrock23:"
    ode_trial = @benchmark solve(
        $(deepcopy(tendency_prob)),
        $ode_alg,
        dt = $dt,
        adaptive = false,
    )
    display(cts_trial)

    @test median(cts_trial).time ≈ median(ode_trial).time rtol = 0.04
end
