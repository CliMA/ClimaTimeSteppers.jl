#=
Tests for ForwardEulerODEFunction.
=#
using ClimaTimeSteppers, DiffEqBase, Test

@testset "ForwardEulerODEFunction" begin
    @testset "Basic forward Euler update" begin
        # du/dt = -0.5 * u, forward Euler: un = u + dt * (-0.5 * u) = u * (1 - 0.5*dt)
        fe_func = ForwardEulerODEFunction((un, u, p, t, dt) -> (un .= u .+ dt .* p .* u))

        u = [2.0, 4.0]
        un = similar(u)
        p = -0.5
        t = 0.0
        dt = 0.1

        fe_func(un, u, p, t, dt)
        @test un ≈ u .+ dt .* p .* u  # [2*(1-0.05), 4*(1-0.05)] = [1.9, 3.8]
        @test un ≈ [1.9, 3.8]
    end

    @testset "ODEFunction wrapping is identity" begin
        fe_func = ForwardEulerODEFunction((un, u, p, t, dt) -> nothing)
        @test DiffEqBase.ODEFunction(fe_func) === fe_func
        @test DiffEqBase.ODEFunction{true}(fe_func) === fe_func
    end

    @testset "Optional fields default to nothing" begin
        fe_func = ForwardEulerODEFunction((un, u, p, t, dt) -> nothing)
        @test fe_func.jac_prototype === nothing
        @test fe_func.Wfact === nothing
        @test fe_func.tgrad === nothing
    end

    @testset "Keyword constructor with jac_prototype" begin
        jac = zeros(2, 2)
        wfact = (W, u, p, γ, t) -> nothing
        fe_func = ForwardEulerODEFunction(
            (un, u, p, t, dt) -> nothing;
            jac_prototype = jac,
            Wfact = wfact,
        )
        @test fe_func.jac_prototype === jac
        @test fe_func.Wfact === wfact
    end

    @testset "Multi-step convergence" begin
        # Verify that repeated forward Euler steps converge to exp(-t)
        fe_func = ForwardEulerODEFunction((un, u, p, t, dt) -> (un .= u .+ dt .* p .* u))
        p = -1.0
        u = [1.0]
        t = 0.0
        dt = 0.001
        for _ in 1:1000
            un = similar(u)
            fe_func(un, u, p, t, dt)
            u = un
            t += dt
        end
        @test u[1] ≈ exp(-1.0) atol = 0.001
    end
end
