#=
Type stability tests: verify key hot-path functions are type-stable.
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import ClimaTimeSteppers: ODEProblem, ODEFunction, IncrementingODEFunction, SplitODEProblem
using ClimaTimeSteppers: SparseCoeffs, fused_increment, SparseContainer

@testset "Type stability" begin

    @testset "fused_increment" begin
        u = [1.0, 2.0, 3.0]
        tend = ntuple(i -> u .* i, 3)
        coeffs = ones(3, 3)
        sc = SparseCoeffs(coeffs)
        dt = 0.5
        @inferred fused_increment(u, dt, sc, tend, Val(3))
    end

    @testset "SparseContainer" begin
        a1 = ones(3) .* 1
        a2 = ones(3) .* 2
        v = SparseContainer((a1, a2), (1, 3))
        @test (@inferred v[1]) == a1
        @test (@inferred v[3]) == a2
    end

    @testset "UpdateSignalHandler" begin
        handler = UpdateEvery(NewTimeStep)
        @test (@inferred CTS.needs_update!(handler, NewTimeStep(0.0))) == true

        handler_n = UpdateEveryN(3, NewNewtonIteration)
        @inferred CTS.needs_update!(handler_n, NewNewtonIteration())
    end

    @testset "ConvergenceChecker" begin
        checker = ConvergenceChecker(; norm_condition = MaximumRelativeError(1e-8))
        cache = CTS.allocate_cache(checker, [1.0, 2.0])
        @inferred CTS.is_converged!(checker, cache, [1.0, 2.0], [0.01, 0.02], 1)
    end

    @testset "Solver stepping smoke tests" begin

        @testset "Explicit RK (SSP33ShuOsher)" begin
            prob = ODEProblem(
                ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -0.5 .* u)),
                [1.0, 2.0, 3.0],
                (0.0, 1.0),
                nothing,
            )
            alg = ExplicitAlgorithm(SSP33ShuOsher())
            int = CTS.init(prob, alg; dt = 0.1, save_everystep = false)
            CTS.step!(int)  # warmup
            # step_u! dispatches on cache type; verify step! doesn't error on second call
            CTS.step!(int)
            @test all(isfinite, int.u)
        end

        @testset "LSRK (LSRK54CarpenterKennedy)" begin
            prob = ODEProblem(
                IncrementingODEFunction{true}(
                    (du, u, p, t, α = true, β = false) ->
                        (du .= α .* (-0.5) .* u .+ β .* du),
                ),
                [1.0, 2.0, 3.0],
                (0.0, 1.0),
                nothing,
            )
            int = CTS.init(
                prob,
                LSRK54CarpenterKennedy();
                dt = 0.1,
                save_everystep = false,
            )
            CTS.step!(int)
            CTS.step!(int)
            @test all(isfinite, int.u)
        end

        @testset "IMEX ARK (ARS232)" begin
            n = 3
            Id = Matrix{Float64}(I, n, n)
            prob = ODEProblem(
                ClimaODEFunction(;
                    T_exp! = (du, u, p, t) -> (du .= 0.1 .* u),
                    T_imp! = ODEFunction(
                        (du, u, p, t) -> (du .= -0.5 .* u);
                        jac_prototype = zeros(n, n),
                        Wfact = (W, u, p, dtγ, t) -> (W .= -0.5 * dtγ .* Id .- Id),
                    ),
                ),
                ones(n),
                (0.0, 1.0),
                nothing,
            )
            alg = CTS.IMEXAlgorithm(ARS232(), NewtonsMethod(; max_iters = 2))
            int = CTS.init(
                deepcopy(prob),
                alg;
                dt = 0.1,
                save_everystep = false,
            )
            CTS.step!(int)
            CTS.step!(int)
            @test all(isfinite, int.u)
        end

        @testset "Rosenbrock (SSPKnoth)" begin
            n = 3
            Id = Matrix{Float64}(I, n, n)
            prob = ODEProblem(
                ClimaODEFunction(;
                    T_exp! = (du, u, p, t) -> (du .= 0.1 .* u),
                    T_imp! = ODEFunction(
                        (du, u, p, t) -> (du .= -0.5 .* u);
                        jac_prototype = zeros(n, n),
                        Wfact = (W, u, p, dtγ, t) -> (W .= -0.5 * dtγ .* Id .- Id),
                    ),
                ),
                ones(n),
                (0.0, 1.0),
                nothing,
            )
            alg = CTS.RosenbrockAlgorithm(ClimaTimeSteppers.tableau(SSPKnoth()))
            int = CTS.init(
                deepcopy(prob),
                alg;
                dt = 0.1,
                save_everystep = false,
            )
            CTS.step!(int)
            CTS.step!(int)
            @test all(isfinite, int.u)
        end

        @testset "Multirate (LSRK + MIS3C)" begin
            prob = SplitODEProblem(
                IncrementingODEFunction{true}(
                    (du, u, p, t, α = true, β = false) ->
                        (du .= α .* (-5.0) .* u .+ β .* du),
                ),
                IncrementingODEFunction{true}(
                    (du, u, p, t, α = true, β = false) ->
                        (du .= α .* (-0.5) .* u .+ β .* du),
                ),
                [1.0, 2.0, 3.0],
                (0.0, 1.0),
                nothing,
            )
            alg = Multirate(LSRK54CarpenterKennedy(), MIS3C())
            int = CTS.init(
                deepcopy(prob),
                alg;
                dt = 0.1,
                fast_dt = 0.01,
                save_everystep = false,
            )
            CTS.step!(int)
            CTS.step!(int)
            @test all(isfinite, int.u)
        end
    end

    @testset "Float32 stepping smoke tests" begin

        @testset "Explicit RK Float32" begin
            prob = ODEProblem(
                ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -0.5f0 .* u)),
                Float32[1.0, 2.0, 3.0],
                (0.0f0, 1.0f0),
                nothing,
            )
            alg = ExplicitAlgorithm(SSP33ShuOsher())
            int = CTS.init(prob, alg; dt = 0.1f0, save_everystep = false)
            CTS.step!(int)
            CTS.step!(int)
            @test eltype(int.u) == Float32
            @test all(isfinite, int.u)
        end

        @testset "LSRK Float32" begin
            prob = ODEProblem(
                IncrementingODEFunction{true}(
                    (du, u, p, t, α = true, β = false) ->
                        (du .= α .* (-0.5f0) .* u .+ β .* du),
                ),
                Float32[1.0, 2.0, 3.0],
                (0.0f0, 1.0f0),
                nothing,
            )
            int = CTS.init(
                prob,
                LSRK54CarpenterKennedy();
                dt = 0.1f0,
                save_everystep = false,
            )
            CTS.step!(int)
            CTS.step!(int)
            @test eltype(int.u) == Float32
            @test all(isfinite, int.u)
        end

        @testset "IMEX ARK Float32" begin
            n = 3
            Id = Matrix{Float32}(I, n, n)
            prob = ODEProblem(
                ClimaODEFunction(;
                    T_exp! = (du, u, p, t) -> (du .= 0.1f0 .* u),
                    T_imp! = ODEFunction(
                        (du, u, p, t) -> (du .= -0.5f0 .* u);
                        jac_prototype = zeros(Float32, n, n),
                        Wfact = (W, u, p, dtγ, t) ->
                            (W .= Float32(-0.5) * dtγ .* Id .- Id),
                    ),
                ),
                ones(Float32, n),
                (0.0f0, 1.0f0),
                nothing,
            )
            alg = CTS.IMEXAlgorithm(ARS232(), NewtonsMethod(; max_iters = 2))
            int = CTS.init(
                deepcopy(prob),
                alg;
                dt = 0.1f0,
                save_everystep = false,
            )
            CTS.step!(int)
            CTS.step!(int)
            @test eltype(int.u) == Float32
            @test all(isfinite, int.u)
        end
    end
end
