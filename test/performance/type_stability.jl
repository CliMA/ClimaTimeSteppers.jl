#=
Type stability tests: verify key hot-path functions are type-stable.

Solver stepping is checked with JET.@test_opt on the per-solver `step_u!`
kernel (the hot path). `target_modules = (ClimaTimeSteppers,)` restricts the
analysis to library code, so type instabilities in the toy problem closures
defined here (e.g. a Wfact that captures a non-const global) are ignored — we
are testing the solver, not the test fixtures.
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import JET
import ClimaTimeSteppers as CTS
import ClimaTimeSteppers: ODEProblem, ODEFunction, IncrementingODEFunction, SplitODEProblem
using ClimaTimeSteppers: SparseCoeffs, fused_increment, SparseContainer
using ClimaComms

@static isdefined(ClimaComms, :device_type) && ClimaComms.@import_required_backends
const device = ClimaComms.device()
const ArrayType = ClimaComms.array_type(device)

if device isa ClimaComms.CUDADevice
    import CUDA
    CUDA.allowscalar(false)
end

# Diagonal stand-in for the implicit Jacobian/`W` operator, so the implicit
# problems below can use device arrays (a dense matrix prototype would not run
# on the GPU). `NewtonsMethod` calls `zero(jac_prototype)` and `ldiv!(Δx, W, f)`.
struct MockDiagonalJacobian{A}
    diag::A
end
Base.zero(x::MockDiagonalJacobian) = MockDiagonalJacobian(zero(x.diag))
LinearAlgebra.ldiv!(x, A::MockDiagonalJacobian, b) = (x .= b ./ A.diag)

# Device-aware IMEX/Rosenbrock test problem: explicit forcing `0.1u`, implicit
# `-0.5u`, with a diagonal mock `W = dtγ J - I = (-0.5 dtγ - 1) I`.
function mock_implicit_prob(::Type{FT} = Float64; n = 3) where {FT}
    ODEProblem(
        ClimaODEFunction(;
            T_exp! = (du, u, p, t) -> (du .= FT(0.1) .* u),
            T_imp! = ODEFunction(
                (du, u, p, t) -> (du .= FT(-0.5) .* u);
                jac_prototype = MockDiagonalJacobian(ArrayType(zeros(FT, n))),
                Wfact = (W, u, p, dtγ, t) -> (W.diag .= FT(-0.5) * dtγ - FT(1)),
            ),
        ),
        ArrayType(ones(FT, n)),
        (FT(0), FT(1)),
        nothing,
    )
end

# Assert that one `step_u!` of `int` is free of runtime dispatch / type
# instability within ClimaTimeSteppers, after a warmup step to force compilation.
function test_step_u_inferred(int)
    CTS.step_u!(int, int.cache)  # warmup
    JET.@test_opt target_modules = (ClimaTimeSteppers,) CTS.step_u!(int, int.cache)
    CTS.step_u!(int, int.cache)
    @test all(isfinite, int.u)
end

@testset "Type stability" begin

    @testset "fused_increment" begin
        u = ArrayType([1.0, 2.0, 3.0])
        tend = ntuple(i -> u .* i, 3)
        coeffs = ones(3, 3)
        sc = SparseCoeffs(coeffs)
        dt = 0.5
        @inferred fused_increment(u, dt, sc, tend, Val(3))
    end

    @testset "SparseContainer" begin
        a1 = ArrayType(ones(3) .* 1)
        a2 = ArrayType(ones(3) .* 2)
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
        cache = CTS.allocate_cache(checker, ArrayType([1.0, 2.0]))
        @inferred CTS.is_converged!(
            checker,
            cache,
            ArrayType([1.0, 2.0]),
            ArrayType([0.01, 0.02]),
            1,
        )
    end

    @testset "Solver stepping (step_u! type stability)" begin

        @testset "Explicit RK (SSP33ShuOsher)" begin
            prob = ODEProblem(
                ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -0.5 .* u)),
                ArrayType([1.0, 2.0, 3.0]),
                (0.0, 1.0),
                nothing,
            )
            alg = ExplicitAlgorithm(SSP33ShuOsher())
            int = CTS.init(prob, alg; dt = 0.1, save_everystep = false)
            test_step_u_inferred(int)
        end

        @testset "LSRK (LSRK54CarpenterKennedy)" begin
            prob = ODEProblem(
                IncrementingODEFunction{true}(
                    (du, u, p, t, α = true, β = false) ->
                        (du .= α .* (-0.5) .* u .+ β .* du),
                ),
                ArrayType([1.0, 2.0, 3.0]),
                (0.0, 1.0),
                nothing,
            )
            int = CTS.init(
                prob,
                LSRK54CarpenterKennedy();
                dt = 0.1,
                save_everystep = false,
            )
            test_step_u_inferred(int)
        end

        @testset "IMEX ARK (ARS232)" begin
            prob = mock_implicit_prob()
            alg = CTS.IMEXAlgorithm(ARS232(), NewtonsMethod(; max_iters = 2))
            int = CTS.init(deepcopy(prob), alg; dt = 0.1, save_everystep = false)
            test_step_u_inferred(int)
        end

        @testset "IMEX SSPRK (SSP333)" begin
            prob = mock_implicit_prob()
            alg = CTS.IMEXAlgorithm(SSP333(), NewtonsMethod(; max_iters = 2))
            int = CTS.init(deepcopy(prob), alg; dt = 0.1, save_everystep = false)
            test_step_u_inferred(int)
        end

        @testset "Rosenbrock (SSPKnoth)" begin
            prob = mock_implicit_prob()
            alg = CTS.RosenbrockAlgorithm(ClimaTimeSteppers.tableau(SSPKnoth()))
            int = CTS.init(deepcopy(prob), alg; dt = 0.1, save_everystep = false)
            test_step_u_inferred(int)
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
                ArrayType([1.0, 2.0, 3.0]),
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
            test_step_u_inferred(int)
        end
    end

    @testset "Float32 solver stepping (step_u! type stability)" begin

        @testset "Explicit RK Float32" begin
            prob = ODEProblem(
                ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -0.5f0 .* u)),
                ArrayType(Float32[1.0, 2.0, 3.0]),
                (0.0f0, 1.0f0),
                nothing,
            )
            alg = ExplicitAlgorithm(SSP33ShuOsher())
            int = CTS.init(prob, alg; dt = 0.1f0, save_everystep = false)
            test_step_u_inferred(int)
            @test eltype(int.u) == Float32
        end

        @testset "LSRK Float32" begin
            prob = ODEProblem(
                IncrementingODEFunction{true}(
                    (du, u, p, t, α = true, β = false) ->
                        (du .= α .* (-0.5f0) .* u .+ β .* du),
                ),
                ArrayType(Float32[1.0, 2.0, 3.0]),
                (0.0f0, 1.0f0),
                nothing,
            )
            int = CTS.init(
                prob,
                LSRK54CarpenterKennedy();
                dt = 0.1f0,
                save_everystep = false,
            )
            test_step_u_inferred(int)
            @test eltype(int.u) == Float32
        end

        @testset "IMEX ARK Float32" begin
            prob = mock_implicit_prob(Float32)
            alg = CTS.IMEXAlgorithm(ARS232(), NewtonsMethod(; max_iters = 2))
            int = CTS.init(deepcopy(prob), alg; dt = 0.1f0, save_everystep = false)
            test_step_u_inferred(int)
            @test eltype(int.u) == Float32
        end
    end
end
