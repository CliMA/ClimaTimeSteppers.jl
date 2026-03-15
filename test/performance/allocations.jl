#=
Allocation tests: verify that stepping allocations stay within bounds.

These tests catch allocation regressions. Explicit methods should be
allocation-free. Implicit methods may allocate small amounts for the
linear solver, so we use upper bounds based on current behavior.
=#
using ClimaTimeSteppers, DiffEqBase, LinearAlgebra, Test
using ClimaComms
import ClimaTimeSteppers as CTS

@static isdefined(ClimaComms, :device_type) && ClimaComms.@import_required_backends
const device = ClimaComms.device()
const ArrayType = ClimaComms.array_type(device)

# Type piracy (confined to test code): Base.zero(::LU) is not defined in
# LinearAlgebra. NewtonsMethod calls `zero(jac_prototype)` to initialize the
# Jacobian cache, and we pass an LU factorization as the prototype for
# in-place solves. This extends Base.zero for that purpose.
Base.zero(x::LU) = lu(zero(x.factors); check = false)

# ============================================================================ #
# Test problems parameterized by float type FT
# ============================================================================ #

function make_split_prob_for_alloc_test(::Type{FT}) where {FT}
    n = 3
    Id = Matrix{FT}(I, n, n)
    ODEProblem(
        ClimaODEFunction(;
            T_exp! = (du, u, p, t) -> (du .= FT(0.1) .* u),
            T_imp! = DiffEqBase.ODEFunction(
                (du, u, p, t) -> (du .= FT(-0.5) .* u);
                jac_prototype = lu(zeros(FT, n, n), check = false),
                Wfact = (W, u, p, dtγ, t) -> begin
                    W.factors .= FT(-0.5) * dtγ .* Id .- Id
                    W.ipiv .= 1:n
                    # Trigger factorization in-place
                    LinearAlgebra.lu!(W.factors, NoPivot(); check = false)
                end,
            ),
        ),
        ones(FT, n),
        (FT(0), FT(1)),
        nothing,
    )
end

function make_explicit_prob_for_alloc_test(::Type{FT}) where {FT}
    ODEProblem(
        ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= FT(-0.5) .* u)),
        FT[1.0, 2.0, 3.0],
        (FT(0), FT(1)),
        nothing,
    )
end

function make_lsrk_prob_for_alloc_test(::Type{FT}) where {FT}
    ODEProblem(
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* FT(-0.5) .* u .+ β .* du),
        ),
        FT[1.0, 2.0, 3.0],
        (FT(0), FT(1)),
        nothing,
    )
end

function make_multirate_prob_for_alloc_test()
    SplitODEProblem(
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* (-5.0) .* u .+ β .* du),
        ),
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* (-0.5) .* u .+ β .* du),
        ),
        [1.0, 2.0, 3.0],
        (0.0, 1.0),
        nothing,
    )
end

"""
    test_step_allocations(alg, prob, dt)

Warm up with one step, then measure allocations on second step.
"""
function test_step_allocations(alg, prob, dt)
    hide_warning = (; kwargshandle = DiffEqBase.KeywordArgSilent)
    extra_kwargs = alg isa Multirate ? (; fast_dt = dt / 10) : (;)
    integrator = DiffEqBase.init(
        deepcopy(prob),
        alg;
        dt,
        save_everystep = false,
        extra_kwargs...,
        hide_warning...,
    )
    # Warmup step
    DiffEqBase.step!(integrator)
    # Measure allocations on second step
    allocs = @allocated DiffEqBase.step!(integrator)
    return allocs
end

@testset "Step allocations" begin

    for FT in (Float64, Float32)
        ft_name = FT == Float64 ? "" : " (Float32)"
        dt = FT(0.01)

        @testset "Explicit RK — zero allocations$ft_name" begin
            prob = make_explicit_prob_for_alloc_test(FT)
            for name in (SSP22Heuns(), SSP33ShuOsher(), RK4())
                alg = ExplicitAlgorithm(name)
                allocs = test_step_allocations(alg, prob, dt)
                @test allocs == 0
            end
        end

        @testset "LSRK — zero allocations$ft_name" begin
            prob = make_lsrk_prob_for_alloc_test(FT)
            for alg in
                (LSRKEulerMethod(), LSRK54CarpenterKennedy(), LSRK144NiegemannDiehlBusch())
                allocs = test_step_allocations(alg, prob, dt)
                @test allocs == 0
            end
        end

        @testset "IMEX ARK — bounded allocations$ft_name" begin
            prob = make_split_prob_for_alloc_test(FT)
            imex_algs = if FT == Float64
                (ARS111(), ARS232(), ARS343(), ARK437L2SA1(), ARK548L2SA2())
            else
                (ARS111(), ARS232())
            end
            for name in imex_algs
                alg = CTS.IMEXAlgorithm(name, NewtonsMethod(; max_iters = 2))
                allocs = test_step_allocations(alg, prob, dt)
                # High-order methods (ARK437, ARK548) need up to ~1200 bytes for
                # Newton solve workspace. Lower-order methods stay under 500.
                @test allocs ≤ 1200
            end
        end

        if FT == Float64
            @testset "IMEX SSPRK — bounded allocations" begin
                prob = make_split_prob_for_alloc_test(FT)
                for name in (SSP222(), SSP333())
                    alg = CTS.IMEXAlgorithm(name, NewtonsMethod(; max_iters = 2))
                    allocs = test_step_allocations(alg, prob, dt)
                    @test allocs ≤ 500
                end
            end

            @testset "Rosenbrock — bounded allocations" begin
                prob = make_split_prob_for_alloc_test(FT)
                alg = CTS.RosenbrockAlgorithm(ClimaTimeSteppers.tableau(SSPKnoth()))
                allocs = test_step_allocations(alg, prob, dt)
                @test allocs ≤ 550
            end

            @testset "Multirate — bounded allocations" begin
                prob = make_multirate_prob_for_alloc_test()
                for slow_alg in (LSRK54CarpenterKennedy(), MIS3C(), WSRK2())
                    alg = Multirate(LSRK54CarpenterKennedy(), slow_alg)
                    allocs = test_step_allocations(alg, prob, 0.1)
                    @test allocs < 10000
                end
            end
        end
    end

    @testset "Device (GPU) allocations CPU overhead" begin
        # Ensures that using device arrays (e.g. CuArray) does not introduce CPU
        # scalar indexing allocations. Measures CPU allocations only.
        prob_explicit = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -0.5 .* u)),
            ArrayType([1.0, 2.0, 3.0]),
            (0.0, 1.0),
            nothing,
        )
        for name in (SSP22Heuns(), RK4())
            alg = ExplicitAlgorithm(name)
            allocs = test_step_allocations(alg, prob_explicit, 0.01)
            # 0 allocations is ideal, but we allow up to 100 bytes for small device array wrappers if necessary
            @test allocs < 100
        end

        prob_lsrk = ODEProblem(
            IncrementingODEFunction{true}(
                (du, u, p, t, α = true, β = false) ->
                    (du .= α .* (-0.5) .* u .+ β .* du),
            ),
            ArrayType([1.0, 2.0, 3.0]),
            (0.0, 1.0),
            nothing,
        )
        for alg in (LSRKEulerMethod(), LSRK54CarpenterKennedy())
            allocs = test_step_allocations(alg, prob_lsrk, 0.01)
            @test allocs < 100
        end
    end
end
