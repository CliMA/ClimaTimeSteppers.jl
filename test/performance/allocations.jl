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

# ============================================================================ #
# Test problems
# ============================================================================ #

function make_split_prob_for_alloc_test()
    n = 3
    Id = Matrix{Float64}(I, n, n)
    ODEProblem(
        ClimaODEFunction(;
            T_exp! = (du, u, p, t) -> (du .= 0.1 .* u),
            T_imp! = DiffEqBase.ODEFunction(
                (du, u, p, t) -> (du .= -0.5 .* u);
                jac_prototype = zeros(n, n),
                Wfact = (W, u, p, γ, t) -> (W .= -0.5 * γ .* Id .- Id),
            ),
        ),
        ones(n),
        (0.0, 1.0),
        nothing,
    )
end

function make_explicit_prob_for_alloc_test()
    ODEProblem(
        ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -0.5 .* u)),
        [1.0, 2.0, 3.0],
        (0.0, 1.0),
        nothing,
    )
end

function make_lsrk_prob_for_alloc_test()
    ODEProblem(
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* (-0.5) .* u .+ β .* du),
        ),
        [1.0, 2.0, 3.0],
        (0.0, 1.0),
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

    @testset "Explicit RK — zero allocations" begin
        prob = make_explicit_prob_for_alloc_test()
        for name in (SSP22Heuns(), SSP33ShuOsher(), RK4())
            alg = ExplicitAlgorithm(name)
            allocs = test_step_allocations(alg, prob, 0.01)
            @test allocs == 0
        end
    end

    @testset "LSRK — zero allocations" begin
        prob = make_lsrk_prob_for_alloc_test()
        for alg in
            (LSRKEulerMethod(), LSRK54CarpenterKennedy(), LSRK144NiegemannDiehlBusch())
            allocs = test_step_allocations(alg, prob, 0.01)
            @test allocs == 0
        end
    end

    @testset "IMEX ARK — bounded allocations" begin
        prob = make_split_prob_for_alloc_test()
        for name in (ARS111(), ARS343(), ARS232())
            alg = CTS.IMEXAlgorithm(name, NewtonsMethod(; max_iters = 2))
            allocs = test_step_allocations(alg, prob, 0.01)
            # Implicit methods allocate for LU factorization; bound at 5KB
            @test allocs < 5000
        end
    end

    @testset "IMEX SSPRK — bounded allocations" begin
        prob = make_split_prob_for_alloc_test()
        for name in (SSP222(), SSP333())
            alg = CTS.IMEXAlgorithm(name, NewtonsMethod(; max_iters = 2))
            allocs = test_step_allocations(alg, prob, 0.01)
            @test allocs < 5000
        end
    end

    @testset "Rosenbrock — bounded allocations" begin
        prob = make_split_prob_for_alloc_test()
        alg = CTS.RosenbrockAlgorithm(ClimaTimeSteppers.tableau(SSPKnoth()))
        allocs = test_step_allocations(alg, prob, 0.01)
        @test allocs < 5000
    end

    @testset "Multirate — bounded allocations" begin
        prob = make_multirate_prob_for_alloc_test()
        for slow_alg in (LSRK54CarpenterKennedy(), MIS3C(), WSRK2())
            alg = Multirate(LSRK54CarpenterKennedy(), slow_alg)
            allocs = test_step_allocations(alg, prob, 0.1)
            @test allocs < 10000
        end
    end

    # Float32 allocation tests: verify allocation behavior matches Float64
    @testset "Float32 — Explicit RK zero allocations" begin
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -0.5f0 .* u)),
            Float32[1.0, 2.0, 3.0],
            (0.0f0, 1.0f0),
            nothing,
        )
        for name in (SSP22Heuns(), SSP33ShuOsher(), RK4())
            alg = ExplicitAlgorithm(name)
            allocs = test_step_allocations(alg, prob, 0.01f0)
            @test allocs == 0
        end
    end

    @testset "Float32 — LSRK zero allocations" begin
        prob = ODEProblem(
            IncrementingODEFunction{true}(
                (du, u, p, t, α = true, β = false) ->
                    (du .= α .* (-0.5f0) .* u .+ β .* du),
            ),
            Float32[1.0, 2.0, 3.0],
            (0.0f0, 1.0f0),
            nothing,
        )
        for alg in
            (LSRKEulerMethod(), LSRK54CarpenterKennedy(), LSRK144NiegemannDiehlBusch())
            allocs = test_step_allocations(alg, prob, 0.01f0)
            @test allocs == 0
        end
    end

    @testset "Float32 — IMEX ARK bounded allocations" begin
        n = 3
        Id = Matrix{Float32}(I, n, n)
        prob = ODEProblem(
            ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du .= 0.1f0 .* u),
                T_imp! = DiffEqBase.ODEFunction(
                    (du, u, p, t) -> (du .= -0.5f0 .* u);
                    jac_prototype = zeros(Float32, n, n),
                    Wfact = (W, u, p, γ, t) -> (W .= Float32(-0.5) * γ .* Id .- Id),
                ),
            ),
            ones(Float32, n),
            (0.0f0, 1.0f0),
            nothing,
        )
        for name in (ARS111(), ARS232())
            alg = CTS.IMEXAlgorithm(name, NewtonsMethod(; max_iters = 2))
            allocs = test_step_allocations(alg, prob, 0.01f0)
            @test allocs < 5000
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
