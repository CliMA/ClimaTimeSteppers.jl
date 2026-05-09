using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import CUDA

@testset "Async Jacobian dispatch (CPU fallback)" begin
    # On CPU, `init_jac_resources` returns nothing, `async_Wfact!` runs Wfact
    # synchronously and returns no token, and `sync_jacobian_update!(nothing)`
    # is a no-op.
    n = 3
    u = ones(n)
    p = nothing
    jacobian = zeros(n, n)
    dtγ = 0.5
    t = 1.0
    Wfact_called = Ref(false)

    f = CTS.ODEFunction(
        (du, u, p, t) -> (du .= -u);
        jac_prototype = jacobian,
        Wfact = (W, u, p, dtγ, t) -> begin
            Wfact_called[] = true
            W .= dtγ
        end,
    )

    @test CTS.init_jac_resources(u) === nothing

    token = CTS.async_Wfact!(f, jacobian, u, p, dtγ, t, nothing)
    @test token === nothing
    @test Wfact_called[]
    @test all(jacobian .== dtγ)

    @test CTS.sync_jacobian_update!(nothing) === nothing
end

@testset "claim_jacobian_update! gating" begin
    # Verifies that the function consumes the update-schedule signal each call,
    # so it cannot be used for inspection without corrupting the schedule.
    jacobian = zeros(2, 2)
    cache = (j = jacobian,)
    f = CTS.ODEFunction(
        (du, u, p, t) -> (du .= -u);
        jac_prototype = jacobian,
        Wfact = (W, u, p, dtγ, t) -> nothing,
    )

    # Handler that always fires on NewTimeStep: every call returns the buffer.
    nm_always = NewtonsMethod(; update_j = UpdateEvery(NewTimeStep))
    @test CTS.claim_jacobian_update!(f, nm_always, cache, 1.0) === jacobian
    @test CTS.claim_jacobian_update!(f, nm_always, cache, 1.0) === jacobian
    @test CTS.claim_jacobian_update!(f, nm_always, cache, 2.0) === jacobian

    # Handler that never fires on NewTimeStep: always returns nothing.
    nm_silent = NewtonsMethod(; update_j = UpdateEvery(NewNewtonSolve))
    @test CTS.claim_jacobian_update!(f, nm_silent, cache, 1.0) === nothing
    @test CTS.claim_jacobian_update!(f, nm_silent, cache, 2.0) === nothing

    # UpdateEveryN(2, NewTimeStep): fires every other call. A second call
    # within the same step "burns" the schedule and the next legitimate call
    # is skipped — the hazard the function name warns about.
    nm_alternating = NewtonsMethod(; update_j = UpdateEveryN(2, NewTimeStep))
    @test CTS.claim_jacobian_update!(f, nm_alternating, cache, 1.0) === jacobian
    @test CTS.claim_jacobian_update!(f, nm_alternating, cache, 1.0) === nothing
    @test CTS.claim_jacobian_update!(f, nm_alternating, cache, 2.0) === jacobian

    # Guard branches: any of T_imp!, newtons_method, or cache.j being nothing
    # short-circuits to nothing without consuming a signal.
    @test CTS.claim_jacobian_update!(nothing, nm_always, cache, 1.0) === nothing
    @test CTS.claim_jacobian_update!(f, nothing, cache, 1.0) === nothing
    @test CTS.claim_jacobian_update!(f, nm_always, (j = nothing,), 1.0) === nothing
end

@testset "Async Jacobian dispatch (CUDA)" begin
    if !CUDA.functional()
        @info "Skipping CUDA async Jacobian test: no functional CUDA device"
        return
    end

    n = 3
    u = CUDA.ones(Float64, n)
    p = nothing
    jacobian = CUDA.zeros(Float64, n, n)
    dtγ = 0.5
    t = 1.0

    f = CTS.ODEFunction(
        (du, u, p, t) -> (du .= -u);
        jac_prototype = jacobian,
        Wfact = (W, u, p, dtγ, t) -> (W .= dtγ),
    )

    resources = CTS.init_jac_resources(u)
    @test resources isa Tuple{CUDA.CuStream, CUDA.CuEvent}

    token = CTS.async_Wfact!(f, jacobian, u, p, dtγ, t, resources)
    @test token isa CUDA.CuEvent
    CTS.sync_jacobian_update!(token)
    CUDA.synchronize()
    @test all(Array(jacobian) .== dtγ)
end
