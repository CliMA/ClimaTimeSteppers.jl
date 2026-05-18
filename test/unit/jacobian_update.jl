using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS

@testset "maybe_update_jacobian!" begin
    n = 3
    jacobian = zeros(n, n)
    u = ones(n)
    p = nothing
    dtγ = 0.5
    t = 1.0
    dt = 1.0
    Wfact_called = Ref(false)

    f = CTS.ODEFunction(
        (du, u, p, t) -> (du .= -u);
        jac_prototype = jacobian,
        Wfact = (W, u, p, dtγ, t) -> begin
            Wfact_called[] = true
            W .= dtγ
        end,
    )

    # UpdateEvery(NewTimeStep): fires on every call with a new time.
    nm = NewtonsMethod(; max_iters = 2, update_j = UpdateEvery(NewTimeStep))
    cache = (j = jacobian, )

    CTS.maybe_update_jacobian!(f, nm, cache, u, p, t, dt, 0.5, CTS.IMEXAlgorithm(ARS111(), nm))
    @test Wfact_called[]
    @test all(jacobian .== dt * 0.5)

    # Guard: no-op when T_imp! is nothing.
    Wfact_called[] = false
    CTS.maybe_update_jacobian!(nothing, nm, cache, u, p, t, dt, 0.5, CTS.IMEXAlgorithm(ARS111(), nm))
    @test !Wfact_called[]

    # Guard: no-op when newtons_method is nothing.
    CTS.maybe_update_jacobian!(f, nothing, cache, u, p, t, dt, 0.5, CTS.IMEXAlgorithm(ARS111(), nm))
    @test !Wfact_called[]

    # Guard: no-op when jacobian buffer is nothing.
    CTS.maybe_update_jacobian!(f, nm, (j = nothing,), u, p, t, dt, 0.5, CTS.IMEXAlgorithm(ARS111(), nm))
    @test !Wfact_called[]
end
