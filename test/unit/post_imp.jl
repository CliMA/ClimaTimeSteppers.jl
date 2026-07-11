#=
The optional post-Newton hook `T_post_imp!` (a public field of `ClimaODEFunction`)
is evaluated at the Newton-solved stage state `U*` and applied as
`U ← U* + dtγ · dY_post`. It must be honored by BOTH the IMEX-ARK and the
IMEX-SSPRK stepping paths, and must NOT be called when the field is `nothing`.
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import ClimaTimeSteppers: ODEProblem, ODEFunction

@testset "T_post_imp! is honored on both IMEX paths" begin
    n = 3
    Id = Matrix{Float64}(I, n, n)
    # T_imp!(du,u,p,t) = -u  ⟹  J = -I  ⟹  W = dtγ J - I = -(dtγ + 1) I
    # T_post_imp!(du,u,p,t) = c · u — small linear correction, counted per call.
    function make_prob(calls; use_post = true, c = 0.05)
        T_post_imp! = use_post ? ((du, u, p, t) -> (du .= c .* u; calls[] += 1)) : nothing
        ODEProblem(
            ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du .= 0.1 .* u),
                T_imp! = ODEFunction(
                    (du, u, p, t) -> (du .= -u);
                    jac_prototype = zeros(n, n),
                    Wfact = (W, u, p, dtγ, t) -> (W .= -dtγ .* Id .- Id),
                ),
                T_post_imp!,
            ),
            ones(n),
            (0.0, 1.0),
            nothing,
        )
    end

    # SSP333 exercises the SSPRK path; ARS343 exercises the ARK path.
    for name in (SSP333(), ARS343())
        # With `T_post_imp!`: hook is called; solution stays finite.
        calls = Ref(0)
        prob = make_prob(calls; use_post = true)
        alg = CTS.IMEXAlgorithm(name, NewtonsMethod(; max_iters = 2))
        integrator = CTS.init(prob, alg; dt = 0.1)
        CTS.step!(integrator)
        @test calls[] > 0                     # hook fired at implicit stages
        @test all(isfinite, integrator.u)

        # Without `T_post_imp!`: hook must NOT fire (default is `nothing`).
        calls_off = Ref(0)
        prob_off = make_prob(calls_off; use_post = false)
        integrator_off = CTS.init(prob_off, alg; dt = 0.1)
        CTS.step!(integrator_off)
        @test calls_off[] == 0

        # With a nonzero correction the result must differ from the no-post run.
        @test integrator.u != integrator_off.u

        # Quantitative sanity: with a zero-coefficient `T_post_imp!` the hook
        # writes du = 0, so `U ← U* + dtγ · 0 = U*` at every stage and the
        # result must be bitwise-identical to the `use_post = false` run.
        calls_zero = Ref(0)
        prob_zero = make_prob(calls_zero; use_post = true, c = 0.0)
        integrator_zero = CTS.init(prob_zero, alg; dt = 0.1)
        CTS.step!(integrator_zero)
        @test calls_zero[] > 0                # hook still fires
        @test integrator_zero.u == integrator_off.u
    end
end
