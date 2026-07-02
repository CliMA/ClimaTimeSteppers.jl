#=
The implicit-stage hook `initialize_imp!` (a public field of `ClimaODEFunction`,
"called once per implicit stage to set up the Newton solve") must be honored by
BOTH the IMEX-ARK and the IMEX-SSPRK stepping paths.
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import ClimaTimeSteppers: ODEProblem, ODEFunction

@testset "initialize_imp! is honored on both IMEX paths" begin
    n = 3
    Id = Matrix{Float64}(I, n, n)
    # T_imp!(du,u,p,t) = -u  ⟹  J = -I  ⟹  W = dtγ J - I = -(dtγ + 1) I
    function make_prob(calls)
        ODEProblem(
            ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du .= 0.1 .* u),
                T_imp! = ODEFunction(
                    (du, u, p, t) -> (du .= -u);
                    jac_prototype = zeros(n, n),
                    Wfact = (W, u, p, dtγ, t) -> (W .= -dtγ .* Id .- Id),
                ),
                # Count the calls; does not mutate the guess, so the solve still
                # converges to the same answer.
                initialize_imp! = (u, p, γdt) -> (calls[] += 1),
            ),
            ones(n),
            (0.0, 1.0),
            nothing,
        )
    end

    # SSP333 exercises the SSPRK path; ARS343 exercises the ARK path.
    for name in (SSP333(), ARS343())
        calls = Ref(0)
        prob = make_prob(calls)
        alg = CTS.IMEXAlgorithm(name, NewtonsMethod(; max_iters = 2))
        integrator = CTS.init(prob, alg; dt = 0.1)
        CTS.step!(integrator)
        # Called once per implicit stage; 0 means the hook was ignored.
        @test calls[] > 0
        @test all(isfinite, integrator.u)
    end
end
