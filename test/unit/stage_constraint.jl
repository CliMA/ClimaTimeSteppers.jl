#=
With `update_constrain_state = UpdateEvery(EndOfStage)` (or `UpdateEvery(WithDSS)`),
`constrain_state!` also fires on the Newton-solved value of each implicit stage.
That constrained value is what `cache!` and the explicit tendencies see, but the
constraint increment must NOT be recorded in the implicit tendency
`T_imp[i] = (U - temp) / dtγ`: otherwise it is re-applied in later stages and in
the step update with weights `a_imp[j, i] / γ` and `b_imp[i] / γ`, and a clip can
overshoot the bound it enforces. Both the IMEX-ARK and the IMEX-SSPRK paths share
the stage solve, so both are tested.
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import ClimaTimeSteppers:
    ODEProblem, ODEFunction, UpdateEvery, EndOfStep, EndOfStage, WithDSS

const n = 2
const Id = Matrix{Float64}(I, n, n)

# T_imp!(du, u, p, t) = c (constant)  ⟹  J = 0  ⟹  W = dtγ J - I = -I, so a single
# Newton iteration solves the implicit stage equation exactly.
constant_imp(c) = ODEFunction(
    (du, u, p, t) -> (du .= c);
    jac_prototype = zeros(n, n),
    Wfact = (W, u, p, dtγ, t) -> (W .= .-Id),
)

clip_u1!(u, p, t) = (u[1] = max(u[1], 0); nothing)

function step_once(
    name,
    update_constrain_state;
    u0,
    T_exp! = (du, u, p, t) -> (du .= 0),
    T_imp! = constant_imp(0),
    T_post_imp! = nothing,
    constrain_state! = clip_u1!,
    initialize_imp! = nothing,
    dt = 1.0,
    nsteps = 1,
    max_iters = 1,
)
    f = ClimaODEFunction(;
        T_exp!,
        T_imp!,
        T_post_imp!,
        constrain_state!,
        initialize_imp!,
        update_constrain_state,
    )
    prob = ODEProblem(f, copy(u0), (0.0, nsteps * dt), nothing)
    alg = CTS.IMEXAlgorithm(name, NewtonsMethod(; max_iters))
    integrator = CTS.init(prob, alg; dt)
    for _ in 1:nsteps
        CTS.step!(integrator)
    end
    return copy(integrator.u)
end

# ARS222 and ARS232 are FSAL (the last-stage constraint is skipped), ARS343 is not
# and has a negative weight `b_imp[3] / γ`; SSP222 and SSP333 exercise the SSPRK
# path (SSP222 also fails before the fix with `WithDSS`).
const algs = (ARS222(), ARS232(), ARS343(), SSP222(), SSP333())
const handlers = (UpdateEvery(EndOfStep), UpdateEvery(EndOfStage), UpdateEvery(WithDSS))

@testset "stage constraint is not integrated as an implicit tendency" begin
    # T ≡ 0 and u0[1] < 0: the state does not change, so the only thing that can
    # move it is the clip, which must give max(u0[1], 0) = 0 and no more. Before
    # the fix, `EndOfStage` gave u[1] = 1.41 (ARS222, ARS232), 1.25 (ARS343) and
    # 0.42 (SSP333).
    for name in algs, handler in handlers
        @test step_once(name, handler; u0 = [-1.0, 0.5]) == [0.0, 0.5]
    end
end

@testset "stage constraint does not change later implicit-stage inputs" begin
    # u1' = -1 (implicit), from u1 = 0.1 over dt = 1: stage values cross zero
    # and are clipped. `initialize_imp!` sees the (unconstrained) input of every
    # implicit stage; these must be the same as without stage constraints.
    T_imp! = constant_imp(-1)
    for name in algs
        inputs = map((UpdateEvery(EndOfStep), UpdateEvery(EndOfStage))) do handler
            record = Float64[]
            initialize_imp! = (U, p, dtγ) -> (push!(record, U[1]); nothing)
            step_once(name, handler; u0 = [0.1, 0.0], T_imp!, initialize_imp!)
            record
        end
        @test length(inputs[1]) == length(inputs[2]) > 0
        @test all(isapprox.(inputs[1], inputs[2]; atol = 10eps()))
        # With `WithDSS`, the stage input is also clipped before the stage solve
        # (so before `temp` is saved); with T_imp[j] = -1 exactly for all j, the
        # recorded inputs are the clipped `EndOfStep` inputs.
        record = Float64[]
        initialize_imp! = (U, p, dtγ) -> (push!(record, U[1]); nothing)
        step_once(name, UpdateEvery(WithDSS); u0 = [0.1, 0.0], T_imp!, initialize_imp!)
        @test length(record) == length(inputs[1])
        @test all(isapprox.(record, max.(inputs[1], 0); atol = 10eps()))
    end
end

@testset "stage constraint is still seen by the explicit tendencies" begin
    # u2' = -u1 (explicit), u1' = -1 (implicit): clipping the stage values of u1
    # changes u2. The end-of-step clip enforces u1 ≥ 0 in both modes.
    T_exp! = (du, u, p, t) -> (du[1] = 0; du[2] = -u[1])
    T_imp! = constant_imp(-1)
    for name in algs
        kwargs = (; u0 = [0.1, 1.0], T_exp!, T_imp!)
        u_step = step_once(name, UpdateEvery(EndOfStep); kwargs...)
        u_stage = step_once(name, UpdateEvery(EndOfStage); kwargs...)
        @test u_step[1] == u_stage[1] == 0
        @test u_step[2] != u_stage[2]
    end
end

@testset "inactive stage constraint leaves the solution unchanged" begin
    # Nonlinear tendencies plus `T_post_imp!`, with a constraint that is called
    # but never changes the state: bitwise identical to no stage constraint.
    T_exp! = (du, u, p, t) -> (du .= 0.3 .* sin.(u) .+ t)
    T_imp! = ODEFunction(
        (du, u, p, t) -> (du .= -2 .* u .- 0.1 .* u .^ 3);
        jac_prototype = zeros(n, n),
        Wfact = (W, u, p, dtγ, t) -> (W .= dtγ .* Diagonal(-2 .- 0.3 .* u .^ 2) .- Id),
    )
    T_post_imp! = (du, u, p, t) -> (du .= 0.05 .* u)
    calls = Ref(0)
    count_calls! = (u, p, t) -> (calls[] += 1; nothing)
    for name in algs, max_iters in (1, 2)
        kwargs = (;
            u0 = [0.7, -0.4],
            T_exp!,
            T_imp!,
            T_post_imp!,
            dt = 0.2,
            nsteps = 5,
            max_iters,
        )
        u_ref = step_once(
            name,
            UpdateEvery(EndOfStep);
            kwargs...,
            constrain_state! = Returns(nothing),
        )
        for handler in (UpdateEvery(EndOfStage), UpdateEvery(WithDSS))
            @test step_once(name, handler; kwargs..., constrain_state! = count_calls!) ==
                  u_ref
        end
    end
    @test calls[] > 0
end
