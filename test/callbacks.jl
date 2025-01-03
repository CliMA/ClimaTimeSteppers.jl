using Test
using ClimaComms
using ClimaTimeSteppers, DiffEqBase
using ClimaTimeSteppers.Callbacks
# import MPI
@static isdefined(ClimaComms, :device_type) && ClimaComms.@import_required_backends
device = ClimaComms.device()
comm_ctx = ClimaComms.context(device)

ClimaComms.init(comm_ctx)


mutable struct MyCallback
    initialized::Bool
    calls::Int
    finalized::Bool
    last_t::Real
end
MyCallback() = MyCallback(false, 0, false, -1.0)

function Callbacks.initialize!(cb::MyCallback, integrator)
    cb.initialized = true
end
function Callbacks.finalize!(cb::MyCallback, integrator)
    cb.finalized = true
end
function (cb::MyCallback)(integrator)
    cb.calls += 1
    cb.last_t = integrator.t
end

cb1 = MyCallback()
cb2 = MyCallback()
cb3 = MyCallback()
cb4 = MyCallback()
cb5 = MyCallback()
cb6 = MyCallback()
cb7 = MyCallback()
cb8 = MyCallback()
cb9 = MyCallback()
cb10 = MyCallback()
cb11 = MyCallback()

cbs = CallbackSet(
    EveryXSimulationTime(cb1, 1 / 4),
    EveryXSimulationTime(cb2, 1 / 2, atinit = true),
    EveryXSimulationSteps(cb3, 1),
    EveryXSimulationSteps(cb4, 4, atinit = true),
    EveryXSimulationSteps(_ -> sleep(1 / 32), 1),
    EveryXWallTimeSeconds(cb5, 0.49, comm_ctx),
    EveryXSimulationTime(cb6, 0.49, call_at_end = true),
    EveryXSimulationSteps(cb7, 3, call_at_end = true),
    EveryXSimulationTime(cb8, 0.3, call_at_end = false),
    EveryXSimulationSteps(cb9, 3, call_at_end = false),
)

@test_throws AssertionError EveryXSimulationTime(cb10, Inf)
@test_throws AssertionError EveryXSimulationSteps(cb11, Inf)

const_prob_inc = ODEProblem(
    IncrementingODEFunction{true}((du, u, p, t, α = true, β = false) -> (du .= α .* p .+ β .* du)),
    [0.0],
    (0.0, 1.0),
    2.0,
)

solve(const_prob_inc, LSRKEulerMethod(), dt = 1 / 32, callback = cbs)

@test cb1.initialized
@test cb2.initialized
@test cb3.initialized
@test cb4.initialized
@test cb5.initialized

@test cb1.calls == 4
@test cb2.calls == 3
@test cb3.calls == 32
@test cb4.calls == 9
@test cb5.calls >= 2

@test cb6.last_t == 1.0
@test cb7.last_t == 1.0
@test cb8.last_t == (1 / 32) * 29
@test cb9.last_t == (1 / 32) * 30

if isdefined(DiffEqBase, :finalize!)

    @test cb1.finalized
    @test cb2.finalized
    @test cb3.finalized
    @test cb4.finalized
    @test cb5.finalized
end
