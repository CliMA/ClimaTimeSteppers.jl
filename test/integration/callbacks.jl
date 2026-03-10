using Test
using ClimaComms
using ClimaTimeSteppers, DiffEqBase
using ClimaTimeSteppers.Callbacks
import MPI
@static isdefined(ClimaComms, :device_type) && ClimaComms.@import_required_backends
device = ClimaComms.device()
comm_ctx = ClimaComms.context(device)

ClimaComms.init(comm_ctx)

"""
Tests the `CallbackSet` and individual `EveryX` callback triggers.
Verifies that initialization, finalization, and periodic triggering occur correctly
based on simulation time, simulation steps, and real wall-clock time.
"""

"""
Mock callback to record initialization, finalization, and the number of triggers.
"""
mutable struct MyCallback
    initialized::Bool
    calls::Int
    finalized::Bool
end
MyCallback() = MyCallback(false, 0, false)

function Callbacks.initialize!(cb::MyCallback, integrator)
    cb.initialized = true
end
function Callbacks.finalize!(cb::MyCallback, integrator)
    cb.finalized = true
end
function (cb::MyCallback)(integrator)
    cb.calls += 1
end

cb1 = MyCallback()
cb2 = MyCallback()
cb3 = MyCallback()
cb4 = MyCallback()
cb5 = MyCallback()

cbs = CallbackSet(
    EveryXSimulationTime(cb1, 1 / 4),
    EveryXSimulationTime(cb2, 1 / 2, atinit = true),
    EveryXSimulationSteps(cb3, 1),
    EveryXSimulationSteps(cb4, 4, atinit = true),
    EveryXSimulationSteps(_ -> begin
            # Simulate uneven computational load across MPI ranks to test 
            # the distributed robustness of EveryXWallTimeSeconds. 
            # The root rank checks wall time and broadcasts, ensuring all ranks 
            # fire the callback at the same step despite different loads.
            if ClimaComms.iamroot(comm_ctx)
                sleep(1 / 32)
            else
                sleep(1 / 64)
            end
        end, 1),
    EveryXWallTimeSeconds(cb5, 0.49, comm_ctx),
)

const_prob_inc = ODEProblem(
    IncrementingODEFunction{true}(
        (du, u, p, t, α = true, β = false) -> (du .= α .* p .+ β .* du),
    ),
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

# Total simulation time is 1.0, with dt = 1/32 (32 steps)

# cb1: Every 0.25s of simulation time (no atinit). Triggers at t = 0.25, 0.50, 0.75, 1.00 -> 4 calls.
@test cb1.calls == 4
# cb2: Every 0.50s of simulation time (with atinit). Triggers at t = 0.00, 0.50, 1.00 -> 3 calls.
@test cb2.calls == 3
# cb3: Every 1 simulation step. Triggers at steps 1 through 32 -> 32 calls.
@test cb3.calls == 32
# cb4: Every 4 simulation steps (with atinit). Triggers at steps 0, 4, 8, 12, 16, 20, 24, 28, 32 -> 9 calls.
@test cb4.calls == 9
# cb5: Every 0.49s of wall time. Root rank sleeps for 1/32s, so 32 steps take ~1.0 real seconds.
# This should trigger at ~0.49s and ~0.98s, leading to at least 2 calls.
# Due to the internal broadcast in EveryXWallTimeSeconds, all ranks will fire the callback
# exactly the same number of times even if non-root ranks slept for half as long.
@test cb5.calls >= 2

if isdefined(DiffEqBase, :finalize!)

    @test cb1.finalized
    @test cb2.finalized
    @test cb3.finalized
    @test cb4.finalized
    @test cb5.finalized
end
