module ClimaTimeSteppersCUDAExt

using CUDA
using ClimaComms
using ClimaTimeSteppers

# Global persistent handlers to avoid GC thrashing by recycling stream/event objects.
const _JAC_STREAM = Ref{Union{Nothing, CUDA.Stream}}(nothing)
const _JAC_EVENT = Ref{Union{Nothing, CUDA.Event}}(nothing)

function get_jac_resources()
    if isnothing(_JAC_STREAM[])
        _JAC_STREAM[] = CUDA.Stream(priority = :high)
        _JAC_EVENT[] = CUDA.Event(CUDA.EVENT_DISABLE_TIMING)
    end
    return _JAC_STREAM[], _JAC_EVENT[]
end

"""
    async_Wfact!(T_imp!, jacobian, u, p, dtγ, t)

Specialized CUDA launcher that branches the matrix update workflow into an
isolated compute stream to permit concurrent execution with standard Stage 1
RHS evaluations on the GPU.
"""
function ClimaTimeSteppers.async_Wfact!(T_imp!, jacobian, u, p, dtγ, t)
    # Validate hardware match. If running in mixed context, direct fallbacks are safe.
    if ClimaComms.device() isa ClimaComms.CUDADevice
        jac_stream, jac_event = get_jac_resources()

        # Route the call queue to the auxiliary stream.
        CUDA.stream!(jac_stream) do
            T_imp!.Wfact(jacobian, u, p, dtγ, t)
        end

        # Non-blocking GPU record marker tracking the progress of the async queue.
        CUDA.record(jac_event, jac_stream)
        return jac_event
    else
        # Fallback for active CPU contexts
        T_imp!.Wfact(jacobian, u, p, dtγ, t)
        return nothing
    end
end

"""
    sync_jacobian_update!(event::CUDA.Event)

Native direct wait mechanism locking the compute stream pipeline until the Event
flags complete, enforcing absolute safety without halting the CPU thread.
"""
function ClimaTimeSteppers.sync_jacobian_update!(event::CUDA.Event)
    # Inject GPU control node blocking active stream until event yields.
    CUDA.wait(event)
    return nothing
end

end # module
