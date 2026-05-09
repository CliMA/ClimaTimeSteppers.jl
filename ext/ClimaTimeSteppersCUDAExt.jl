module ClimaTimeSteppersCUDAExt

using CUDA
using ClimaComms
using ClimaTimeSteppers

# Global persistent handlers to avoid GC thrashing by recycling stream/event objects.
const _JAC_STREAM = Ref{Union{Nothing, CUDA.CuStream}}(nothing)
const _JAC_EVENT = Ref{Union{Nothing, CUDA.CuEvent}}(nothing)

function get_jac_resources()
    if isnothing(_JAC_STREAM[])
        _JAC_STREAM[] = CUDA.CuStream()
        _JAC_EVENT[] = CUDA.CuEvent()
    end
    return _JAC_STREAM[], _JAC_EVENT[]
end

"""
    async_Wfact!(T_imp!, jacobian, u, p, dtγ, t)

Specialized CUDA launcher that branches the matrix update workflow into an
isolated compute stream to permit concurrent execution with standard Stage 1
RHS evaluations on the GPU.
"""
function ClimaTimeSteppers.async_Wfact!(T_imp!, jacobian, u::CUDA.CuArray, p, dtγ, t)
    # Fork execution onto secondary compute stream if running on a GPU hardware backend
    if ClimaComms.device() isa ClimaComms.CUDADevice
        jac_stream, jac_event = get_jac_resources()

        # Execute Wfact! inside the alternate stream, parallel with default Stage 1 tendencies
        CUDA.stream!(jac_stream) do
            T_imp!.Wfact(jacobian, u, p, dtγ, t)
        end

        # Record non-blocking event so default stream knows when Jacobian has finished assembling
        CUDA.record(jac_event, jac_stream)
        return jac_event
    else
        # Synchronous CPU fallback fallback
        T_imp!.Wfact(jacobian, u, p, dtγ, t)
        return nothing
    end
end

"""
    sync_jacobian_update!(event::CUDA.CuEvent)

Native direct wait mechanism locking the compute stream pipeline until the Event
flags complete, enforcing absolute safety without halting the CPU thread.
"""
function ClimaTimeSteppers.sync_jacobian_update!(event::CUDA.CuEvent)
    CUDA.wait(event)
    return nothing
end

end # module
