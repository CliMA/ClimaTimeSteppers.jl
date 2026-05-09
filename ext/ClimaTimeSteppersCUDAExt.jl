module ClimaTimeSteppersCUDAExt

using CUDA
using ClimaTimeSteppers

function ClimaTimeSteppers.init_jac_resources(u::CUDA.CuArray)
    return (CUDA.CuStream(), CUDA.CuEvent())
end

function ClimaTimeSteppers.async_Wfact!(
    T_imp!,
    jacobian,
    u::CUDA.CuArray,
    p,
    dtγ,
    t,
    jac_resources::Tuple{CUDA.CuStream, CUDA.CuEvent},
)
    jac_stream, jac_event = jac_resources
    CUDA.stream!(jac_stream) do
        T_imp!.Wfact(jacobian, u, p, dtγ, t)
    end
    CUDA.record(jac_event, jac_stream)
    return jac_event
end

function ClimaTimeSteppers.sync_jacobian_update!(event::CUDA.CuEvent)
    CUDA.wait(event)
    return nothing
end

end # module
