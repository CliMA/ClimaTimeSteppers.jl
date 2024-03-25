module CudaExt

import CUDA
import ClimaComms: SingletonCommsContext, CUDADevice
import ClimaTimeSteppers: compute_fj!

@inline function compute_fj!(f, j, U, f!, j!, ::SingletonCommsContext{CUDADevice})
    # TODO: we should benchmark these two options to
    #       see if one is preferrable over the other
    if Base.Threads.nthreads() > 1
        compute_fj_spawn!(f, j, U, f!, j!)
    else
        compute_fj_streams!(f, j, U, f!, j!)
    end
end

@inline function compute_fj_streams!(f, j, U, f!, j!)
    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(event, CUDA.stream()) # record event on main stream

    stream1 = CUDA.CuStream() # make a stream
    local event1
    CUDA.stream!(stream1) do # work to be done by stream1
        CUDA.wait(event, stream1) # make stream1 wait on event (host continues)
        f!(f, U)
        event1 = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    end
    CUDA.record(event1, stream1) # record event1 on stream1

    stream2 = CUDA.CuStream() # make a stream
    local event2
    CUDA.stream!(stream2) do # work to be done by stream2
        CUDA.wait(event, stream2) # make stream2 wait on event (host continues)
        j!(j, U)
        event2 = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    end
    CUDA.record(event2, stream2) # record event2 on stream2

    CUDA.wait(event1, CUDA.stream()) # make main stream wait on event1
    CUDA.wait(event2, CUDA.stream()) # make main stream wait on event2
end

@inline function compute_fj_spawn!(f, j, U, f!, j!)

    CUDA.synchronize()
    CUDA.@sync begin
        Base.Threads.@spawn begin
            f!(f, U)
            CUDA.synchronize()
            nothing
        end
        Base.Threads.@spawn begin
            j!(j, U)
            CUDA.synchronize()
            nothing
        end
    end
end

end
