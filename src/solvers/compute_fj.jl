@inline function compute_fj!(f, j, U, f!, j!, ::Union{Nothing, ClimaComms.AbstractCommsContext})
    f!(f, U)
    j!(j, U)
end

@inline function compute_fj!(f, j, U, f!, j!, ::ClimaComms.SingletonCommsContext{ClimaComms.CPUMultiThreaded})
    Base.@sync begin
        Base.Threads.@spawn begin
            f!(f, U)
            nothing
        end
        Base.Threads.@spawn begin
            j!(j, U)
            nothing
        end
    end
end
