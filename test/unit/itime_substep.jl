#=
julia --project=test
using Revise; include("test/unit/itime_substep.jl")
=#
using Test
import Dates
import ClimaTimeSteppers as CTS
import ClimaUtilities.TimeManager: ITime

@testset "ITime sub_timestep and refine_time" begin
    ns = Dates.Nanosecond(1)

    @testset "exact sub-step division" begin
        dt = ITime(1; period = Dates.Second(1))
        sub = CTS.sub_timestep(dt, 4)
        @test sub.period == ns
        @test sub.counter == 250_000_000
        @test 4 * float(sub) == float(dt)
    end

    @testset "non-divisible division throws" begin
        dt = ITime(1; period = Dates.Second(1))
        @test_throws ArgumentError CTS.sub_timestep(dt, 3)
    end

    @testset "refine_time to the nanosecond period" begin
        t = ITime(2; period = Dates.Second(1))
        r = CTS.refine_time(t)
        @test r.period == ns
        @test r.counter == 2_000_000_000
        @test float(r) == float(t)
    end

    @testset "epoch preservation" begin
        epoch = Dates.DateTime(2020, 1, 1)
        dt = ITime(1; period = Dates.Second(1), epoch = epoch)
        @test CTS.sub_timestep(dt, 4).epoch == epoch
        @test CTS.refine_time(dt).epoch == epoch
    end
end
