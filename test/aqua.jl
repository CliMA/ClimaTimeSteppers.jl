using Aqua
using Test
using ClimaTimeSteppers

@testset "Aqua tests" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    Aqua.test_unbound_args(ClimaTimeSteppers)

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities
    @test isempty(Aqua.detect_ambiguities(ClimaTimeSteppers; recursive = true))
end
