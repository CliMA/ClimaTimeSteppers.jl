using Aqua
using Test
using ClimaTimeSteppers

@testset "Aqua tests" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393

    # Aqua.test_unbound_args(ClimaTimeSteppers)
    # Fails with:
    #=
      Expression: detect_unbound_args_recursively(m) == []
       Evaluated: Any[
       ClimaTimeSteppers.WickerSkamarockRungeKuttaTableau(c::Tuple{Vararg{RT, Nstages}}) where {Nstages, RT}
            in ClimaTimeSteppers at ClimaTimeSteppers.jl/ClimaTimeSteppers.jl/src/solvers/wickerskamarock.jl:18,
       ClimaTimeSteppers.LowStorageRungeKutta2NTableau(A::Tuple{Vararg{RT, Nstages}}, B::Tuple{Vararg{RT, Nstages}}, C::Tuple{Vararg{RT, Nstages}}) where {Nstages, RT}
            in ClimaTimeSteppers at ClimaTimeSteppers.jl/ClimaTimeSteppers.jl/src/solvers/lsrk.jl:23,
       ClimaTimeSteppers.MultirateInfinitesimalStepCache(ΔU::Tuple{Vararg{A, Nstages}}, F::Tuple{Vararg{A, Nstages}}, tableau::T) where {Nstages, A, T<:ClimaTimeSteppers.MultirateInfinitesimalStepTableau}
            in ClimaTimeSteppers at ClimaTimeSteppers.jl/ClimaTimeSteppers.jl/src/solvers/mis.jl:45,
       ClimaTimeSteppers.StrongStabilityPreservingRungeKuttaTableau(A1::Tuple{Vararg{RT, Nstages}}, A2::Tuple{Vararg{RT, Nstages}}, B::Tuple{Vararg{RT, Nstages}}, C::Tuple{Vararg{RT, Nstages}}) where {Nstages, RT}
            in ClimaTimeSteppers at ClimaTimeSteppers.jl/ClimaTimeSteppers.jl/src/solvers/ssprk.jl:19] == Any[]
    =#

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities
    @test isempty(Aqua.detect_ambiguities(ClimaTimeSteppers; recursive = true))

    # If the number of unbound args is less than the limit below,
    # then please lower the limit. We're trying to drive this number
    # down to zero.
    ua = Aqua.detect_unbound_args_recursively(ClimaTimeSteppers)
    @test length(ua) ≤ 5
    @test_broken length(ua) ≠ 5
end
