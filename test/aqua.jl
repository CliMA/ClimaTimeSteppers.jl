using Aqua
using Test

@testset "Aqua tests - unbound args" begin
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
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(ClimaTimeSteppers; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("ClimaTimeSteppers", pkgdir(last(x).module)), ambs)
    for method_ambiguity in ambs
        @show method_ambiguity
    end
    # If the number of ambiguities is less than the limit below,
    # then please lower the limit based on the new number of ambiguities.
    # We're trying to drive this number down to zero to reduce latency.
    @info "Number of method ambiguities: $(length(ambs))"
    @test length(ambs) ≤ 100

    # returns a vector of all unbound args
    # ua = Aqua.detect_unbound_args_recursively(ClimaCore)
end
