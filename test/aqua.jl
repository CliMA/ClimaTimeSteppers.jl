using Test
using ClimaTimeSteppers
using Aqua

@testset "Aqua tests (performance)" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    # Aqua.test_unbound_args(ClimaTimeSteppers)
    ua = Aqua.detect_unbound_args_recursively(ClimaTimeSteppers)
    @test length(ua) == 0

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(ClimaTimeSteppers; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("ClimaTimeSteppers", pkgdir(last(x).module)), ambs)

    # Uncomment for debugging:
    # for method_ambiguity in ambs
    #     @show method_ambiguity
    # end
    @test length(ambs) == 0
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(ClimaTimeSteppers)
    Aqua.test_stale_deps(ClimaTimeSteppers)
    Aqua.test_deps_compat(ClimaTimeSteppers)
    Aqua.test_project_extras(ClimaTimeSteppers)
    Aqua.test_project_toml_formatting(ClimaTimeSteppers)
    Aqua.test_piracy(ClimaTimeSteppers)
end

nothing
