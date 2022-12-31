using ClimaTimeSteppers
using Test
cts_dir = pkgdir(ClimaTimeSteppers)

ENV["GKSwstype"] = "nul" # avoid displaying plots

include(joinpath(@__DIR__, "plotting_utils.jl"))
include(joinpath(cts_dir, "test", "convergence_orders.jl"))
include(joinpath(cts_dir, "test", "utils.jl"))
include(joinpath(cts_dir, "test", "problems.jl"))

@testset "IMEX ARK Algorithms" begin
    tab1 = (ARS111, ARS121)
    tab2 = (ARS122, ARS232, ARS222, IMKG232a, IMKG232b, IMKG242a, IMKG242b)
    tab2 = (tab2..., IMKG252a, IMKG252b, IMKG253a, IMKG253b, IMKG254a)
    tab2 = (tab2..., IMKG254b, IMKG254c, HOMMEM1)
    tab3 = (ARS233, ARS343, ARS443, IMKG342a, IMKG343a, DBM453)
    tabs = [tab1..., tab2..., tab3...]
    tabs = map(t -> t(), tabs)
    test_algs("IMEX ARK", tabs, ark_analytic_nonlin_test_cts(Float64), 400)
    test_algs("IMEX ARK", tabs, ark_analytic_sys_test_cts(Float64), 60)
    test_algs("IMEX ARK", tabs, ark_analytic_test_cts(Float64), 16000; super_convergence = ARS121())
end
