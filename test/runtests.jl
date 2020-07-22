include(joinpath("testhelper.jl"))

@testset "ODE Tests: Basic" begin
    runmpi(joinpath(@__DIR__, "ode_tests_basic.jl"))
end

# FIXME: Should consolodate all convergence tests into single
# testset --- this test is slightly redundant
# @testset "ODE Tests: Convergence" begin
#     runmpi(joinpath(@__DIR__, "ode_tests_convergence.jl"))
# end
