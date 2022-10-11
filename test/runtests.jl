if get(ARGS,1,"Array") == "CuArray"
    using CUDA
    const ArrayType = CUDA.CuArray
else
    const ArrayType = Array
end

include("testhelper.jl")
include("problems.jl")
include("utils.jl")

include("integrator.jl")
include("convergence.jl")
include("callbacks.jl")
include("test_convergence_checker.jl")
include("single_column_ARS_test.jl")
include("compare_generated.jl") # TODO: Remove this.
include("aqua.jl")

#=
@testset "ODE Tests: Basic" begin
    runmpi(joinpath(@__DIR__, "basic.jl"))
end
=#
# FIXME: Should consolodate all convergence tests into single
# testset --- this test is slightly redundant
# @testset "ODE Tests: Convergence" begin
#     runmpi(joinpath(@__DIR__, "ode_tests_convergence.jl"))
# end
