using SafeTestsets

#=
TODO: add separate GPU tests
if get(ARGS,1,"Array") == "CuArray"
    using CUDA
    const ArrayType = CUDA.CuArray
else
    const ArrayType = Array
end
=#

@safetestset "SparseContainers" begin
    include("sparse_containers.jl")
end
@safetestset "Newtons method" begin
    include("test_newtons_method.jl")
end
@safetestset "Single column ARS" begin
    include("single_column_ARS_test.jl")
end
@safetestset "Callbacks" begin
    include("callbacks.jl")
end
@safetestset "Aqua" begin
    include("aqua.jl")
end
@safetestset "Integrator tests" begin
    include("integrator.jl")
end
@safetestset "Consistency" begin
    include("ars_consistency.jl")
end
@safetestset "Algorithm convergence" begin
    include("convergence.jl")
end
@safetestset "Convergence checker unit tests" begin
    include("test_convergence_checker.jl")
end
