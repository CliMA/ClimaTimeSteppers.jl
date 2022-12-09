using SafeTestsets
if get(ARGS,1,"Array") == "CuArray"
    using CUDA
    const ArrayType = CUDA.CuArray
else
    const ArrayType = Array
end

@safetestset "SparseContainers" begin include("sparse_containers.jl") end
@safetestset "Newtons method" begin include("test_newtons_method.jl") end
@safetestset "Single column ARS" begin include("single_column_ARS_test.jl") end
@safetestset "Callbacks" begin include("callbacks.jl") end
@safetestset "Aqua" begin include("aqua.jl") end

include("problems.jl")
include("utils.jl")

include("integrator.jl")
include("convergence.jl")
include("test_convergence_checker.jl")
include("compare_generated.jl") # TODO: Remove this.
