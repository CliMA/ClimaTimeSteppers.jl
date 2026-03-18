using SafeTestsets

if "CuArray" in ARGS
    import CUDA
end
# ============================================================================ #
# Unit tests (fast, no convergence studies)
# ============================================================================ #

@safetestset "SparseContainers" begin
    include("unit/sparse_containers.jl")
end

@safetestset "Fused increment" begin
    include("unit/fused_increment.jl")
end

@safetestset "Tableau consistency" begin
    include("unit/tableaus.jl")
end

@safetestset "UpdateSignalHandler" begin
    include("unit/update_signal_handler.jl")
end

@safetestset "LineSearch" begin
    include("unit/line_search.jl")
end

@safetestset "Convergence checker" begin
    include("utils/convergence_checker.jl")
end

@safetestset "ForwardEulerODEFunction" begin
    include("unit/forward_euler.jl")
end

@safetestset "IncrementingODEFunction" begin
    include("unit/incrementing_ode.jl")
end

@safetestset "Edge cases" begin
    include("unit/edge_cases.jl")
end

@safetestset "Jacobian accuracy" begin
    include("unit/jacobian_accuracy.jl")
end

# ============================================================================ #
# Solver correctness tests (Newton's method)
# ============================================================================ #

@safetestset "Newton's method" begin
    include("solvers/newtons_method.jl")
end

# ============================================================================ #
# Integration / system-level tests
# ============================================================================ #

@safetestset "Single column ARS" begin
    include("integration/single_column_ars.jl")
end

@safetestset "Callbacks" begin
    include("integration/callbacks.jl")
end

@safetestset "Integrator tests" begin
    include("integration/integrator.jl")
end

@safetestset "Limiter" begin
    include("integration/limiter.jl")
end

@safetestset "Second Newton solve" begin
    include("integration/second_newton.jl")
end

@safetestset "SSP monotonicity" begin
    include("integration/ssp_monotonicity.jl")
end

@safetestset "Dense output" begin
    include("integration/dense_output.jl")
end

@safetestset "Long-time stability" begin
    include("integration/long_time_stability.jl")
end

# ============================================================================ #
# Performance / allocation tests
# ============================================================================ #

@safetestset "Type stability" begin
    include("performance/type_stability.jl")
end

@safetestset "Step allocations" begin
    include("performance/allocations.jl")
end

# ============================================================================ #
# Solver convergence order tests
# On CI (Buildkite), each solver file runs as a separate parallel job.
# Locally (Pkg.test()), they run sequentially within this testset.
# ============================================================================ #

@safetestset "Algorithm convergence" begin
    include("solvers/explicit_rk.jl")
    include("solvers/lsrk.jl")
    include("solvers/multirate.jl")
    include("solvers/rosenbrock.jl")
    include("solvers/esdirk.jl")
    include("solvers/imex_ark.jl")
end

# ============================================================================ #
# Code quality
# ============================================================================ #

@safetestset "Aqua" begin
    include("aqua.jl")
end
