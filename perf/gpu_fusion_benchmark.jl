#=
Benchmark for Lazy Broadcast Fusion Optimization
Measures the kernel execution time reduction when fusing multiple increments.
Run on a machine with a GPU:
    julia --project=perf perf/gpu_fusion_benchmark.jl
=#
using BenchmarkTools
using CUDA
using StaticArrays
using ClimaTimeSteppers:
    SparseCoeffs, fused_increment!, fused_raw_increment, SparseContainer

# --- Define Fake Containers & Functions ---
# Replicate the old (unfused) logic for benchmarking comparison
function compute_stage_value_old!(U, u, dt, a_exp, a_imp, T_exp, T_imp, val_i)
    @. U = u
    fused_increment!(U, dt, a_exp, T_exp, val_i)
    fused_increment!(U, dt, a_imp, T_imp, val_i)
    return nothing
end

# The new fused logic
function compute_stage_value_new!(U, u, dt, a_exp, a_imp, T_exp, T_imp, val_i)
    inc_exp = fused_raw_increment(dt, a_exp, T_exp, val_i)
    inc_imp = fused_raw_increment(dt, a_imp, T_imp, val_i)
    @. U = u + inc_exp + inc_imp
    return nothing
end

# --- Benchmark Setup ---
n = 2^22 # ~4M elements to saturate GPU
U = CUDA.zeros(Float64, n)
u = CUDA.ones(Float64, n)
tend1 = CUDA.fill(0.5, n)
tend2 = CUDA.fill(0.7, n)
tend3 = CUDA.fill(1.2, n)
tend4 = CUDA.fill(-0.3, n)
dt = 0.1

# Mock containers
T_exp = SparseContainer((tend1, tend2), (1, 2))
T_imp = SparseContainer((tend3, tend4), (1, 2))

# Mock coefficients for i=3 stage (uses indices 1 and 2)
mat_a_exp = @SMatrix [
    0.0 0.0 0.0
    0.5 0.0 0.0
    0.2 0.3 0.0
]
mat_a_imp = @SMatrix [
    0.0 0.0 0.0
    0.0 0.5 0.0
    0.1 0.1 0.5
]
sc_exp = SparseCoeffs(mat_a_exp)
sc_imp = SparseCoeffs(mat_a_imp)

val_i = Val(3)

# Warmup
println("Warming up kernels...")
compute_stage_value_old!(U, u, dt, sc_exp, sc_imp, T_exp, T_imp, val_i)
compute_stage_value_new!(U, u, dt, sc_exp, sc_imp, T_exp, T_imp, val_i)
CUDA.synchronize()

# --- Execution ---
println("Benchmarking OLD (unfused) approach...")
t_old = @belapsed begin
    compute_stage_value_old!($U, $u, $dt, $sc_exp, $sc_imp, $T_exp, $T_imp, $val_i)
    CUDA.synchronize()
end

println("Benchmarking NEW (fused) approach...")
t_new = @belapsed begin
    compute_stage_value_new!($U, $u, $dt, $sc_exp, $sc_imp, $T_exp, $T_imp, $val_i)
    CUDA.synchronize()
end

println("-------------------------------------------")
println("OLD time: ", round(t_old * 1e6, digits = 2), " μs")
println("NEW time: ", round(t_new * 1e6, digits = 2), " μs")
println("Speedup: ", round(t_old / t_new, digits = 2), "x")
println("-------------------------------------------")
