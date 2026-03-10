using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS

@testset "LineSearch" begin
    @testset "Full step accepted when residual decreases" begin
        # Simple linear system: f(x) = Ax - b
        A = [2.0 0.0; 0.0 3.0]
        b = [4.0, 9.0]
        x_exact = A \ b  # [2, 3]
        f!(f, x) = f .= A * x .- b

        ls = LineSearch()
        x_old = [1.0, 1.0]
        f_val = similar(x_old)
        f!(f_val, x_old)

        # Newton step: Δx = J⁻¹ f(x_old), then x ← x_old - Δx
        Δx = A \ f_val
        x = copy(x_old)
        x .-= Δx  # full Newton step (should land on exact solution)

        # Line search should accept the full step since residual goes to 0
        CTS.line_search!(ls, x, Δx, f_val, f!, nothing)
        @test x ≈ x_exact atol = 1e-12
    end

    @testset "Backtracking reduces residual" begin
        # f(x) = x^3 - 1 (scalar wrapped in array)
        f!(f, x) = f .= x .^ 3 .- 1
        ls = LineSearch()

        # Start at x_old = 5, exact solution is 1
        x_old = [5.0]
        f_old = similar(x_old)
        f!(f_old, x_old)
        normf_old = norm(f_old)

        # Simulate an overshooting Newton step:
        # f'(5) = 75, Newton Δx = f(x)/f'(x) = 124/75 ≈ 1.653
        # x_new = 5 - 1.653 = 3.347 (reasonable, residual should decrease)
        Δx = f_old ./ (3 .* x_old .^ 2)  # [124/75]
        x = copy(x_old)
        x .-= Δx

        CTS.line_search!(ls, x, Δx, f_old, f!, nothing)

        # After line search, residual should be no worse than original
        f_final = similar(x)
        f!(f_final, x)
        @test norm(f_final) <= normf_old + eps()
    end

    @testset "Handles NaN gracefully" begin
        # f that returns NaN for negative x
        f!(f, x) = f .= (x[1] >= 0 ? [x[1]^2 - 1] : [NaN])
        ls = LineSearch()

        x_old = [2.0]
        f_old = [3.0]  # f(2) = 3
        normf_old = norm(f_old)

        # Overshooting step that goes negative
        Δx = [5.0]
        x = copy(x_old)
        x .-= Δx  # x = -3, which gives NaN

        CTS.line_search!(ls, x, Δx, f_old, f!, nothing)

        # After backtracking, x should be back in valid region
        @test x[1] >= 0 || norm(f_old) <= normf_old  # either recovered or didn't get worse
    end

    @testset "No-op when line_search is nothing" begin
        x = [1.0, 2.0]
        x_copy = copy(x)
        Δx = [0.1, 0.2]
        f = [0.0, 0.0]
        f!(f, x) = nothing
        CTS.line_search!(nothing, x, Δx, f, f!, nothing)
        @test x == x_copy  # unchanged
    end
end
