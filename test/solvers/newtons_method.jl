#=
julia --project=test
using Revise; include("test/solvers/newtons_method.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Random, Test
import ClimaTimeSteppers as CTS

function linear_equation(FT, n)
    rng = MersenneTwister(1)
    A = rand(rng, FT, n, n)
    b = rand(rng, FT, n)
    f!(f, x) = f .= A * x .- b
    j!(j, x) = j .= A
    x_exact = A \ b
    x_init = zeros(FT, n)
    return (f!, j!, x_exact, x_init)
end

function nonlinear_equation(FT, n)
    rng = MersenneTwister(1)
    A = rand(rng, FT, n, n)
    b = rand(rng, FT, n)
    f!(f, x) = f .= A * sinh.(A * x .- b) .- b
    j!(j, x) = j .= A * diagm(cosh.(A * x .- b)) * A
    x_exact = A \ (asinh.(A \ b) .+ b)
    x_init = zeros(FT, n)
    return (f!, j!, x_exact, x_init)
end

function nonlinear_equation_gentle(FT, n)
    rng = MersenneTwister(1)
    A = rand(rng, FT, n, n)
    # Ensure diagonally dominant for easier convergence with frozen Jacobian
    A = A + I*n
    b = rand(rng, FT, n)
    f!(f, x) = f .= A * x .+ 0.1 .* sin.(x) .- b
    j!(j, x) = j .= A .+ diagm(0.1 .* cos.(x))

    # Approximate exact solution by solving f(x)=0 with full Newton
    x_exact = A \ b
    for _ in 1:10
        f_val = A * x_exact .+ 0.1 .* sin.(x_exact) .- b
        j_val = A .+ diagm(0.1 .* cos.(x_exact))
        x_exact .-= j_val \ f_val
    end

    x_init = zeros(FT, n)
    return (f!, j!, x_exact, x_init)
end

@testset "Newton's Method" begin
    for (is_linear, FT, n, step_adjustment) in (
        (true, Float32, 1, 100000),
        (true, Float32, 10, 1000000),
        (true, Float64, 1, 1000000000),
        (true, Float64, 10, 10000000000),
        (false, Float32, 1, 1),
        (false, Float32, 10, 1),
        (false, Float64, 1, 0.1),
        (false, Float64, 10, 10),
    )
        equation = is_linear ? linear_equation : nonlinear_equation
        max_iters = is_linear ? 1 : 10
        rtol = 100 * eps(FT)
        convergence_checker =
            ConvergenceChecker(; norm_condition = MaximumRelativeError(rtol))
        alg1 = NewtonsMethod(; max_iters, convergence_checker)
        alg2 = NewtonsMethod(;
            max_iters,
            krylov_method = KrylovMethod(; forcing_term = ConstantForcing(rtol)),
            convergence_checker,
        )
        alg3 = NewtonsMethod(;
            max_iters,
            krylov_method = KrylovMethod(;
                jacobian_free_jvp = ForwardDiffJVP(; step_adjustment),
                forcing_term = ConstantForcing(rtol),
            ),
            convergence_checker,
        )
        alg4 = NewtonsMethod(; max_iters, convergence_checker, line_search = LineSearch())
        f!, j!, x_exact, x_init = equation(FT, n)
        for (alg, use_j) in ((alg1, true), (alg2, true), (alg3, false), (alg4, true))
            x = copy(x_init)
            j_prototype = similar(x, length(x), length(x))
            cache = CTS.allocate_cache(alg, x, use_j ? j_prototype : nothing)
            CTS.solve_newton!(alg, cache, x, f!, use_j ? j! : nothing)
            @test norm(x .- x_exact) / norm(x_exact) < rtol
        end
    end
end

@testset "EisenstatWalkerForcing" begin
    # Test that EisenstatWalkerForcing converges on a nonlinear problem.
    # TODO: n≥10 fails because default GMRES workspace (10 Krylov vectors) is
    # insufficient. Add n=10 once KrylovMethod exposes a subspace size parameter.
    for (FT, n) in ((Float64, 1), (Float64, 3))
        f!, j!, x_exact, x_init = nonlinear_equation(FT, n)
        rtol = 100 * eps(FT)
        convergence_checker =
            ConvergenceChecker(; norm_condition = MaximumRelativeError(rtol))
        alg = NewtonsMethod(;
            max_iters = 30,
            krylov_method = KrylovMethod(; forcing_term = EisenstatWalkerForcing()),
            convergence_checker,
        )
        x = copy(x_init)
        j_prototype = similar(x, length(x), length(x))
        cache = CTS.allocate_cache(alg, x, j_prototype)
        CTS.solve_newton!(alg, cache, x, f!, j!)
        @test norm(x .- x_exact) / norm(x_exact) < rtol
    end
end

@testset "ForwardDiffStepSize variants" begin
    # Test ForwardDiffStepSize2 and ForwardDiffStepSize3
    FT = Float64
    n = 5
    f!, j!, x_exact, x_init = nonlinear_equation_gentle(FT, n)
    rtol = 100 * eps(FT)
    convergence_checker = ConvergenceChecker(; norm_condition = MaximumRelativeError(rtol))

    for step_size in (ForwardDiffStepSize2(), ForwardDiffStepSize3())
        alg = NewtonsMethod(;
            max_iters = 30,
            krylov_method = KrylovMethod(;
                jacobian_free_jvp = ForwardDiffJVP(;
                    default_step = step_size,
                    step_adjustment = 1.0,
                ),
                forcing_term = ConstantForcing(rtol),
            ),
            convergence_checker,
        )
        x = copy(x_init)
        cache = CTS.allocate_cache(alg, x, nothing)
        CTS.solve_newton!(alg, cache, x, f!, nothing)
        @test norm(x .- x_exact) / norm(x_exact) < 0.01
    end
end

@testset "Chord method (Jacobian reuse)" begin
    # Use UpdateEvery(NewNewtonSolve) to only update Jacobian once per solve,
    # not every iteration. For a linear problem, this is exact in 1 iteration.
    FT = Float64
    n = 5
    f!, j!, x_exact, x_init = linear_equation(FT, n)
    rtol = 100 * eps(FT)
    convergence_checker = ConvergenceChecker(; norm_condition = MaximumRelativeError(rtol))
    alg = NewtonsMethod(;
        max_iters = 1,
        update_j = UpdateEvery(NewNewtonSolve),
        convergence_checker,
    )
    x = copy(x_init)
    j_prototype = similar(x, length(x), length(x))
    cache = CTS.allocate_cache(alg, x, j_prototype)
    CTS.solve_newton!(alg, cache, x, f!, j!)
    @test norm(x .- x_exact) / norm(x_exact) < rtol

    # Nonlinear chord test using a gentle, diagonally dominant function.
    f!, j!, x_exact, x_init = nonlinear_equation_gentle(FT, n)
    alg_chord = NewtonsMethod(;
        max_iters = 30,
        update_j = UpdateEvery(NewNewtonSolve),
        convergence_checker,
    )
    x = copy(x_init)
    cache = CTS.allocate_cache(alg_chord, x, j_prototype)
    CTS.solve_newton!(alg_chord, cache, x, f!, j!)
    @test norm(x .- x_exact) / norm(x_exact) < 0.01
end

@testset "Dense Matrix From Operator" begin
    n = 10
    rng = MersenneTwister(1)
    A = rand(rng, n, n)
    vector = similar(A, n)
    matrix = similar(A, n, n)
    ClimaTimeSteppers.dense_matrix_from_operator!(matrix, vector, A)
    @test matrix == A
    ClimaTimeSteppers.dense_inverse_matrix_from_operator!(matrix, vector, lu(A))
    @test matrix ≈ inv(A)
end
