# ============================================================================ #
# Array-based PDE test problems
# ============================================================================ #

"""
2D heat equation test utilizing standard finite differences.
PDE: ∂u/∂t = ∇²u on [0,1]² with homogeneous Dirichlet BCs.

The full Laplacian is treated explicitly (T_exp!). The analytic solution
uses the discrete eigenvalue of the finite-difference Laplacian, so the
convergence test measures only the temporal discretization error.
"""
function finitediff_2Dheat_test_cts(::Type{FT}) where {FT}
    Nx = 5
    Ny = 5
    Δx = FT(1) / FT(Nx)
    Δy = FT(1) / FT(Ny)

    x = range(Δx, step = Δx, length = Nx - 1)
    y = range(Δy, step = Δy, length = Ny - 1)

    n_x = 1
    n_y = 1

    init_state = Array{FT}(undef, Nx - 1, Ny - 1)
    for i in 1:(Nx - 1), j in 1:(Ny - 1)
        init_state[i, j] = sin(2 * FT(π) * n_x * x[i]) * sin(2 * FT(π) * n_y * y[j])
    end
    init_state = vec(init_state)

    # 2D Laplacian matrix (Kronecker product of 1D operators)
    I_x = Matrix{FT}(I, Nx - 1, Nx - 1)
    I_y = Matrix{FT}(I, Ny - 1, Ny - 1)
    A_x = Matrix(
        Tridiagonal(
            fill(FT(1 / Δx^2), Nx - 2),
            fill(FT(-2 / Δx^2), Nx - 1),
            fill(FT(1 / Δx^2), Nx - 2),
        ),
    )
    A_y = Matrix(
        Tridiagonal(
            fill(FT(1 / Δy^2), Ny - 2),
            fill(FT(-2 / Δy^2), Ny - 1),
            fill(FT(1 / Δy^2), Ny - 2),
        ),
    )
    A = kron(I_y, A_x) + kron(A_y, I_x)

    function T_exp!(tendency, state, _, t)
        mul!(tendency, A, state)
    end

    # Analytic solution via discrete eigenvalue (exact for the FD discretization)
    function analytic_sol(t)
        λ_x = (4 / Δx^2) * sin(FT(π) * n_x * Δx / 2)^2
        λ_y = (4 / Δy^2) * sin(FT(π) * n_y * Δy / 2)^2
        λ_discrete = λ_x + λ_y
        return exp(-λ_discrete * t) .* init_state
    end

    t_end = FT(0.05)
    tendency_func = ClimaODEFunction(; T_exp!)
    split_tendency_func = tendency_func
    make_prob(func) = ODEProblem(func, init_state, (FT(0), t_end), nothing)
    IntegratorTestCase(
        "2D Heat Equation (FD)",
        false,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        make_prob(split_tendency_func),
        50,
        1,
    )
end

"""
1D heat equation test utilizing standard finite differences.
PDE: ∂u/∂t = ∂²u/∂z² on [0,1] with homogeneous Dirichlet BCs.

The full Laplacian is treated explicitly (T_exp!). The analytic solution
uses the discrete eigenvalue of the finite-difference Laplacian, so the
convergence test measures only the temporal discretization error.
"""
function finitediff_1Dheat_test_cts(::Type{FT}) where {FT}
    N = 10
    Δz = FT(1) / FT(N)

    # Interior points only (Dirichlet boundary conditions u=0)
    z = range(Δz, step = Δz, length = N - 1)

    # Laplacian matrix
    A = Matrix(
        Tridiagonal(
            fill(FT(1 / Δz^2), N - 2),
            fill(FT(-2 / Δz^2), N - 1),
            fill(FT(1 / Δz^2), N - 2),
        ),
    )

    n_z = 1
    φ_sin = @. sin(2 * FT(π) * n_z * z)

    init_state = Array{FT}(φ_sin)

    function T_exp!(tendency, state, _, t)
        mul!(tendency, A, state)
    end

    # Analytic solution via discrete eigenvalue (exact for the FD discretization)
    function analytic_sol(t)
        λ_discrete = (4 / Δz^2) * sin(FT(π) * n_z * Δz / 2)^2
        return exp(-λ_discrete * t) .* init_state
    end

    t_end = FT(0.1)
    tendency_func = ClimaODEFunction(; T_exp!)
    split_tendency_func = tendency_func
    make_prob(func) = ODEProblem(func, init_state, (FT(0), t_end), nothing)
    IntegratorTestCase(
        "1D Heat Equation (FD)",
        false,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        make_prob(split_tendency_func),
        50,
        1,
    )
end
