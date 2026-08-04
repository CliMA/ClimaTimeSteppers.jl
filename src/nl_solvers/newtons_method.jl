#####
##### Newton's method for solving `f(x) = 0`, driving the implicit stage
##### of IMEX algorithms. Each Newton iteration reduces to a linear solve
##### `j · Δx = f`, delegated to an [`AbstractLinearSolveMethod`](@ref)
##### defined in `linear_solvers.jl`.
#####

export NewtonsMethod

"""
    NewtonsMethod(;
        max_iters = 1,
        update_j = UpdateEvery(NewNewtonIteration),
        krylov_method = nothing,
        convergence_checker = nothing,
        verbose = Silent(),
        line_search = false,
    )

Solve `f(x) = 0` by iterating `x[n+1] = x[n] - j(x[n]) \\ f(x[n])`, where
`j(x) = f'(x)` is the Jacobian. Called via
`solve_newton!(method, cache, x, f!, j!, prepare_for_f!)`;
`x` is modified in-place from its initial guess.

# Keyword Arguments
- `max_iters`: maximum Newton iterations (default `1`)
- `update_j`: [`UpdateSignalHandler`](@ref) controlling when the Jacobian is
  recomputed (see *Jacobian update strategies* below)
- `krylov_method`: a [`KrylovMethod`](@ref) to solve the linear system
  approximately. If `nothing`, uses direct `ldiv!` (see *Krylov variant* below)
- `convergence_checker`: a [`ConvergenceChecker`](@ref) that can terminate
  early based on `x[n]` and `Δx[n]`. Without one, always runs `max_iters`
  iterations; if `verbose` is set, a warning is printed when convergence has
  not been reached by `max_iters`.
- `verbose`: `Verbose()` to print `‖x‖` and `‖Δx‖` each iteration
- `line_search`: a [`LineSearch`](@ref) instance to apply backtracking
  (halving up to 5×) when the residual norm does not decrease or becomes
  `NaN`. Default `nothing` (no line search).

# Jacobian update strategies

The `update_j` parameter accepts any [`UpdateSignalHandler`](@ref):
- `UpdateEvery(NewNewtonIteration)` — fresh Jacobian every iteration (default)
- `UpdateEvery(NewNewtonSolve)` — reuse across iterations within one solve
  (the *chord method*: `j(x[n]) ≈ j(x₀)`)
- `UpdateEvery(NewTimeStep)` — reuse across solves within a timestep

External signals can also update the Jacobian between solves via
`update!(method, cache, signal, j!)`.

# Krylov variant

When `krylov_method` is set, `Δx[n]` is computed approximately — this is
a *Newton-Krylov* method. If the Krylov method additionally uses a
Jacobian-free JVP (see [`ForwardDiffJVP`](@ref)), neither `j_prototype`
nor `j!` need to be specified (*Jacobian-free Newton-Krylov*). When both
a JVP and `j` are provided, `j` serves as a left preconditioner.

# Notes on `j_prototype` (in `allocate_cache`)

`j_prototype` should support `ldiv!` directly (e.g., a pre-factorized matrix
or `LinearOperator`). Dense matrices are accepted for convenience but trigger
an `lu` factorization on every solve — suitable only for testing. Note that
`Krylov.jl` does not support dense-matrix preconditioners; when using a
Jacobian-free JVP, `j_prototype` must be `ldiv!`-compatible.
"""
Base.@kwdef struct NewtonsMethod{
    U <: UpdateSignalHandler,
    K <: Union{Nothing, AbstractLinearSolveMethod},
    C <: Union{Nothing, ConvergenceChecker},
    V <: AbstractVerbosity,
    L <: Union{Nothing, LineSearch},
}
    max_iters::Int = 1
    update_j::U = UpdateEvery(NewNewtonIteration)
    krylov_method::K = nothing
    convergence_checker::C = nothing
    verbose::V = Silent()
    line_search::L = nothing
end

function allocate_cache(alg::NewtonsMethod, x_prototype, j_prototype = nothing)
    (; update_j, krylov_method, convergence_checker) = alg
    @assert !(
        isnothing(j_prototype) &&
        (isnothing(krylov_method) || isnothing(krylov_method.jacobian_free_jvp))
    )
    return (;
        krylov_method_cache = isnothing(krylov_method) ? nothing :
                              allocate_cache(krylov_method, x_prototype),
        convergence_checker_cache = isnothing(convergence_checker) ? nothing :
                                    allocate_cache(convergence_checker, x_prototype),
        Δx = zero(x_prototype),
        f = zero(x_prototype),
        j = isnothing(j_prototype) ? nothing : zero(j_prototype),
    )
end

solve_newton!(
    alg::NewtonsMethod,
    cache::Nothing,
    x,
    f!,
    j! = nothing,
    prepare_for_f! = nothing,
) = nothing

NVTX.@annotate function solve_newton!(
    alg::NewtonsMethod,
    cache,
    x,
    f!,
    j! = nothing,
    prepare_for_f! = nothing,
)
    (; max_iters, update_j, krylov_method, convergence_checker, verbose, line_search) = alg
    (; krylov_method_cache, convergence_checker_cache) = cache
    (; Δx, f, j) = cache
    if (!isnothing(j)) && needs_update!(update_j, NewNewtonSolve())
        j!(j, x)
    end
    f!(f, x)
    j_lu = nothing
    for n in 0:(max_iters - 1)
        # Compute Δx[n].
        if (!isnothing(j)) && needs_update!(update_j, NewNewtonIteration())
            j!(j, x)
            if j isa DenseMatrix && isnothing(krylov_method)
                j_lu = lu(j)
            end
        end
        if isnothing(krylov_method)
            # A dense `j` (CPU testing only) must be factored before `ldiv!`;
            # custom operators and GPU arrays support `ldiv!` directly. We cache
            # the LU factorization `j_lu` to avoid redundant factorizations when
            # the Jacobian is not updated every iteration.
            if j isa DenseMatrix
                if isnothing(j_lu)
                    j_lu = lu(j)
                end
                ldiv!(Δx, j_lu, f)
            else
                ldiv!(Δx, j, f)
            end
        else
            solve_krylov!(
                krylov_method,
                krylov_method_cache,
                Δx,
                x,
                f!,
                f,
                n,
                prepare_for_f!,
                j,
            )
        end
        is_verbose(verbose) &&
            @info "Newton iteration $(n + 1): ‖x‖ = $(norm(x)), ‖Δx‖ = $(norm(Δx))"

        x .-= Δx
        line_search!(line_search, x, Δx, f, f!, prepare_for_f!)

        # Update x[n] with Δx[n], and exit the loop if Δx[n + 1] is not needed.
        # Check for convergence if necessary. The `ConvergenceChecker` interface
        # and `solve_krylov!` are both 0-based, matching the loop variable `n`.
        if is_converged!(convergence_checker, convergence_checker_cache, x, Δx, n)
            break
        elseif n < max_iters - 1 && isnothing(line_search)
            isnothing(prepare_for_f!) || prepare_for_f!(x)
            f!(f, x)
        elseif n == max_iters - 1 && is_verbose(verbose)
            @warn "Newton's method did not converge within $(n + 1) iterations: ‖x‖ = $(norm(x)), ‖Δx‖ = $(norm(Δx))"
        end
    end
end
