export NewtonsMethod, KrylovMethod
export JacobianFreeJVP, ForwardDiffJVP, ForwardDiffStepSize
export ForwardDiffStepSize1, ForwardDiffStepSize2, ForwardDiffStepSize3
export ForcingTerm, ConstantForcing, EisenstatWalkerForcing

# TODO: Define ktypeof(::FieldVector) so that it returns CuVector for a
#       FieldVector backed by CuArrays. Without this, Krylov.jl allocates
#       CPU vectors for its workspace, breaking GPU execution of KrylovMethod.
#       Only matters if KrylovMethod is used on GPU (ClimaAtmos currently uses
#       direct ldiv! with max_iters = 1, so this has not been a blocker).

abstract type AbstractVerbosity end
struct Verbose <: AbstractVerbosity end
struct Silent <: AbstractVerbosity end
is_verbose(v::AbstractVerbosity) = v isa Verbose

const KrylovWorkspace =
    @static pkgversion(Krylov) < v"0.10" ? Krylov.KrylovSolver : Krylov.KrylovWorkspace
const krylov_solve! =
    @static pkgversion(Krylov) < v"0.10" ? Krylov.solve! : Krylov.krylov_solve!
const GmresWorkspace =
    @static pkgversion(Krylov) < v"0.10" ? Krylov.GmresSolver : Krylov.GmresWorkspace

"""
    ForwardDiffStepSize

Abstract type for step-size strategies used by [`ForwardDiffJVP`](@ref).
Subtypes are callable: `ε = step_size(Δx, x)` returns the base step `ε`
before any `step_adjustment` scaling.

See [`ForwardDiffStepSize1`](@ref), [`ForwardDiffStepSize2`](@ref),
[`ForwardDiffStepSize3`](@ref).
"""
abstract type ForwardDiffStepSize end

"""
    ForwardDiffStepSize1()

A [`ForwardDiffStepSize`](@ref) derived from a truncation-vs-roundoff error
analysis of the forward difference approximation
`j(x) * Δx ≈ (f(x + ε Δx) - f(x)) / ε`. Not commonly used with Newton-Krylov
methods in practice, but provides intuition for setting `step_adjustment` in a
[`ForwardDiffJVP`](@ref).

Reference: [Oregon State roundoff/truncation notes](https://web.engr.oregonstate.edu/~webbky/MAE4020_5020_files/Section%204%20Roundoff%20and%20Truncation%20Error.pdf).

# Returns

The optimal step size that minimizes the error upper bound is

    `ε = step_adjustment * sqrt(eps(FT)) / ‖Δx‖`,

where `step_adjustment = 2 * sqrt(S * R)` (default 1). Increase
`step_adjustment` when `f` is very smooth (`S ≫ 1`) or has large roundoff
(`R ≫ 1`). For a central difference approximation, the `sqrt` becomes a cube
root (generally, an `n`-th root for order `n - 1`).

# Derivation

## Forward difference error decomposition

The first-order Taylor expansion of `f(x + ε Δx)` around `x` is

    `f(x + ε Δx) = f(x) + j(x)(ε Δx) + e_trunc(x, ε Δx)`,

where `j(x) = f'(x)`. In floating point we can only evaluate `f̂(x)`, with

    `f(x) = f̂(x) + e_round(x)`.

Substituting and rearranging gives the approximation error

    `‖error‖ = ‖e_trunc(x, ε Δx) - e_round(x + ε Δx) + e_round(x)‖ / ε`.

Applying the triangle inequality and approximating
`‖e_round(x + ε Δx)‖ ≈ ‖e_round(x)‖` for small `ε`:

    `‖error‖ ≤ (‖e_trunc(x, ε Δx)‖ + 2 ‖e_round(x)‖) / ε`.

## Bounding truncation error

From Taylor's theorem for multivariate vector-valued functions
([proof](https://math.stackexchange.com/questions/3478229)):

    `‖e_trunc(x, ε Δx)‖ ≤ sup_{x̂ ∈ X} ‖f''(x̂)‖ / 2 · ‖ε Δx‖²`.

Defining the smoothness parameter `S = ‖f(x)‖ / sup ‖f''(x̂)‖` (default `S ≈ 1`;
larger values indicate a small Hessian relative to `f`):

    `‖e_trunc(x, ε Δx)‖ ≤ ε² / (2S) · ‖Δx‖² · ‖f(x)‖`.

## Bounding roundoff error

Assuming componentwise roundoff `|e_round(x)[i]| ≤ R · eps(FT) · |f(x)[i]|`
(default `R ≈ 1`):

    `‖e_round(x)‖ ≤ R · eps(FT) · ‖f(x)‖`.

## Optimal step size

Substituting both bounds into the error bound:

    `‖error‖ ≤ ε/(2S) · ‖Δx‖² · ‖f(x)‖ + 2R · eps(FT) · ‖f(x)‖ / ε`.

Minimizing over `ε` (set derivative to zero) gives
`ε = 2√(SR) · √eps(FT) / ‖Δx‖`, i.e., `step_adjustment = 2√(SR)`.
"""
struct ForwardDiffStepSize1 <: ForwardDiffStepSize end
(::ForwardDiffStepSize1)(Δx, x) = sqrt(eps(eltype(Δx))) / norm(Δx)

"""
    ForwardDiffStepSize2()

A [`ForwardDiffStepSize`](@ref) from Knoll & Keyes, "Jacobian-free
Newton–Krylov methods: a survey of approaches and applications".
This is the step size used by the Fortran package NITSOL:

    `ε = √(eps(FT) * (1 + ‖x‖)) / ‖Δx‖`.
"""
struct ForwardDiffStepSize2 <: ForwardDiffStepSize end
(::ForwardDiffStepSize2)(Δx, x) = sqrt(eps(eltype(Δx)) * (1 + norm(x))) / norm(Δx)

"""
    ForwardDiffStepSize3()

A [`ForwardDiffStepSize`](@ref) from Knoll & Keyes, "Jacobian-free
Newton–Krylov methods: a survey of approaches and applications".
This is the average step size obtained from element-wise forward differences:

    `ε = √eps(FT) · Σᵢ(1 + |xᵢ|) / (length(x) · ‖Δx‖)`.

This is the default step size used by [`ForwardDiffJVP`](@ref).
"""
struct ForwardDiffStepSize3 <: ForwardDiffStepSize end
(::ForwardDiffStepSize3)(Δx, x) =
    sqrt(eps(eltype(Δx))) * sum(x_i -> 1 + abs(x_i), x) / (length(x) * norm(Δx))

"""
    JacobianFreeJVP

Abstract type for matrix-free Jacobian-vector product strategies.
Subtypes compute `j(x) * Δx` without forming `j` explicitly, using only
function evaluations of `f`. Called via
`jvp!(method, cache, jΔx, Δx, x, f!, f, prepare_for_f!)`;
`jΔx` is modified in-place. Allocate `cache` with
`allocate_cache(method, x_prototype)`.

See [`ForwardDiffJVP`](@ref).
"""
abstract type JacobianFreeJVP end

"""
    ForwardDiffJVP(; default_step = ForwardDiffStepSize3(), step_adjustment = 1)

A [`JacobianFreeJVP`](@ref) that approximates the Jacobian-vector product via
first-order forward differences:

    `j(x) * Δx ≈ (f(x + ε Δx) - f(x)) / ε`,

where `ε = step_adjustment * default_step(Δx, x)`.

# Keyword Arguments
- `default_step`: a [`ForwardDiffStepSize`](@ref) (default [`ForwardDiffStepSize3`](@ref))
- `step_adjustment`: multiplicative scaling factor for `ε` (default `1`)
"""
Base.@kwdef struct ForwardDiffJVP{S <: ForwardDiffStepSize, T} <: JacobianFreeJVP
    default_step::S = ForwardDiffStepSize3()
    step_adjustment::T = 1
end

allocate_cache(::ForwardDiffJVP, x_prototype) =
    (; x2 = zero(x_prototype), f2 = zero(x_prototype))

function jvp!(alg::ForwardDiffJVP, cache, jΔx, Δx, x, f!, f, prepare_for_f!)
    (; default_step, step_adjustment) = alg
    (; x2, f2) = cache
    FT = eltype(x)
    ε = FT(step_adjustment) * default_step(Δx, x)
    @. x2 = x + ε * Δx
    isnothing(prepare_for_f!) || prepare_for_f!(x2)
    f!(f2, x2)
    @. jΔx = (f2 - f) / ε
end

"""
    ForcingTerm

Abstract type for the relative tolerance schedule `rtol[n]` used by
[`KrylovMethod`](@ref) inside Newton-Krylov iterations. Called via
`get_rtol!(method, cache, f, n)`, which returns `rtol[n]`. Allocate
`cache` with `allocate_cache(method, x_prototype)`.

See [`ConstantForcing`](@ref), [`EisenstatWalkerForcing`](@ref), and
[Eisenstat & Walker (1996)](http://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR94463.pdf)
for convergence guarantees.
"""
abstract type ForcingTerm end

"""
    ConstantForcing(rtol)

A [`ForcingTerm`](@ref) that returns the fixed value `rtol ∈ [0, 1)` on every
Newton iteration.

# Convergence properties
- `rtol > 0`: linear convergence with asymptotic rate ≤ `rtol`
- `rtol = 0`: exact Krylov solve → quadratic Newton convergence

Smaller `rtol` gives faster asymptotic convergence but increases the risk of
*oversolving* (spending Krylov iterations on accuracy that Newton discards).
"""
struct ConstantForcing{T} <: ForcingTerm
    rtol::T
end

allocate_cache(::ConstantForcing, x_prototype) = (;)

function get_rtol!(alg::ConstantForcing, cache, f, n)
    FT = eltype(f)
    return FT(alg.rtol)
end

"""
    EisenstatWalkerForcing(;
        initial_rtol = 0.5,
        γ = 1,
        α = 2,
        min_rtol_threshold = 0.1,
        max_rtol = 0.9,
    )

Adaptive [`ForcingTerm`](@ref) ("Choice 2" from Eisenstat & Walker, 1996) that
automatically tightens `rtol[n]` as `‖f(x[n])‖` decreases, balancing convergence
speed against oversolving risk.

# Keyword Arguments
- `initial_rtol ∈ [0, 1)`: tolerance for the first Newton iteration
- `γ ∈ [0, 1]`: scaling factor for the tolerance update
- `α ∈ (1, 2]`: convergence-order exponent — larger means faster
  convergence but higher oversolving risk
- `min_rtol_threshold ∈ [0, 1)`: safeguard against tolerance decreasing too
  quickly
- `max_rtol ∈ [0, 1)`: upper bound on `rtol[n]`

# Notes
This is "Choice 2" (not "Choice 1") because it only requires `‖f(x[n])‖`
to compute `rtol[n]`, whereas "Choice 1" also needs the final Krylov residual.
"""
Base.@kwdef struct EisenstatWalkerForcing{T1, T2, T3, T4, T5} <: ForcingTerm
    initial_rtol::T1 = 0.5
    γ::T2 = 1
    α::T3 = 2
    min_rtol_threshold::T4 = 0.1
    max_rtol::T5 = 0.9
end

function allocate_cache(::EisenstatWalkerForcing, x_prototype)
    FT = eltype(x_prototype)
    return (; prev_norm_f = Ref{FT}(), prev_rtol = Ref{FT}())
end

function get_rtol!(alg::EisenstatWalkerForcing, cache, f, n)
    (; initial_rtol, γ, α, min_rtol_threshold, max_rtol) = alg
    (; prev_norm_f, prev_rtol) = cache
    FT = eltype(f)
    norm_f = norm(f)
    if n == 0
        rtol = FT(initial_rtol)
    else
        α = α isa Integer ? α : FT(α)
        rtol = FT(γ) * (norm_f / prev_norm_f[])^α
        min_rtol = FT(γ) * prev_rtol[]^α
        if min_rtol > FT(min_rtol_threshold)
            rtol = max(rtol, min_rtol)
        end
    end
    rtol = min(rtol, FT(max_rtol))
    prev_norm_f[] = norm_f
    prev_rtol[] = rtol
    return rtol
end

"""
    KrylovMethodDebugger

Abstract type for diagnostic hooks run before each Krylov solve. Called via
`print_debug!(method, cache, j, M)`. Allocate `cache` with
`allocate_cache(method, x_prototype)`.

See [`PrintConditionNumber`](@ref).
"""
abstract type KrylovMethodDebugger end

"""
    PrintConditionNumber()

A [`KrylovMethodDebugger`](@ref) that prints `cond(j)` and, when a
preconditioner `M` is available, `cond(M⁻¹ j)` (the effective condition number
seen by the Krylov solver).

!!! warning
    This computes dense representations of `j` and `M⁻¹ j`, which is
    much slower than the Krylov solve itself. Use only for debugging.
"""
struct PrintConditionNumber <: KrylovMethodDebugger end

function allocate_cache(::PrintConditionNumber, x_prototype)
    l = length(x_prototype)
    FT = eltype(x_prototype)
    return (;
        dense_vector = Array{FT}(undef, l),
        dense_j = Array{FT}(undef, l, l),
        dense_inv_M = Array{FT}(undef, l, l),
        dense_inv_M_j = Array{FT}(undef, l, l),
    )
end

print_debug!(::Nothing, cache, j, M) = nothing

function print_debug!(::PrintConditionNumber, cache, j, M)
    (; dense_vector, dense_j, dense_inv_M, dense_inv_M_j) = cache
    dense_matrix_from_operator!(dense_j, dense_vector, j)
    if M === I
        @info "Condition number = $(cond(dense_j))"
    else
        dense_inverse_matrix_from_operator!(dense_inv_M, dense_vector, M)
        mul!(dense_inv_M_j, dense_inv_M, dense_j)
        @info "Condition number = $(cond(dense_inv_M_j)) ($(cond(dense_j)) \
               without the preconditioner)"
    end
end

# Like Matrix(op::AbstractLinearOperator) from LinearOperators.jl, but in-place.
function dense_matrix_from_operator!(dense_matrix, dense_vector, op)
    n_columns = size(dense_matrix)[2]
    dense_vector .= 0
    for column in 1:n_columns
        dense_vector[column] = 1
        mul!(view(dense_matrix, :, column), op, dense_vector)
        dense_vector[column] = 0
    end
end

# Same as dense_matrix_from_operator!, but with ldiv! instead of mul!.
function dense_inverse_matrix_from_operator!(dense_inv_matrix, dense_vector, op)
    n_columns = size(dense_inv_matrix)[2]
    dense_vector .= 0
    for column in 1:n_columns
        dense_vector[column] = 1
        ldiv!(view(dense_inv_matrix, :, column), op, dense_vector)
        dense_vector[column] = 0
    end
end

"""
    KrylovMethod(;
        type = Val(GmresWorkspace),
        jacobian_free_jvp = nothing,
        forcing_term = ConstantForcing(0),
        args = (),
        kwargs = (; memory = 20),
        solve_kwargs = (;),
        disable_preconditioner = false,
        verbose = Silent(),
        debugger = nothing,
    )

Iterative linear solver for Newton's method: finds `Δx[n]` such that
`‖f(x[n]) - j(x[n]) * Δx[n]‖ ≤ rtol[n] * ‖f(x[n])‖`, where `rtol[n]` is
controlled by the [`ForcingTerm`](@ref). Called via
`solve_krylov!(method, cache, Δx, x, f!, f, n, prepare_for_f!, j = nothing)`;
`Δx` is modified in-place. Allocate `cache` with
`allocate_cache(method, x_prototype)`.

This is a wrapper around `Krylov.jl` solvers. By default, GMRES is used with a
Krylov subspace of size 20.

# Keyword Arguments
- `type`: Krylov solver type, wrapped in `Val` (default `Val(GmresWorkspace)`).
- `jacobian_free_jvp`: a [`JacobianFreeJVP`](@ref) for matrix-free operation
  (default `nothing` → uses `j` directly)
- `forcing_term`: a [`ForcingTerm`](@ref) setting `rtol[n]`
  (default `ConstantForcing(0)` → exact solve)
- `args`, `kwargs`: forwarded to the `Krylov.KrylovSolver` constructor
  (default `args = ()`, `kwargs = (; memory = 20)` → GMRES subspace size 20)
- `solve_kwargs`: forwarded to `Krylov.solve!`
- `disable_preconditioner`: if `true`, skip preconditioning even when `j` is
  available (default `false`)
- `verbose`: `Verbose()` to print the Krylov residual each iteration
- `debugger`: a [`KrylovMethodDebugger`](@ref) run before each Krylov solve

# Operator construction

The solver operates on a `LinearOperator` `opj` representing `j(x[n])`:
- **With `jacobian_free_jvp`**: `opj` evaluates `mul!(jΔx, opj, Δx)` via the
  JVP (e.g., finite-difference or AD), so no explicit Jacobian is needed.
- **Without**: `opj` wraps `j` directly, so `mul!` reduces to `mul!(jΔx, j, Δx)`.

# Preconditioning

When *both* a `jacobian_free_jvp` and an explicit `j` are provided (and
`disable_preconditioner` is `false`), `j` is used as a left preconditioner `M`.
The solver calls `ldiv!(Δx′, M, f′)` (not `mul!`), so `M` is treated as an
approximation of `j` rather than as its inverse. If either `j` or the JVP is
missing, or preconditioning is disabled, `M = I`.

# Tolerances

`atol` is fixed to 0 so the convergence criterion remains purely relative:
`‖r‖ ≤ rtol * ‖f‖`. A nonzero `atol` would add a constant floor that prevents
the forcing term from driving the residual to zero, breaking the convergence
guarantees of the Newton-Krylov method.

# Extensibility

All constructor and solver arguments can be overridden via `args`, `kwargs`, and
`solve_kwargs`, so any `Krylov.jl` feature not explicitly covered by this
wrapper remains accessible.
"""
Base.@kwdef struct KrylovMethod{
    T <: Val{<:KrylovWorkspace},
    J <: Union{Nothing, JacobianFreeJVP},
    F <: ForcingTerm,
    A <: Tuple,
    K <: NamedTuple,
    S <: NamedTuple,
    V <: AbstractVerbosity,
    D <: Union{Nothing, KrylovMethodDebugger},
}
    type::T = Val(GmresWorkspace)
    jacobian_free_jvp::J = nothing
    forcing_term::F = ConstantForcing(0)
    args::A = ()
    kwargs::K = (; memory = 20)
    solve_kwargs::S = (;)
    disable_preconditioner::Bool = false
    verbose::V = Silent()
    debugger::D = nothing
end

solver_type(::KrylovMethod{Val{T}}) where {T} = T

function allocate_cache(alg::KrylovMethod, x_prototype)
    (; jacobian_free_jvp, forcing_term, args, kwargs, debugger) = alg
    type = solver_type(alg)
    l = length(x_prototype)

    # Version 0.10 changed how the memory is set
    if pkgversion(Krylov) < v"0.10"
        args = isempty(args) ? (20,) : ()
        kwargs =
            haskey(kwargs, :memory) ?
            Base.structdiff(kwargs, NamedTuple{(:memory,)}((kwargs[:memory],))) : kwargs
    end

    return (;
        jacobian_free_jvp_cache = isnothing(jacobian_free_jvp) ? nothing :
                                  allocate_cache(jacobian_free_jvp, x_prototype),
        forcing_term_cache = allocate_cache(forcing_term, x_prototype),
        solver = type(l, l, args..., Krylov.ktypeof(x_prototype); kwargs...),
        debugger_cache = isnothing(debugger) ? nothing :
                         allocate_cache(debugger, x_prototype),
    )
end

NVTX.@annotate function solve_krylov!(
    alg::KrylovMethod,
    cache,
    Δx,
    x,
    f!,
    f,
    n,
    prepare_for_f!,
    j = nothing,
)
    (; jacobian_free_jvp, forcing_term, solve_kwargs) = alg
    (; disable_preconditioner, debugger) = alg
    type = solver_type(alg)
    (; jacobian_free_jvp_cache, forcing_term_cache, solver, debugger_cache) = cache
    jΔx!(jΔx, Δx) =
        isnothing(jacobian_free_jvp) ? mul!(jΔx, j, Δx) :
        jvp!(jacobian_free_jvp, jacobian_free_jvp_cache, jΔx, Δx, x, f!, f, prepare_for_f!)
    opj = LinearOperator(eltype(x), length(x), length(x), false, false, jΔx!)
    M = disable_preconditioner || isnothing(j) || isnothing(jacobian_free_jvp) ? I : j
    print_debug!(debugger, debugger_cache, opj, M)
    ldiv = true
    atol = zero(eltype(Δx))
    rtol = get_rtol!(forcing_term, forcing_term_cache, f, n)
    verbose = Int(is_verbose(alg.verbose))
    krylov_solve!(solver, opj, f; M, ldiv, atol, rtol, verbose, solve_kwargs...)
    iter = solver.stats.niter
    if !solver.stats.solved
        str1 = isnothing(j) ? () : ("the Jacobian",)
        str2 = isnothing(jacobian_free_jvp) ? () : ("the Jacobian-vector product",)
        str = join((str1..., str2...), " and/or ")
        if solver.stats.inconsistent
            @debug "$type detected that the Jacobian is singular on iteration \
                   $iter; if possible, try improving the approximation of $str"
        else
            @debug "$type did not converge within $iter iterations; if \
                   possible, try improving the approximation of $str, or try \
                   increasing the forcing term"
        end
    elseif iter == 0 && solver.stats.status != "x = 0 is a zero-residual solution"
        @debug "$type set Δx to 0 without running any iterations; if possible, \
               try decreasing the forcing term"
    end
    Δx .= Krylov.solution(solver)
end

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
  iterations; if convergence has not been reached by `max_iters`, a warning
  is printed.
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
    K <: Union{Nothing, KrylovMethod},
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
    for n in 1:max_iters
        # Compute Δx[n].
        if (!isnothing(j)) && needs_update!(update_j, NewNewtonIteration())
            j!(j, x)
        end
        if isnothing(krylov_method)
            if j isa DenseMatrix
                ldiv!(Δx, lu(j), f) # Highly inefficient! Only used for testing.
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
                n - 1,
                prepare_for_f!,
                j,
            )
        end
        is_verbose(verbose) &&
            @info "Newton iteration $n: ‖x‖ = $(norm(x)), ‖Δx‖ = $(norm(Δx))"

        x .-= Δx
        line_search!(line_search, x, Δx, f, f!, prepare_for_f!)

        # Update x[n] with Δx[n - 1], and exit the loop if Δx[n] is not needed.
        # Check for convergence if necessary.
        if is_converged!(convergence_checker, convergence_checker_cache, x, Δx, n)
            break
        elseif n < max_iters && isnothing(line_search)
            isnothing(prepare_for_f!) || prepare_for_f!(x)
            f!(f, x)
        elseif n == max_iters && is_verbose(verbose)
            @warn "Newton's method did not converge within $n iterations: ‖x‖ = $(norm(x)), ‖Δx‖ = $(norm(Δx))"
        end
    end
end
