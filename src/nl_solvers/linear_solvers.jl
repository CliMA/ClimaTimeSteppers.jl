#####
##### Linear solvers used by `NewtonsMethod` to compute Δx.
#####
##### Contains: verbosity types (used by NewtonsMethod too), matrix-free
##### JVP machinery, forcing-term stack for GMRES's inner tolerance, the
##### `AbstractLinearSolveMethod` interface, `KrylovMethod` (GMRES via
##### Krylov.jl), and `RichardsonMethod` (preconditioned minimum-residual
##### Richardson / Orthomin(1)). Must be included before `newtons_method.jl`
##### since `NewtonsMethod`'s `krylov_method` field is typed against
##### `AbstractLinearSolveMethod` and its `verbose` default is `Silent()`.
#####

export KrylovMethod, RichardsonMethod
export AbstractLinearSolveMethod
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
    AbstractLinearSolveMethod

Abstract type for linear-solve methods used inside [`NewtonsMethod`](@ref)
to compute the Newton step `Δx[n]` from `j(x[n]) * Δx[n] = f(x[n])`.
Concrete subtypes provide the interface

    allocate_cache(alg::AbstractLinearSolveMethod, x_prototype)
    solve_krylov!(alg::AbstractLinearSolveMethod, cache, Δx, x, f!, f, n, prepare_for_f!, j = nothing)

`solve_krylov!` computes `Δx` in place. The function name is retained for
backward compatibility; despite it, the dispatch handles every subtype.

Available subtypes:

  - [`KrylovMethod`](@ref) — GMRES (and other `Krylov.jl` solvers).
    Optimal for poorly-conditioned linear systems where the Krylov
    subspace iteration significantly outperforms fixed-point methods.
  - [`RichardsonMethod`](@ref) — fixed-iteration preconditioned
    Richardson iteration (iterative refinement). Much lower per-iteration
    overhead than Krylov (no orthogonalization, Hessenberg update, or
    least-squares solve), but requires a preconditioner `M ≈ j` for
    competitive convergence. Best when the manual Jacobian is close to
    the true tangent and only 1–3 iterations are needed.
"""
abstract type AbstractLinearSolveMethod end

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

[`AbstractLinearSolveMethod`](@ref) that solves the Newton linear system
via a `Krylov.jl` iterative method (GMRES by default). Finds `Δx[n]`
such that `‖f(x[n]) - j(x[n]) * Δx[n]‖ ≤ rtol[n] * ‖f(x[n])‖`, where
`rtol[n]` is controlled by the [`ForcingTerm`](@ref). Called via
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
- `preconditioner`: a custom left preconditioner `M` (e.g., a matrix-free 
   preconditioner or block-diagonal operator) supporting `ldiv!` (default `nothing`)

# Operator construction

The solver operates on a `LinearOperator` `opj` representing `j(x[n])`:
- **With `jacobian_free_jvp`**: `opj` evaluates `mul!(jΔx, opj, Δx)` via the
  JVP (e.g., finite-difference or AD), so no explicit Jacobian is needed.
- **Without**: `opj` wraps `j` directly, so `mul!` reduces to `mul!(jΔx, j, Δx)`.

# Preconditioning

When `disable_preconditioner` is `false` and a `preconditioner` is provided, it
is used as the left preconditioner `M`. Otherwise, when *both* a
`jacobian_free_jvp` and an explicit `j` are provided (and `disable_preconditioner`
is `false`), `j` is used as a left preconditioner `M`.
The solver calls `ldiv!(Δx′, M, f′)` (not `mul!`), so `M` is treated as an
approximation of `j` rather than as its inverse. If no preconditioner or `j` is
available, or preconditioning is disabled, `M = I`.

# Tolerances

`atol` is fixed to 0 so the convergence criterion remains purely relative:
`‖r‖ ≤ rtol * ‖f‖`. A nonzero `atol` would add a constant floor that prevents
the forcing term from driving the residual to zero, breaking the convergence
guarantees of the Newton-Krylov method.

# Convergence failures

A failed Krylov solve (most often a singular or inconsistent Jacobian) is *not*
raised or warned about: warning from inside the hot, GPU-dispatched Newton loop
would be both noisy and problematic on accelerators. Instead, `Δx` is set to the
least-squares solution returned by `Krylov.jl`, and the failure is reported only
at `@debug` level. To diagnose suspected non-convergence, set `verbose =
Verbose()` (to print the Krylov residual each iteration) or attach a
[`KrylovMethodDebugger`](@ref) such as [`PrintConditionNumber`](@ref).

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
    P,
} <: AbstractLinearSolveMethod
    type::T = Val(GmresWorkspace)
    jacobian_free_jvp::J = nothing
    forcing_term::F = ConstantForcing(0)
    args::A = ()
    kwargs::K = (; memory = 20)
    solve_kwargs::S = (;)
    disable_preconditioner::Bool = false
    verbose::V = Silent()
    debugger::D = nothing
    preconditioner::P = nothing
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
    (; disable_preconditioner, debugger, preconditioner) = alg
    type = solver_type(alg)
    (; jacobian_free_jvp_cache, forcing_term_cache, solver, debugger_cache) = cache
    jΔx!(jΔx, Δx) =
        isnothing(jacobian_free_jvp) ? mul!(jΔx, j, Δx) :
        jvp!(jacobian_free_jvp, jacobian_free_jvp_cache, jΔx, Δx, x, f!, f, prepare_for_f!)
    opj = LinearOperator(eltype(x), length(x), length(x), false, false, jΔx!)
    M =
        disable_preconditioner ? I :
        (
            !isnothing(preconditioner) ? preconditioner :
            ((isnothing(j) || isnothing(jacobian_free_jvp)) ? I : j)
        )
    print_debug!(debugger, debugger_cache, opj, M)
    ldiv = true
    atol = zero(eltype(Δx))
    rtol = get_rtol!(forcing_term, forcing_term_cache, f, n)
    verbose = Int(is_verbose(alg.verbose))
    krylov_solve!(solver, opj, f; M, ldiv, atol, rtol, verbose, solve_kwargs...)
    iter = solver.stats.niter
    if !solver.stats.solved
        # The Krylov solve failed (a singular/inconsistent Jacobian is the usual
        # cause). We report this only at `@debug` level rather than `@warn`:
        # warning from inside this hot, GPU-dispatched loop is both noisy and
        # problematic on accelerators. A singular Jacobian leaves `Δx` at the
        # least-squares solution Krylov returns; see the docstring for diagnosis.
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
    RichardsonMethod(;
        jacobian_free_jvp,
        n_iters = 1,
        rtol = 0,
        preconditioner = nothing,
        verbose = Silent(),
    )

[`AbstractLinearSolveMethod`](@ref) that solves the Newton linear system
`j(x[n]) * Δx[n] = f(x[n])` by fixed-iteration preconditioned
minimum-residual Richardson (Orthomin(1)). A parallel alternative to
[`KrylovMethod`](@ref) with the same GMRES(1)-quality single-step
reduction, but without the Krylov.jl orthogonalization / Hessenberg /
Givens / least-squares scaffolding.

Called via
`solve_krylov!(method, cache, Δx, x, f!, f, n, prepare_for_f!, j = nothing)`;
`Δx` is modified in-place. Allocate `cache` with
`allocate_cache(method, x_prototype)`.

# Algorithm

Preload `z_1 = M⁻¹ f`. Then for `k = 1, …, n_iters`:

    w_k  = j · z_k                              (JVP)
    v_k  = M⁻¹ · w_k
    α_k  = ⟨z_k, v_k⟩ / ⟨v_k, v_k⟩              (minimizes ‖M⁻¹ r_{k+1}‖)
    Δx_{k+1} = Δx_k + α_k · z_k
    z_{k+1}  = z_k − α_k · v_k                  (algebraic update — no M-solve)

`M` is the preconditioner (`preconditioner` or, if unset, the Newton
`j`); `j · z_k` is evaluated matrix-free by the
[`JacobianFreeJVP`](@ref).

The `z` update uses the identity
`M⁻¹ r_{k+1} = M⁻¹ (r_k − α_k w_k) = z_k − α_k v_k`, so the
preconditioned residual can be maintained without any additional
`ldiv!`. This is the same trick used inside preconditioned CG/GMRES
to avoid re-preconditioning the residual each iteration. `r_k` itself
is never materialized explicitly.

At iteration 1, `α₁ = ⟨M⁻¹f, M⁻¹j·M⁻¹f⟩ / ‖M⁻¹j·M⁻¹f‖²` — the same
coefficient that left-preconditioned GMRES computes on its first
iteration (see [`KrylovMethod`](@ref), called with `ldiv = true` and
`M = j`). Both methods therefore produce the same `Δx` at `n_iters = 1`
up to floating-point rounding, and their reported `‖M⁻¹ r‖` values are
directly comparable.

# Cost per iteration

  - 1 preconditioner solve (`ldiv!(M)`) for `v_k = M⁻¹ w_k`. (Plus one
    preload `ldiv!` before the loop for `z_1 = M⁻¹ f`.)
  - 1 Jacobian-vector product (`f!` evaluation via `jacobian_free_jvp`).
  - 2 inner products (for `α_k`).
  - 3 vector `axpy!`-style updates.

Matches left-preconditioned GMRES exactly in M-solve and JVP count
per iteration. No Gram–Schmidt orthogonalization, no Hessenberg
matrix, no Givens rotations, no least-squares solve. Per-iteration
work is bounded and independent of `n_iters`, unlike GMRES where
iteration `k` performs `k` orthogonalizations.

# Convergence tradeoff

Minimum-residual Richardson (Orthomin(1) with preconditioned norm)
minimizes `‖M⁻¹ r‖` over the 1-D span `{M⁻¹ r_k}` at each step. Its
residual reduction ratio is governed by the spectrum of `M⁻¹·j`;
convergence is superlinear in easy regimes and linear otherwise. It
matches GMRES(1) exactly at `n_iters = 1`; for `n_iters ≥ 2` GMRES
optimizes over the full `k`-dim Krylov subspace and is strictly
better per iteration, at the cost of orthogonalization overhead. Use
`RichardsonMethod` when `n_iters` is small (≲ 3) and per-iter
overhead matters; use [`KrylovMethod`](@ref) when many iterations or
the strongest possible subspace approximation is needed.

# Keyword Arguments

  - `jacobian_free_jvp`: required [`JacobianFreeJVP`](@ref) used to
    evaluate `j · z` matrix-free on every iteration.
  - `n_iters`: maximum number of Richardson iterations (default `1`).
    `n_iters = 0` skips the loop entirely and returns the direct
    preconditioner solve `Δx = M⁻¹ f` (1 M-solve, 0 JVPs) — the
    cheapest possible mode. `n_iters ≥ 1` runs the min-residual
    correction. If `rtol > 0` the loop can exit earlier once the
    residual target is met.
  - `rtol`: relative tolerance for early exit (default `0`, i.e. no
    early exit). When positive, the iteration stops as soon as
    `‖M⁻¹ r_k‖ ≤ rtol · ‖M⁻¹ f‖` — the *preconditioned* residual
    ratio, matching the semantics of [`ConstantForcing`](@ref) in
    [`KrylovMethod`](@ref) so tolerances are directly comparable.
  - `preconditioner`: a custom left preconditioner `M` supporting
    `ldiv!` (default `nothing` → use the Newton `j` supplied at solve
    time). A preconditioner is essential; without one, the method
    errors at solve time.
  - `verbose`: `Verbose()` to print `α` and `‖r‖` per iteration.
"""
Base.@kwdef struct RichardsonMethod{
    J <: JacobianFreeJVP,
    V <: AbstractVerbosity,
    P,
    T,
} <: AbstractLinearSolveMethod
    jacobian_free_jvp::J
    n_iters::Int = 1
    rtol::T = 0
    preconditioner::P = nothing
    verbose::V = Silent()
end

function allocate_cache(alg::RichardsonMethod, x_prototype)
    return (;
        jacobian_free_jvp_cache = allocate_cache(alg.jacobian_free_jvp, x_prototype),
        z = zero(x_prototype),      # preconditioned residual M⁻¹ r, also search direction
        w = zero(x_prototype),      # matvec buffer j·z
        v = zero(x_prototype),      # preconditioned matvec M⁻¹·w
    )
end

NVTX.@annotate function solve_krylov!(
    alg::RichardsonMethod,
    cache,
    Δx,
    x,
    f!,
    f,
    n,
    prepare_for_f!,
    j = nothing,
)
    (; jacobian_free_jvp, n_iters, rtol, preconditioner, verbose) = alg
    (; jacobian_free_jvp_cache, z, w, v) = cache
    M = isnothing(preconditioner) ? j : preconditioner
    isnothing(M) && error(
        "RichardsonMethod requires either an explicit `preconditioner` or a \
        Newton Jacobian `j` supporting `ldiv!`.",
    )

    # Preload z = M⁻¹ f (both the initial search direction and the
    # initial preconditioned residual — r is never materialized).
    ldiv!(z, M, f)
    # Initial ‖z‖² and the (squared) early-exit threshold. Inside the
    # loop `z_norm2` is updated *recursively* from the α and ⟨v, v⟩ we
    # already compute — no per-iter `norm(z)` reduction is needed.
    # `rtol = 0` sets `tol = 0`, so the pre-loop / in-loop early-exit
    # branches only fire when `z_norm2` reaches a floating-point zero
    # (rare, e.g. f ≈ 0).
    FT = eltype(f)
    rtol_FT = convert(FT, rtol)
    z_norm2 = sum(abs2, z)
    tol = rtol_FT * rtol_FT * z_norm2
    is_verbose(verbose) &&
        @info "RichardsonMethod iteration 0: ‖M⁻¹ f‖ = $(sqrt(z_norm2))"
    # Fall through to Δx = M⁻¹ f when either (a) `n_iters = 0` (user
    # requested no correction) or (b) we're already at the residual
    # target. Doing this here (rather than inside the loop) guarantees
    # `Δx` is written and lets the loop drop its iter-1 guard.
    if n_iters == 0 || z_norm2 <= tol
        @. Δx = z
        return nothing
    end

    for k in 1:n_iters
        # Early exit before doing work. Safe from k=1 because the
        # pre-loop check above already rules out `z_norm2 <= tol` on
        # entry — this branch only fires after `z_norm2` has been
        # driven down by a previous iter.
        z_norm2 <= tol && break
        # w = j · z; v = M⁻¹ · w. α minimizes ‖M⁻¹ r − α·v‖, matching
        # left-preconditioned GMRES's objective at iteration 1.
        # Degenerate `⟨v, v⟩ = 0` (z in the null space of M⁻¹ j) → α = 0.
        jvp!(
            jacobian_free_jvp,
            jacobian_free_jvp_cache,
            w,
            z,
            x,
            f!,
            f,
            prepare_for_f!,
        )
        ldiv!(v, M, w)
        # α = ⟨z, v⟩ / ⟨v, v⟩, computed with single-tree walks:
        # `sum(abs2, v)` replaces `dot(v, v)`, and `sum(z .* v)` (with
        # `z .* v` materialized into the now-unused `w` buffer) replaces
        # `dot(z, v)`. Avoids the paired-tree dispatch overhead of
        # `LinearAlgebra.dot` on nested `FieldVector`s. Same reduction
        # *count* (2) but each is a single mapreduce over one tree.
        vnorm2 = sum(abs2, v)
        @. w = z * v
        zv = sum(w)
        α = vnorm2 > 0 ? zv / vnorm2 : zero(vnorm2)
        # Order matters: Δx += α·z uses z_k, then z −= α·v updates it to
        # z_{k+1} = M⁻¹ r_{k+1} via the algebraic identity — no M-solve.
        if k == 1
            @. Δx = α * z
        else
            @. Δx += α * z
        end
        @. z -= α * v
        # Recursive ‖z‖² update: ‖z_{k+1}‖² = ‖z_k‖² − α²·⟨v, v⟩.
        # Floor at 0 to guard against float drift over many iterations.
        z_norm2 = max(z_norm2 - α * α * vnorm2, zero(z_norm2))
        is_verbose(verbose) && @info "RichardsonMethod iteration $k: \
                                        α = $α, ‖M⁻¹ r‖ = $(sqrt(z_norm2))"
    end
    return nothing
end
