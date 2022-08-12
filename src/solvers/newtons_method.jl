export NewtonsMethod

"""
    NewtonsMethod(;
        linsolve,
        convergence_checker = nothing,
        max_iters = 1,
        update_j = true,
    )

Solves the equation `f(x) = 0`, given the Jacobian `j(x) = ∂f/∂x`. This is done
by calling `run!(::NewtonsMethod, cache, x, f!, j!)`, where `f!(f, x)` is a
function that sets `f(x)` in-place and `j!(j, x)` is a function that sets `j(x)`
in-place. The `x` passed to a `NewtonsMethod` is modified in-place, and its
initial value is used as a starting guess for the root. The `cache` can be
obtained with `allocate_cache(::NewtonsMethod, x_prototype, j_prototype)`,
where `x_prototype` is `similar` to `x` and `f(x)`, and where `j_prototype` is
`similar` to `j(x)`. Note that `x` and `f(x)` must be `similar` to each other in
order for `j(x)` to be invertible (i.e., in order for it to be a square matrix).

Let `x[n]` denote the value of `x` on the `n`-th Newton iteration (with `x[0]`
denoting the initial value of `x`), and suppose that `x[n]` is sufficiently
close to some root `x̂` of `f(x)` to make the first-order approximation
    `f(x̂) ≈ f(x[n]) + j(x[n]) * (x̂ - x[n])`.
Since `f(x̂) = 0`, the error on the `n`-th iteration is roughly
    `x[n] - x̂ ≈ Δx[n]`, where `Δx[n] = j(x[n]) \\ f(x[n])`.
`NewtonsMethod` sets `x[n + 1]` to be the value of `x̂` given by this
approximation:
    `x[n + 1] = x[n] - Δx[n]`.
If `j(x)` changes sufficiently slowly, `update_j` can be set to `false` in order
to make the approximation `j(x[n]) ≈ j(x[0])` (i.e, to use the "chord method"
instead of Newton's method).
If a convergence checker is provided, it gets used to determine whether to stop
iterating on iteration `n` based on the value `x[n]` and its error `Δx[n]`;
otherwise, `NewtonsMethod` iterates from `n = 0` to `n = max_iters`. If the
convergence checker determines that `x[n]` has not converged by the time
`n = max_iters`, a warning gets printed.
"""
Base.@kwdef struct NewtonsMethod{L, C <: Union{Nothing, ConvergenceChecker}}
    linsolve::L
    convergence_checker::C = nothing
    max_iters::Int = 1
    update_j::Bool = false
end

allocate_cache(alg::NewtonsMethod, x_prototype, j_prototype) =
    (;
        linsolve! = alg.linsolve(Val{:init}, j_prototype, x_prototype),
        convergence_checker_cache = isnothing(alg.convergence_checker) ?
            nothing : allocate_cache(alg.convergence_checker, x_prototype),
        Δx = similar(x_prototype),
        f = similar(x_prototype),
        j = similar(j_prototype),
    )

function run!(alg::NewtonsMethod, cache, x, f!, j!)
    (; convergence_checker, max_iters, update_j) = alg
    (; linsolve!, convergence_checker_cache, Δx, f, j) = cache
    for iter in 0:max_iters
        iter > 0 && (x .-= Δx)
        isnothing(convergence_checker) && iter == max_iters && break
        (update_j || iter == 0) && j!(j, x)
        f!(f, x)
        linsolve!(Δx, j, f)
        if !isnothing(convergence_checker)
            run!(convergence_checker, convergence_checker_cache, x, Δx, iter) &&
                break
            iter == max_iters &&
                @warn "Newton's method didn't converge in $max_iters iterations"
        end
    end
end

# TODO: Instead of just passing j! to Newton's method, wrap it in various
# approximations, like ChordMethod, BroydensMethod, BadBroydensMethod, etc.
# TODO: Allow the Jacobian to be computed once per timestep by defining
# set_jacobian!(::NewtonsMethod, cache) and generalizing update_j.
