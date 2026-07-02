export LineSearch

using LinearAlgebra: norm

"""
    LineSearch(; norm = norm)

Applies a simple backtracking line search to a Newton update.

The line search assumes that `x` has already been updated by the full
Newton step `x ← x - Δx`, and that `f` initially contains `f(x_old)`.
The full Newton step is evaluated first. If the residual norm does not
decrease or becomes non-finite, the step is repeatedly reduced by
moving `x` back toward the previous iterate along the Newton direction.
At most five halvings are performed.

The residual norm is measured using `norm(f)`, which can be customized
by providing an alternative `norm` function.

The update is performed in-place. The final `(x, f)` correspond to the
lowest-residual iterate tried; in particular, a non-finite iterate is never
returned when a finite one was found (falling back to the previous iterate if
no step is finite).
"""
Base.@kwdef struct LineSearch{N}
    norm::N = norm
end

line_search!(alg::Nothing, x, Δx, f, f!, prepare_for_f!) = nothing
function line_search!(alg::LineSearch, x, Δx, f, f!, prepare_for_f!)
    (; norm) = alg
    normf0 = norm(f)     # f(x_old)

    isnothing(prepare_for_f!) || prepare_for_f!(x)
    f!(f, x)
    normf = norm(f)

    # Track the lowest-residual iterate tried, as a step fraction `s` with
    # x = x_old - s*Δx (the full Newton step is s = 1). This lets us recover the
    # best iterate if backtracking does not end on it, and avoids returning a
    # non-finite iterate when a finite one was found — without logging from this
    # hot, GPU-dispatched path.
    s = 1.0
    best_s = s
    best_norm = isfinite(normf) ? normf : oftype(normf, Inf)

    # if not improved (or NaN), bisect back toward x_old up to 5 times
    i = 1
    α = 0.5
    while ((normf > normf0) || !isfinite(normf)) && (i <= 5)
        x .+= α * Δx    # move back toward x_old
        s -= α
        isnothing(prepare_for_f!) || prepare_for_f!(x)
        f!(f, x)
        normf = norm(f)
        if isfinite(normf) && normf < best_norm
            best_norm = normf
            best_s = s
        end
        α /= 2
        i += 1
    end

    # Reposition x to the best iterate found (s = 0, the previous iterate, if no
    # tried step was finite) and refresh f, unless we are already there.
    target_s = isfinite(best_norm) ? best_s : zero(best_s)
    if s != target_s
        x .+= (s - target_s) * Δx
        isnothing(prepare_for_f!) || prepare_for_f!(x)
        f!(f, x)
    end
    return nothing
end
