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

The update is performed in-place and the final `(x, f)` correspond to
the last step tested.
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

    # if not improved (or NaN), bisect back toward x_old up to 5 times
    i = 1
    α = 0.5
    while ((normf > normf0) || !isfinite(normf)) && (i <= 5)
        x .+= α * Δx    # move back toward x_old 
        isnothing(prepare_for_f!) || prepare_for_f!(x)
        f!(f, x)
        normf = norm(f)
        α /= 2
        i += 1
    end
end
