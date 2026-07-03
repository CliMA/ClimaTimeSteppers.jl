# ============================================================================ #
# IMEX split problems (used by multirate convergence tests)
# ============================================================================ #

"""
IMEX problem with autonomous linear part: du/dt = u + cos(t)/α.
u₀ = 1/2, α = 4. Solution: u(t) = (eᵗ + sin(t) - cos(t))/(2α) + u₀ eᵗ.
"""
function imex_autonomous_prob(::Type{ArrayType}) where {ArrayType}
    SplitODEProblem(
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* u .+ β .* du),
        ),
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* cos(t) / p .+ β .* du),
        ),
        ArrayType([0.5]),
        (0.0, 1.0),
        4.0,
    )
end

function imex_autonomous_sol(u0, p, t)
    (exp(t) + sin(t) - cos(t)) / 2p .+ exp(t) .* u0
end

function imex_nonautonomous_prob(::Type{ArrayType}) where {ArrayType}
    SplitODEProblem(
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* cos(t) * u .+ β .* du),
        ),
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* cos(t) / p .+ β .* du),
        ),
        ArrayType([0.5]),
        (0.0, 2.0),
        4.0,
    )
end

function imex_nonautonomous_sol(u0, p, t)
    (exp(sin(t)) .* (1 .+ p .* u0) .- 1) ./ p
end

"""
Split problem whose **slow** tendency depends on the state `u`:
`du/dt = -10u (fast, f1) + -u (slow, f2)`, with `u(t) = u₀ e^{-11t}`.

The other two problems have a state-independent slow tendency (`cos(t)/p`), so a
multirate scheme that corrupts the intermediate stage state still feeds the right
slow forcing and converges anyway. Here the slow forcing `f2(U) = -U` sees the
stage state, so stage-coupling bugs (e.g. an inner integrator that overshoots its
substep) show up as order reduction. Used to guard the WSRK/MIS/LSRK coupling.
"""
function imex_statedep_slow_prob(::Type{ArrayType}) where {ArrayType}
    SplitODEProblem(
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* (-10) .* u .+ β .* du),
        ),
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* (-1) .* u .+ β .* du),
        ),
        ArrayType([1.0]),
        (0.0, 1.0),
        nothing,
    )
end

imex_statedep_slow_sol(u0, p, t) = exp(-11 * t) .* u0
