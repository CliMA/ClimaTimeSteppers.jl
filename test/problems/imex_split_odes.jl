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
