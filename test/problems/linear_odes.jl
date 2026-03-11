# ============================================================================ #
# Simple scalar / vector ODE problems (used by LSRK convergence tests)
# ============================================================================ #

"""
Single variable linear ODE: du/dt = αu, u₀ = 1/2, α = 1.01.
Solution: u(t) = u₀ exp(αt).
"""
function linear_prob(::Type{FT} = Float64) where {FT}
    ODEProblem(
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* p .* u .+ β .* du),
        ),
        FT[1 / 2],
        (0.0, 1.0),
        1.01,
    )
end

function linear_sol(u0, p, t)
    u0 .* exp(p * t)
end

"""
Two variable rotation ODE: du₁/dt = αu₂, du₂/dt = -αu₁, u₀ = [0,1], α = 2.
Solution: u(t) = [cos(αt) sin(αt); -sin(αt) cos(αt)] u₀.
"""
function sincos_prob()
    ODEProblem(
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) ->
                (du[1] = α * p * u[2] + β * du[1];
                    du[2] = -α * p * u[1] + β * du[2]),
        ),
        [0.0, 1.0],
        (0.0, 1.0),
        2.0,
    )
end

function sincos_sol(u0, p, t)
    s, c = sincos(p * t)
    [c s; -s c] * u0
end
