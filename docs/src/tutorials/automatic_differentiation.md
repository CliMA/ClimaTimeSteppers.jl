# Automatic Differentiation

ClimaTimeSteppers is fully compatible with
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). Dual numbers
propagate correctly through all solver families — explicit RK, IMEX ARK,
and Rosenbrock — enabling differentiation of the ODE solution with respect
to initial conditions or parameters.

## Differentiating with respect to a parameter

Consider the decay ODE ``du/dt = -\lambda u`` with exact solution
``u(T) = u_0 e^{-\lambda T}``. We can compute ``du/d\lambda`` at
``\lambda = 1`` by wrapping the solve in a function:

```@example ad
using ClimaTimeSteppers
import ClimaTimeSteppers as CTS
using ForwardDiff

function solve_decay(λ)
    f = ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -p[1] .* u))

    # Construct problem with types matching λ (required for dual propagation)
    prob = CTS.ODEProblem(f, [one(λ)], (zero(λ), one(λ)), [λ])
    sol = CTS.solve(prob, ExplicitAlgorithm(RK4()); dt = oftype(λ, 0.1), saveat = (one(λ),))
    return sol.u[end][1]
end

λ = 1.0
value    = solve_decay(λ)
gradient = ForwardDiff.derivative(solve_decay, λ)

println("u(1)     = ", round(value;    digits = 6), "   (exact: ", round(exp(-1); digits = 6), ")")
println("du/dλ    = ", round(gradient;  digits = 6), "   (exact: ", round(-exp(-1); digits = 6), ")")
```

The key requirement: when differentiating with respect to a variable, all
quantities derived from it (`u0`, `tspan`, `dt`, `jac_prototype`) must be
constructed with matching type (e.g. `one(λ)` instead of `1.0`) so that
ForwardDiff's dual numbers are not accidentally clipped by the solver.

## IMEX methods

Implicit solvers work the same way. The Jacobian prototype must also be
typed to match:

```@example ad
using LinearAlgebra

function solve_imex(λ)
    T = typeof(λ)

    T_imp! = (du, u, p, t) -> (du .= -p[1] .* u)
    function Wfact!(W, u, p, dtγ, t)
        fill!(W, zero(eltype(W)))
        for i in axes(W, 1)
            W[i, i] = -p[1] * dtγ - 1
        end
    end

    imp = CTS.ODEFunction(T_imp!; jac_prototype = zeros(T, 1, 1), Wfact = Wfact!)
    f = ClimaODEFunction(;
        T_exp! = (du, u, p, t) -> (du .= zero(u)),
        T_imp! = imp,
    )

    prob = CTS.ODEProblem(f, [one(T)], (zero(T), one(T)), [λ])
    alg = IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 1, update_j = UpdateEvery(NewTimeStep)))
    sol = CTS.solve(prob, alg; dt = T(0.1), saveat = (one(T),))
    return sol.u[end][1]
end

value_imex = solve_imex(1.0)
grad_imex  = ForwardDiff.derivative(solve_imex, 1.0)

println("IMEX u(1)  = ", round(value_imex; digits = 6), "   (exact: ", round(exp(-1); digits = 6), ")")
println("IMEX du/dλ = ", round(grad_imex;  digits = 6), "   (exact: ", round(-exp(-1); digits = 6), ")")
```

## Rosenbrock methods

Rosenbrock solvers use a direct linear solve (`lu` factorization) instead of
Newton iteration, and dual numbers propagate through this path as well:

```@example ad
function solve_rosenbrock(λ)
    T = typeof(λ)

    T_imp! = (du, u, p, t) -> (du .= -p[1] .* u)
    function Wfact!(W, u, p, dtγ, t)
        fill!(W, zero(eltype(W)))
        for i in axes(W, 1)
            W[i, i] = -p[1] * dtγ - 1
        end
    end

    imp = CTS.ODEFunction(T_imp!; jac_prototype = zeros(T, 1, 1), Wfact = Wfact!)
    f = ClimaODEFunction(; T_imp! = imp)

    prob = CTS.ODEProblem(f, [one(T)], (zero(T), one(T)), [λ])
    alg = CTS.RosenbrockAlgorithm(CTS.tableau(SSPKnoth()))
    sol = CTS.solve(prob, alg; dt = T(0.1), saveat = (one(T),))
    return sol.u[end][1]
end

value_rb = solve_rosenbrock(1.0)
grad_rb  = ForwardDiff.derivative(solve_rosenbrock, 1.0)

println("Rosenbrock u(1)  = ", round(value_rb; digits = 6), "   (exact: ", round(exp(-1); digits = 6), ")")
println("Rosenbrock du/dλ = ", round(grad_rb;  digits = 6), "   (exact: ", round(-exp(-1); digits = 6), ")")
```

## Tips

- **Type your allocations**: use `zeros(typeof(λ), n, n)` for `jac_prototype`,
  `[one(λ)]` for `u0`, and `oftype(λ, 0.1)` for `dt`.
  **Propagate your types**: use `zeros(typeof(λ), n, n)` for `jac_prototype`,
- **Construct a fresh `ODEProblem` each iteration** if you call `solve` in a
  loop — the integrator mutates `u0` in place.
- **Rosenbrock methods** work as shown above — the `lu` factorization and
  `ldiv!` both handle dual-valued matrices and right-hand sides.
