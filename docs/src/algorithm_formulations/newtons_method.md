# Newton's Method

## Mathematical Formulation

In implicit and implicit-explicit (IMEX) ODE solvers, every implicit equation for a stage ``U_i`` has the form ``f_i(U_i) = 0``, where

```math
f_i(x) = R_i + \Delta t a_{i,i} T_{\text{imp}}(x, t_0 + \Delta t c_i) - x.
```

In this function, ``R_i``, ``\Delta t a_{i,i}``, and ``t_0 + \Delta t c_i`` are all quantities that do not depend on ``x``. The Jacobian of this function is

```math
W_i(x) = \frac{d}{dx}f_i(x) = \Delta t a_{i,i} J_{\text{imp}}(x, t_0 + \Delta t c_i) - I,
```

where ``J_{\text{imp}}`` is the Jacobian of the implicit tendency,

```math
J_{\text{imp}}(x, t) = \frac{\partial}{\partial x}T_{\text{imp}}(x, t).
```

The value of ``U_i`` can be computed by running Newton's method with ``f = f_i`` and ``j = W_i``.
Note that "``W``" is used to denote the exact same quantity in [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl/blob/v6.0.0/src/derivative_utils.jl).

## Implementation in ClimaTimeSteppers.jl

ClimaTimeSteppers provides `NewtonsMethod` to solve the nonlinear system ``f(x) = 0``. The iterative update at step ``n`` is defined as:

```math
\begin{aligned}
W(x_n)\, \Delta x_n &= -f(x_n), \\
x_{n+1} &= x_n + \Delta x_n.
\end{aligned}
```

### Linear Solvers

The linear system ``W \Delta x = -f`` can be solved using:
- **Direct solvers**: When the explicit Jacobian ``W`` is available as a matrix (via the user-provided `Wfact!`), standard factorization methods (e.g., LU from `LinearAlgebra`) apply.
- **Krylov methods**: Iterative solvers such as GMRES from `Krylov.jl`, which only require matrix-vector products ``W v``. This avoids forming ``W`` explicitly and is essential for large-scale problems.

### Jacobian-Free Newton-Krylov (JFNK)

When using a Krylov method, we only need the action of the Jacobian on a vector ``v`` (a Jacobian-Vector Product, JVP), rather than the full Jacobian matrix. This allows for essentially "Jacobian-free" methods. ClimaTimeSteppers supports:

- **`ForwardDiffJVP`**: Approximates ``W v`` using a forward finite difference:
  ```math
  W v \approx \frac{f(x + \epsilon v) - f(x)}{\epsilon}
  ```
  The step size ``\epsilon`` is controlled by a `ForwardDiffStepSize` strategy (e.g., `ForwardDiffStepSize1()`, `ForwardDiffStepSize2()`, `ForwardDiffStepSize3()`), which balances truncation and roundoff errors.

Users can implement custom subtypes of `JacobianFreeJVP` to provide alternative JVP approximations (e.g., complex-step or higher-order finite differences).

### Forcing Strategies (Inexact Newton)

In a Newton-Krylov method, solving the linear system ``W \Delta x = -f`` perfectly at every Newton iteration is unnecessary and computationally expensive, especially when ``x_n`` is far from the true root.

Instead, we solve the system *inexactly* such that the residual satisfies:
```math
\| f(x_n) + W(x_n) \Delta x_n \| \leq \eta_n \| f(x_n) \|
```
where ``\eta_n`` is the *forcing term*. ClimaTimeSteppers provides:

- **`ConstantForcing(η)`**: Uses a fixed tolerance ``\eta \in [0,1)`` for every Newton iteration. Setting ``\eta = 0`` (or `eps(FT)`) forces an exact (machine-precision) linear solve and recovers quadratic Newton convergence; larger ``\eta`` risks slower convergence but reduces Krylov work.
- **`EisenstatWalkerForcing()`**: An adaptive strategy from [Eisenstat and Walker (1996)](http://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR94463.pdf) that automatically tightens ``\eta_n`` as the Newton iteration converges, balancing Krylov work against nonlinear progress and avoiding oversolving.

### Convergence and Jacobian Update

The Newton iteration terminates when the residual norm ``\|f(x_n)\|`` falls below a user-specified absolute tolerance, or when a maximum number of iterations (`max_iters`) is reached. The Jacobian ``W`` can optionally be recomputed at each Newton iteration via the `update_j` option; freezing it after the first evaluation (lagged Jacobian) reduces cost at the expense of convergence rate.
