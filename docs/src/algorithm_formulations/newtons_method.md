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
The matrix ``W_i`` is the shifted Jacobian that arises when applying Newton's method to the implicit stage equation of a DIRK method; see [HW1996](@cite), Chapter IV.8, for the analogous matrix ``I - h\gamma J`` in simplified Newton iterations for implicit Runge-Kutta methods.

## Implementation in ClimaTimeSteppers.jl

ClimaTimeSteppers provides [`NewtonsMethod`](@ref) to solve the nonlinear system ``f(x) = 0``. The iterative update at step ``n`` is:

```math
\begin{aligned}
W(x_n)\, \Delta x_n &= f(x_n), \\
x_{n+1} &= x_n - \Delta x_n.
\end{aligned}
```

Note: the sign convention in the code is ``W \Delta x = f`` followed by ``x \mathrel{-}= \Delta x``, which is algebraically equivalent to the more common textbook form ``W \Delta x = -f,\; x \mathrel{+}= \Delta x``.

### Linear Solvers

The linear system ``W \Delta x = f`` can be solved using:
- **Direct solvers**: When the explicit Jacobian ``W`` is available as a matrix (via the user-provided `Wfact`), standard factorization methods (e.g., LU) apply. The matrix is passed as `j_prototype` and must support `ldiv!`.
- **Krylov methods**: Iterative solvers such as GMRES from `Krylov.jl`, which only require matrix-vector products ``W v``. This avoids forming ``W`` explicitly and is essential for large-scale problems. Configured via [`KrylovMethod`](@ref).

### Jacobian-Free Newton-Krylov (JFNK)

When using a Krylov method, we only need the action of the Jacobian on a vector ``v`` (a Jacobian-vector product, JVP), rather than the full Jacobian matrix. ClimaTimeSteppers provides:

- **[`ForwardDiffJVP`](@ref)**: Approximates ``W v`` using a forward finite difference:
  ```math
  W v \approx \frac{f(x + \epsilon v) - f(x)}{\epsilon}
  ```
  The step size ``\epsilon`` is controlled by a [`ForwardDiffStepSize`](@ref) strategy:
  - `ForwardDiffStepSize1()`: ``\epsilon = c \sqrt{\text{eps}} / \|\Delta x\|`` (optimal for forward differences; see derivation in source)
  - `ForwardDiffStepSize2()`: ``\epsilon = \sqrt{\text{eps}(1 + \|x\|)} / \|\Delta x\|`` (NITSOL convention)
  - `ForwardDiffStepSize3()`: same as above but averaged over components of ``x`` (default)

  Users can implement custom subtypes of [`JacobianFreeJVP`](@ref) for alternative JVP approximations.

When both a Jacobian-free JVP and an explicit Jacobian (`j_prototype`) are provided, the Jacobian is used as a *left preconditioner* for the Krylov solver (unless `disable_preconditioner = true`).

### Forcing Strategies (Inexact Newton)

In a Newton-Krylov method, the linear system is solved *inexactly* such that:
```math
\| f(x_n) + W(x_n) \Delta x_n \| \leq \eta_n \| f(x_n) \|
```
where ``\eta_n`` is the *forcing term* (called `rtol` in the code). Available strategies:

- **[`ConstantForcing`](@ref)`(rtol)`**: Uses a fixed tolerance ``\eta = \texttt{rtol} \in [0,1)`` for every iteration. Setting `rtol = 0` recovers quadratic Newton convergence at higher Krylov cost; larger values reduce Krylov work but risk slower nonlinear convergence.
- **[`EisenstatWalkerForcing`](@ref)`()`**: An adaptive strategy from [Eisenstat and Walker (1996)](http://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR94463.pdf) ("Choice 2") that automatically tightens ``\eta_n`` as Newton converges. Tunable parameters: `initial_rtol`, `γ`, `α` (convergence order, in ``(1, 2]``), `min_rtol_threshold`, and `max_rtol`.

### Convergence Control

The Newton iteration terminates when one of the following occurs:
1. The [`ConvergenceChecker`](@ref) (if provided) reports convergence based on ``x_n`` and ``\Delta x_n``.
2. The maximum number of iterations `max_iters` is reached.

If no convergence checker is provided, Newton's method always runs for exactly `max_iters` iterations (default: 1).

### Jacobian Update Strategies

The [`update_j`](@ref UpdateSignalHandler) parameter controls how often the Jacobian is recomputed:
- `UpdateEvery(NewNewtonIteration)` — fresh Jacobian every iteration (default; standard Newton)
- `UpdateEvery(NewNewtonSolve)` — reuse across iterations within one `solve_newton!` call (the *chord method*)
- `UpdateEvery(NewTimeStep)` — reuse across multiple solves within a timestep (cheapest, but slowest convergence)

Freezing the Jacobian (chord method or per-timestep) reduces cost at the expense of convergence rate.

### Line Search

When `line_search = true`, a backtracking strategy is applied after each Newton step. If the residual norm ``\|f(x_{n+1})\|`` does not decrease (or becomes `NaN`), the step ``\Delta x_n`` is repeatedly halved (up to 5 times) to find a step that reduces the residual.
