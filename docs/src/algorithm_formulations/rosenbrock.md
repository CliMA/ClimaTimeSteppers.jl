# Rosenbrock methods

In this page, we introduce Rosenbrock-type methods to solve ordinary
differential equations. In doing do, we roughly follow Chapter IV.7 of "Solving
Ordinary Differential Equations II" ([HW1996](@cite)). (Beware, notation is
not the same). In this spirit, let us introduce the need for Rosenbrock-type
methods in the same way as [HW1996](@cite), by quoting Rosenbrock himself:

> When the functions are non-linear, implicit equations can in general be solved
> only by iteration. This is a severe drawback, as it adds to the problem of
> stability, that of convergence of the iterative process. An alternative, which
> avoids this difficulty, is ...

Rosenbrock method!

Before reading this page, we recommend reading the page on ODE solvers first
([ODE Solvers](ode_solvers.md)).

## Introduction to the formalism

Let us consider an ordinary differential equation of the form

```math
\frac{d}{dt}u(t) = T(u(t), t)\,,
```

where $u$ is the state, $T$ is the tendency,
and $t$ the time. For the sake of simplicity, let us ignore the explicit time
dependency in the tendency (we will get back to that at the end).

The simplest way to introduce the Rosenbrock method is to start from a
diagonally implicit Runge-Kutta scheme (see page on DIRK). In an implicit
Runge-Kutta method with $s$ stages and tableau $a, b, c$, the updated value
$u_1$ for a step of size $\Delta t$ is obtained starting from the known value $u_0$
with

```math
u_1 = u_0 + \sum_{i=1}^s b_i k_i\,,
```
with
```math
k_i = \Delta t  T ( u_0 + \sum_{j=1}^{i-1}\alpha_{ij}k_j + \alpha_{ii} k_i)\,.
```
$\alpha_{ij}$, $b_i$ are carefully chosen coefficients.

Rosenbrock's idea consists in linearizing $T$ around $g_i$. Rather than
solving the resulting nonlinear system iteratively, this is equivalent to
taking a single Newton step with the Jacobian frozen at $u_0$ (rather than
at $g_i$), yielding

```math
k_i = \Delta t T(g_i) + \Delta t J \alpha_{ii} k_i
```

where $J = T'(u_0)$ is the Jacobian of the tendency evaluated at $u_0$, and

```math
g_i = u_0 + \sum_{j=1}^{i-1}\alpha_{ij} k_j
```

In Rosenbrock-type methods, the per-stage Jacobian $J(g_i)$ is replaced with
a fixed reference $J = T'(u_0)$, evaluated once at the beginning of each
timestep. Each stage then approximates:
```math
T'(g_i) \alpha_{ii} k_i \approx J \sum_{j=1}^{i-1}\gamma_{ij} k_j + J \gamma_{ii} k_i\,,
```
with $\gamma_{ij}$ additional tableau coefficients.

Now, each stage consists of solving a system of linear equations in $k_i$ with
matrix $I - \Delta t \gamma_{ii}$:
```math
(I - J \Delta t \gamma_{ii}) k_i = \Delta
t T(g_i) + J \sum_{j=1}^{i-1}\gamma_{ij} k_j
```
for each $i$ in $1, \dots, s$. Once $k_i$ are known, $u_1$ can is easily computed and the process repeated for
the next step.

In practice, there are computational advantages at implementing a slight
variation of what presented here.

Let us define $\tilde{k}_{i} = \sum_{j=1}^{i} \gamma_{ij} k_j$. If the matrix
$\gamma_{ij}$ is invertible, we can move freely from $k_i$ to $\tilde{k}_i$. A
convenient step is to also define $C$, as

```math
C = diag(\gamma_{11}^{-1}, \gamma_{22}^{-1}, \dots, \gamma_{ss}^{-1}) - \Gamma^{-1}
```

Which establishes that

```math
k_i = \frac{\tilde{k}_i}{\gamma_{ii}} - \sum_{j=1}^{i -1} c_{ij} \tilde{k}_j
```
Substituting this, we obtain the following equations

```math
(J \Delta t \gamma_{ii} - 1) \tilde{k}_i = - \Delta
t \gamma_{ii} T(g_i) - \gamma_{ii} J \sum_{j=1}^{i-1}c_{ij} \tilde{k}_j \,,
```

```math
g_i =  u_0 + \sum_{j=1}^{i-1}a_{ij} \tilde{k}_j \,,
```

```math
u_1 = u_0 + \sum_{j=1}^{s} m_j \tilde{k}_j\,,
```
with
```math
(a_{ij}) = (\alpha_{ij}) \Gamma^{-1}\,,
```
and
```math
(m_i) = (b_i) \Gamma^{-1}
```

Finally, small changes are required to add support for explicit time derivative:

```math
(J \Delta t \gamma_{ii} - 1) \tilde{k}_i = - \Delta
t \gamma_{ii} T( t_0 + \alpha_i \Delta t,  g_i) - \gamma_{ii} J \sum_{j=1}^{i-1}c_{ij} \tilde{k}_j - \Delta
t \gamma_{ii} \gamma_i \Delta
t \frac{\partial T}{\partial t}(t_0, u_0)
```

where we defined

$\alpha_{i} = \sum_{j=1}^{i-1}\alpha_{ij} $

$ \gamma _{i} = \sum_{j=1}^{i}\gamma_{ij} $

> **Sign convention.** The system solved at each stage keeps the prefactor $(J \Delta t \gamma_{ii} - I)$ — the matrix known as `Wfact` in `DifferentialEquations.jl`. This sign convention is used throughout `ClimaTimeSteppers.jl` for consistency with that ecosystem.

## Implementation

In `ClimaTimeSteppers.jl`, we implement the last equation presented in the
previous section. Currently, the only Rosenbrock-type algorithm implemented is
`SSPKnoth`. `SSPKnoth` is 3-stage, second-order accurate Rosenbrock with

```math
\alpha = \begin{bmatrix}
    0 & 0 & 0 \\
    1 & 0 & 0 \\
    \frac{1}{4} & \frac{1}{4} & 0 \\
    \end{bmatrix}
```

```math
\Gamma = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    -\frac{3}{4} & -\frac{3}{4} & 1 \\
    \end{bmatrix}
```

and
```math
b = (\frac{1}{6}, \frac{1}{6}, \frac{2}{3})
```

At each stage, the state $g_i$ is computed as a linear combination of the previous stage increments $k_j$. At stage $i > 1$, DSS and cache updates are applied to $g_i$ before evaluating tendencies, ensuring the state is continuous and consistent with the cache. The first stage is identical to the end of the previous timestep (which was already DSS'd), so DSS is skipped there. After all stages are complete, the final state $u$ is updated as $u \mathrel{+}= \sum_i m_i k_i$ and DSS is applied once more.
