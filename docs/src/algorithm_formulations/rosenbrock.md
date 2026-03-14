# Rosenbrock methods

In this page, we introduce Rosenbrock-type methods to solve ordinary
differential equations. In doing so, we roughly follow Chapter IV.7 of "Solving
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
diagonally implicit Runge-Kutta scheme (see [Standard IMEX ARK](ode_solvers.md#Standard-IMEX-ARK)). In an implicit
Runge-Kutta method with $s$ stages and tableau $a, b, c$, the updated value
$u_1$ for a step of size $\Delta t$ is obtained starting from the known value $u_0$
with

```math
u_1 = u_0 + \sum_{i=1}^s b_i k_i\,,
```

where the stage increments satisfy

```math
k_i = \Delta t T\!\left( u_0 + \sum_{j=1}^{i-1}\alpha_{ij}k_j + \alpha_{ii} k_i \right)\,,
```

and $\alpha_{ij}$, $b_i$ are carefully chosen tableau coefficients.

Rosenbrock's idea consists in linearizing $T$ around $g_i$ (the explicit part of the
stage value, built from the previously computed increments). The implicit
DIRK stage equation can be written as $k_i = \Delta t\, T(g_i + \alpha_{ii} k_i)$,
which is nonlinear in $k_i$. Applying a first-order Taylor expansion
$T(g_i + \alpha_{ii} k_i) \approx T(g_i) + T'(u_0)\, \alpha_{ii} k_i$ — using
the Jacobian $J = T'(u_0)$ frozen at $u_0$ rather than re-evaluated at each
stage — replaces the nonlinear solve with a single linear system:

```math
k_i = \Delta t T(g_i) + \Delta t J \alpha_{ii} k_i\,,
```

where $J = T'(u_0)$ is the Jacobian of the tendency evaluated at $u_0$, and

```math
g_i = u_0 + \sum_{j=1}^{i-1}\alpha_{ij} k_j.
```

In Rosenbrock-type methods, the per-stage Jacobian $J(g_i)$ is replaced with
a fixed reference $J = T'(u_0)$, evaluated once at the beginning of each
timestep. Each stage then approximates:
```math
T'(g_i) \alpha_{ii} k_i \approx J \sum_{j=1}^{i-1}\gamma_{ij} k_j + J \gamma_{ii} k_i\,,
```
with $\gamma_{ij}$ additional tableau coefficients.

Now, each stage consists of solving a system of linear equations in $k_i$ with
matrix $I - J \Delta t \gamma_{ii}$:
```math
(I - J \Delta t \gamma_{ii}) k_i = \Delta t\, T(g_i) + \Delta t\, J \sum_{j=1}^{i-1}\gamma_{ij} k_j\,,
```
for each $i = 1, \dots, s$. Once $k_i$ are known, $u_1$ can be easily computed and the process repeated for
the next step.

In practice, there are computational advantages in implementing a slight
variation of what is presented here.

Let us define $\widetilde{k}_{i} = \sum_{j=1}^{i} \gamma_{ij} k_j$. If the matrix
$\gamma_{ij}$ is invertible, we can move freely from $k_i$ to $\widetilde{k}_i$. A
convenient step is to also define $C$ as

```math
C = \operatorname{diag}(\gamma_{11}^{-1}, \gamma_{22}^{-1}, \dots, \gamma_{ss}^{-1}) - \Gamma^{-1}\,,
```

which gives

```math
k_i = \frac{\widetilde{k}_i}{\gamma_{ii}} - \sum_{j=1}^{i -1} c_{ij} \widetilde{k}_j\,.
```
Substituting this, we obtain the following equations:

```math
(J \Delta t \gamma_{ii} - I) \widetilde{k}_i = - \Delta t\, \gamma_{ii}\, T(g_i) - \gamma_{ii} \sum_{j=1}^{i-1}c_{ij} \widetilde{k}_j \,,
```

```math
g_i =  u_0 + \sum_{j=1}^{i-1}a_{ij} \widetilde{k}_j \,,
```

```math
u_1 = u_0 + \sum_{j=1}^{s} m_j \widetilde{k}_j\,,
```
with
```math
(a_{ij}) = (\alpha_{ij}) \Gamma^{-1}\,,
```
and
```math
(m_i) = (b_i) \Gamma^{-1}\,.
```

Finally, small changes are required to add support for the explicit time derivative:

```math
(J \Delta t \gamma_{ii} - I) \widetilde{k}_i = - \Delta t\, \gamma_{ii}\, T( t_0 + \alpha_i \Delta t,  g_i) - \gamma_{ii} \sum_{j=1}^{i-1}c_{ij} \widetilde{k}_j - (\Delta t)^2 \gamma_{ii} \gamma_i \frac{\partial T}{\partial t}(t_0, u_0)\,,
```

where we defined

```math
\alpha_{i} = \sum_{j=1}^{i-1}\alpha_{ij}\,,
```

and

```math
\gamma _{i} = \sum_{j=1}^{i}\gamma_{ij}.
```

**Sign convention.** The system solved at each stage keeps the prefactor $(J \Delta t \gamma_{ii} - I)$ — the matrix known as `Wfact` in `DifferentialEquations.jl`. This sign convention is used throughout `ClimaTimeSteppers.jl` for consistency with that ecosystem.

## Implementation

In `ClimaTimeSteppers.jl`, we implement the last equation presented in the
previous section. Currently, the only Rosenbrock-type algorithm implemented is
`SSPKnoth`. `SSPKnoth` is 3-stage, second-order accurate Rosenbrock with

```math
\alpha = \begin{bmatrix}
    0 & 0 & 0 \\
    1 & 0 & 0 \\
    \frac{1}{4} & \frac{1}{4} & 0 \\
    \end{bmatrix}\,,
```

```math
\Gamma = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    -\frac{3}{4} & -\frac{3}{4} & 1 \\
    \end{bmatrix}\,,
```

and

```math
b = \left(\tfrac{1}{6},\, \tfrac{1}{6},\, \tfrac{2}{3}\right)\,.
```

At each stage, the state $g_i$ is computed as a linear combination of the previous stage increments $\widetilde{k}_j$. At stage $i > 1$, DSS and cache updates are applied to $g_i$ before evaluating tendencies, ensuring the state is continuous and consistent with the cache. The first stage is identical to the end of the previous timestep (which was already DSS'd), so DSS is skipped there. After all stages are complete, the final state $u$ is updated as $u \mathrel{+}= \sum_i m_i \widetilde{k}_i$ and DSS is applied once more.
