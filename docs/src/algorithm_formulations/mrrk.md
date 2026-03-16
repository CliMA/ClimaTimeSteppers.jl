# Multirate Runge--Kutta Methods

Many atmospheric and climate models involve processes that evolve on very different timescales. Integrating a fast process at the timestep dictated by the slowest process is wasteful; integrating everything at the fast timestep is prohibitive. Multirate methods address this by using different step sizes for different processes.

Consider an ODE split into a *slow* component $f_S$ and a *fast* component $f_F$:
```math
\frac{du}{dt} = f_S(u, t) + f_F(u, t)\,.
```
The slow component can be stepped with a large $\Delta t$, while the fast component is resolved using many small substeps — each of size $\delta t \ll \Delta t$.

## Multirate (Outer + Inner) Framework

`ClimaTimeSteppers.jl` provides a `Multirate(fast, slow)` wrapper that pairs any inner (fast) algorithm with any outer (slow) algorithm:

```julia
alg = Multirate(LSRK54CarpenterKennedy(), WSRK3())
```

At each outer stage, the slow tendency $f_S$ is evaluated and held fixed; the inner solver then integrates $f_F$ over the substep interval $[\tau_{i-1}, \tau_i]$ using however many fast steps are needed to resolve it.

## Sussman–Knoth–Ascher–Wicker (SKAW) Methods

[SKAW2012](@cite) defines stage values $U^{(i)} = v_i(\tau_i)$ as the solution to the *inner* ODE:
```math
\frac{dv_i}{d\tau}
  = \sum_{j=1}^i \frac{a_{ij} - a_{i-1,j}}{c_i - c_{i-1}} f_S(U^{(j)}, \tau_j)
  + f_F(v_i, \tau),
\quad \tau \in [\tau_{i-1}, \tau_i]\,,
```
where $\tau_i = t + \Delta t c_i$ and $v_i(\tau_{i-1}) = U^{(i-1)}$. Integrating the inner ODE with a constant slow source yields:
```math
U^{(i)} = U^{(i-1)} + \Delta t \sum_{j=1}^i (a_{ij} - a_{i-1,j})\, f_S(U^{(j)}, \tau_j)\,.
```
When $c_i = c_{i-1}$, the integration interval shrinks to zero and the stage reduces to a direct algebraic correction.
The final accumulation stage is treated analogously with $a_{N+1,j} = b_j$ and $c_{N+1} = 1$.

### Low-storage outer solver

When the outer solver is a [Low-Storage RK](lsrk.md) method with coefficients $(A, B, c)$, the inner ODE simplifies to:
```math
\frac{dv_i}{d\tau}
  = \frac{B_{i-1}}{c_i - c_{i-1}}\, dU_S^{(i-1)}
  + f_F(v_i, \tau),
\quad \tau \in [\tau_{i-1}, \tau_i]\,,
```
where $dU_S^{(i)}$ is the low-storage accumulated slow increment:
```math
dU_S^{(i)} = f_S(U^{(i)}, \tau_i) + A_i\, dU_S^{(i-1)}\,.
```

## Multirate Infinitesimal Step (MIS) Methods

Multirate Infinitesimal Step (MIS) methods ([WKG2009](@cite), [KW2014](@cite)) generalize the SKAW framework with additional coupling coefficients. For stage $i$, starting from initial condition $v_i(0) = u^n + \sum_{j < i} \alpha_{ij}(U^{(j)} - u^n)$, the inner ODE is:
```math
\frac{dv_i}{d\tau}
  = \sum_{j=1}^{i-1} \frac{\gamma_{ij}}{d_i \Delta t}(U^{(j)} - u^n)
  + \sum_{j=1}^{i-1} \frac{\beta_{ij}}{d_i} f_S(U^{(j)},\, t + \Delta t c_j)
  + f_F\!\left(v_i,\; t + \Delta t \widetilde{c}_i + \frac{c_i - \widetilde{c}_i}{d_i}\tau\right),
\quad \tau \in [0, \Delta t\, d_i]\,,
```
with $U^{(i)} = v_i(\Delta t\, d_i)$. The method is specified by lower-triangular matrices $\alpha$, $\beta$, $\gamma$, from which the derived quantities are:
```math
d_i = \sum_j \beta_{ij}, \qquad
c = (I - \alpha - \gamma)^{-1} d, \qquad
\widetilde{c} = \alpha c\,.
```

### Implemented MIS algorithms

The following MIS methods from [KW2014](@cite) are implemented as subtypes of `MultirateInfinitesimalStep`:

| Algorithm | Stages | Order | Notes |
|:---|:---|:---|:---|
| `MIS2`    | 3 | 2 | Second-order |
| `MIS3C`   | 3 | 3 | Third-order (with coupling) |
| `MIS4`    | 4 | 3 | Four-stage, third-order |
| `MIS4a`   | 4 | 3 | Variant of MIS4; note a corrected coefficient is used (see source) |
| `TVDMISA` | 3 | 2 | Total variation diminishing |
| `TVDMISB` | 3 | 2 | Total variation diminishing (variant B) |

## Wicker--Skamarock Methods

Wicker and Skamarock ([WS1998](@cite), [WS2002](@cite)) define a simpler class of multirate schemes where each stage resets the inner ODE initial condition to $u^n$:
```math
\begin{aligned}
v_i(t) &= u^n\,,\\
\frac{dv_i}{d\tau}
  &= f_S(U^{(i-1)},\, t + \Delta t\, c_{i-1})
  + f_F(v_i, \tau),
\quad \tau \in [t,\, t + \Delta t\, c_i]\,,\\
U^{(i)} &= v_i(t + \Delta t\, c_i)\,.
\end{aligned}
```
This is a special case of MIS with $\alpha = \gamma = 0$ and a strictly lower-triangular $\beta$ matrix whose subdiagonal entries are $\beta_{i,\, i-1} = c_i$ (all other entries zero).

> [!NOTE]
> The code (`mis.jl`) uses a shifted internal convention: it stores `F[i] = f_S(U^{(i-1)})` and
> places the corresponding coefficient on the *diagonal* of the β matrix rather than the
> subdiagonal. This is numerically equivalent to the paper convention above, but means that β
> tableaux stored in the code are shifted by one column relative to the mathematical definition.

### Implemented Wicker--Skamarock algorithms

| Algorithm | Stages | Order | Reference |
|:---|:---|:---|:---|
| `WSRK2` | 2 | 2 | [WS1998](@cite), $c = (0, 1/2)$ |
| `WSRK3` | 3 | 2 (3 for linear problems) | [WS2002](@cite), $c = (0, 1/3, 1/2)$ |