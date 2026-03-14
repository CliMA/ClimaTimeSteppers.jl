# Low-Storage Runge--Kutta Methods

Low-storage Runge--Kutta (LSRK) methods are a class of explicit Runge--Kutta methods designed to minimize memory usage. The $2N$-storage family stores only two state-sized arrays at any time — the current state $U^{(i)}$ and the accumulated increment $dU^{(i)}$ — rather than the $s + 1$ arrays required by a general $s$-stage explicit RK method. The $2N$-storage format was originally introduced by [Williamson (1980)](@cite Williamson1980), and the specific 4th-order families used here were developed by [Carpenter and Kennedy (1994)](@cite CK1994) and later extended with optimized stability regions by [Niegemann et al. (2012)](@cite NDB2012).

## Formulation

Each timestep begins with $U^{(1)} = u^n$ and proceeds through $s$ stage updates of the form
```math
\begin{aligned}
dU^{(i)} &= f(U^{(i)},\, t + c_i \Delta t) + A_i\, dU^{(i-1)},\\
U^{(i+1)} &= U^{(i)} + \Delta t\, B_i\, dU^{(i)},
\end{aligned}
```
where $A_1 = c_1 = 0$ (so the first accumulation reduces to $dU^{(1)} = f(u^n, t)$). The solution at the next timestep is $u^{n+1} = U^{(s+1)}$.

Because each stage only needs $U^{(i)}$ and $dU^{(i-1)}$, the update requires only two state-sized registers (provided $f$ can be evaluated in incrementing form, i.e., `f(du, u, p, t, α, β)` computes `du = α * f(u,p,t) + β * du`).

## Equivalent Butcher Tableau

The LSRK$(A, B, c)$ scheme can be expressed as a standard explicit Runge--Kutta method with Butcher tableau coefficients defined by the backward recurrence (in $j$):
```math
\begin{aligned}
a_{i,i} &= 0, \\
a_{i,j} &= B_j + A_{j+1}\, a_{i,j+1} \quad (j < i),\\
b_i &= B_i + A_{i+1}\, b_{i+1},\quad b_s = B_s,
\end{aligned}
```
or equivalently by the forward recurrence (in $i$):
```math
\begin{aligned}
a_{j,j} &= 0, \\
a_{i,j} &= a_{i-1,j} + B_{i-1} \prod_{k=j+1}^{i-1} A_k \quad (i > j),
\end{aligned}
```
with $b_j$ treated as $a_{s+1,j}$.

## Implemented Algorithms

The following LSRK algorithms are implemented in `ClimaTimeSteppers.jl` as subtypes of `LowStorageRungeKutta2N`:

| Algorithm | Order | Stages | Reference |
|:---|:---|:---|:---|
| `LSRK54CarpenterKennedy` | 4 | 5 | [Carpenter and Kennedy (1994)](@cite CK1994), Solution 3 |
| `LSRK144NiegemannDiehlBusch` | 4 | 14 | [Niegemann et al. (2012)](@cite NDB2012) |
| `LSRKEulerMethod` | 1 | 1 | Forward Euler (for debugging) |

`LSRK54CarpenterKennedy` is a compact, widely used 4th-order method with only 5 stages. `LSRK144NiegemannDiehlBusch` uses 14 stages to achieve a much larger explicit stability region, making it well suited for hyperbolic problems where the maximum stable timestep is the limiting factor.

## Usage with Multirate Methods

The `LowStorageRungeKutta2N` family can also serve as the *outer solver* in a multirate Runge--Kutta scheme (see [Multirate Runge--Kutta](mrrk.md)). The incremental form of each stage maps naturally onto the slow-tendency accumulation needed by the MRRK framework.
