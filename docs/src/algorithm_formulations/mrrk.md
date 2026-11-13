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

## Step-exchange (split-explicit) methods

The methods above are stage-exchange methods: they evaluate the slow tendency $f_S$ again at every outer stage.
A step-exchange method evaluates $f_S$ only at whole-step states and keeps the result fixed, as the frozen slow forcing, while the fast system sub-cycles over the full step.
Atmospheric dynamical cores use this split-explicit composition, with the expensive physics in the slow tendency and the acoustic and gravity waves in the fast tendency.

Two step-exchange outer methods are provided:

| Algorithm | Slow evaluations per step | Order |
|:---|:---|:---|
| `LieSplitOuter` | 1 | 1 |
| `TrapezoidalSplitOuter` | 2 | 2 |

`LieSplitOuter` freezes $f_S$ at the step start state and sub-cycles the fast system once over $\Delta t$.
`TrapezoidalSplitOuter` freezes $f_S$ at the step start state, sub-cycles to a predicted end state, evaluates $f_S$ again at the predicted end state, averages the two evaluations, and sub-cycles again from the step start state with the averaged forcing.
Before the second sub-cycle, `cache!` runs at the step start state.

With $\Phi^{\Delta t}[f]$ the sub-cycle of the fast system with tendency $f$ over the step $\Delta t$, `LieSplitOuter` is
```math
u^{n+1} = \Phi^{\Delta t}\big[f_F + G\big](u^n)\,, \qquad G = f_S(u^n)\,,
```
and `TrapezoidalSplitOuter` is
```math
\begin{aligned}
G_1 &= f_S(u^n)\,, \qquad G_2 = f_S\Big(\Phi^{\Delta t}\big[f_F + G_1\big](u^n)\Big)\,,\\
u^{n+1} &= \Phi^{\Delta t}\big[f_F + \tfrac{1}{2}(G_1 + G_2)\big](u^n)\,.
\end{aligned}
```

The fast component `f1` of the `SplitODEProblem` is a `ClimaODEFunction`, so the inner algorithm can be an `IMEXAlgorithm`.
The slow component `f2` is `freeze!(G, G_lim, u, p, t)`, which fills the frozen slow forcing pair at the state `u`.
The fast sub-cycle adds `G` to the explicit tendency and `G_lim` to the limited tendency of `f1`.

The caller passes the size of the fast sub-step as `fast_dt`, and the final sub-step is shortened to end at the step end.

### Optional callables

Both outer methods accept three optional callables.
Each callable defaults to `nothing`, and a `nothing` callable makes no call.

The outer implicit complement `complement(u, p, t, dt)` advances `u` in place.
It serves an implicit split in which the inner algorithm solves a part of the implicit operator and the complement solves the remainder once per outer step.
`LieSplitOuter` calls it once over $\Delta t$ before the sub-cycle.
`TrapezoidalSplitOuter` calls it over $\Delta t / 2$ before and after the sub-cycles (Strang splitting), and runs `cache!` at the sub-cycled end state before the second call.
With the complement written as a map $C^{\Delta t}$ of the state, and the step of either method without a complement as $\Psi^{\Delta t}$, `LieSplitOuter` gives $u^{n+1} = \Psi^{\Delta t}\big(C^{\Delta t}(u^n)\big)$ and `TrapezoidalSplitOuter` gives $u^{n+1} = C^{\Delta t/2}\big(\Psi^{\Delta t}(C^{\Delta t/2}(u^n))\big)$.
The caller provides the complement, and the outer method only calls it.

The per-sub-step state update `fast_step!(u, p, t, sub_dt)` advances `u` in place, where `t` is the end time and `sub_dt` the length of the inner step just taken.
`sub_dt` equals `fast_dt`, except for a final sub-step that is shortened to end at the step end.
The sub-cycle calls the update after each inner step, which includes its DSS and its end-of-step `cache_imp!`, and then runs `cache_imp!` again at the updated state.
Inside the sub-cycle, the cache quantities that `cache_imp!` computes are current at the state that the update modifies.
Every other cache quantity dates from the last `cache!` or from the last call of `freeze!`.
The caller must satisfy five requirements on the update:

- It reads from the cache only quantities that `cache_imp!` computes.
- It applies a process that neither `f1` nor the frozen slow forcing contains; it may write a state field that `f1` also advances through other processes.
- It writes neither the frozen slow forcing pair nor any cache quantity that the inner step, `cache_imp!` or `freeze!` reads; it may write a record that nothing reads, such as a diagnostic accumulator.
- Each update is bounded on its own, because neither `lim!` nor `constrain_state!` runs between the update and the next inner step.
- It preserves the continuity that the DSS establishes, for example as a pointwise function of continuous fields, because no DSS runs between the update and the next inner step or the step result.

`TrapezoidalSplitOuter` calls the update in both sub-cycles.
The updates of the first sub-cycle enter only the predicted end state, and only the updates of the second sub-cycle remain in the step result.

The sub-cycle start hook `sub_cycle_start!(u, p, t)` receives the state that a sub-cycle starts from.
The outer method calls it after `freeze!` has evaluated that state and before the first inner step.
The hook lets the fast tendency use a quantity of the start state, for example a reference field that the fast tendency subtracts.
`LieSplitOuter` calls it once per step.
`TrapezoidalSplitOuter` calls it before each of its two sub-cycles, with the step start state both times; before the second call, `cache!` has run at that state.

### Choosing a family

The two families differ in the states at which the fast and slow components exchange information.
A stage-exchange method evaluates the slow tendency at intermediate stage states and integrates the fast system over partial stage intervals.
A step-exchange method evaluates the slow tendency only at whole-step states, and the order of the outer method sets the number of evaluations.
`TrapezoidalSplitOuter` reaches second order with two evaluations per step, against three for the second-order `MIS2`.
Choose stage exchange when the accuracy of the fast and slow components is coupled through the stages.
Choose step exchange when the slow tendency is expensive, or when the composition needs an outer implicit complement.

### Relation to the literature

The stage-exchange family is the multirate infinitesimal step formulation.
A tableau-based implicit-explicit multirate method also exists (IMEX-MRI-GARK, [Chinomona2021](@cite)), and ClimaTimeSteppers does not implement it.
