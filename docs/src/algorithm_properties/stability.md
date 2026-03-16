# Stability of IMEX Schemes

Implicit-explicit (IMEX) Additive Runge-Kutta (ARK) schemes are widely used to step non-hydrostatic atmospheric models and other climate model components forward in time. Their primary advantage is the ability to treat stiff terms (such as acoustic and gravity waves) implicitly while treating advection and other slow processes explicitly, thus avoiding the severe explicit stability constraints that would otherwise limit the timestep.

## Courant Number

For a 1D explicit advection equation, the Courant-Friedrichs-Lewy (CFL) stability condition requires that the Courant number $C$ satisfies:
```math
C = c \frac{\Delta t}{\Delta x} \le C_{\max}
```
where $c$ is the fastest wave speed (e.g., the speed of sound $c_s$), $\Delta t$ is the timestep, $\Delta x$ is the grid spacing, and $C_{\max}$ is the theoretical maximum stable Courant number for the given explicit ODE solver.

For non-hydrostatic atmospheric models using explicit timestepping, the fast-moving acoustic gravity waves ($c_s \approx 340$ m/s) severely restrict $\Delta t$. By utilizing an IMEX scheme — such as a Horizontally Explicit, Vertically Implicit (HEVI) splitting — the vertical acoustic wave propagation can be integrated implicitly (bypassing the vertical CFL limit), while horizontal propagation remains explicit.

## Empirical Stability in Atmospheric Models

Theoretical stability regions for IMEX schemes constructed via linear stability analysis serve as a useful baseline, but applying these schemes to the full 3D non-hydrostatic Euler equations often yields empirical limits different from linear theory. 

[Gardner et al. (2018)](@cite GGHRUW2018) investigated the stability and efficiency of several IMEX schemes on a 3D baroclinic wave test case. The test case employed a continuous Galerkin spectral element spatial discretization with the following parameters:
- **Elements:** 2,400 elements
- **Polynomial order:** 4 (which divides each element into $3 \times 3 = 9$ sub-element areas)
- **Speed of sound:** $c_s \approx 340$ m/s

This configuration yields an average horizontal resolution $\Delta \bar{x}$ of approximately 153.7 km:
```math
\Delta \bar{x} \approx \sqrt{\frac{4 \pi R_{\text{earth}}^2}{N_e \times 9}} = \sqrt{\frac{4 \pi (6375 \text{ km})^2}{2400 \times 9}} \approx 153.7 \text{ km}
```

The table below reproduces the maximum stable timestep sizes ($\Delta t_{\max}$) from [Gardner et al. (2018)](@cite GGHRUW2018), Table 2 (baroclinic wave test, HEVI-B splitting), restricted to methods implemented in `ClimaTimeSteppers.jl`. Entries formatted as "Rosenbrock/Newton" indicate that both solvers were tested; "–/X" means the Rosenbrock-like solver was unstable while Newton was stable at X seconds. Empirical horizontal acoustic Courant numbers are derived as $C_h = c_s \frac{\Delta t_{\max}}{\Delta \bar{x}}$.

| Scheme Group | Algorithm | $\Delta t_{\max}$ (s) | $C_h$ |
|:--- |:--- |:--- |:--- |
| **Second Order** | `ARS222` | 160 | $\approx 0.35$ |
| | `ARS232` | 200 | $\approx 0.44$ |
| | `SSP222` | 160 | $\approx 0.35$ |
| | `SSP332` (a/b variants) | 225 | $\approx 0.50$ |
| | `SSP322` | 320 | $\approx 0.71$ |
| **Third Order** | `SSP433` | 200 / 216 | $\approx 0.44$ / $0.48$ |
| | `ARS443` | 300 | $\approx 0.66$ |
| | `ARS233`, `SSP333` (b/c) | –/320 | – / $\approx 0.71$ |
| | `SSP333` (a) | – (unstable) | – |
| | `ARS343` | 450 | $\approx 1.00$ |

*Table: Empirical maximum stable timesteps for IMEX methods implemented in `ClimaTimeSteppers.jl`, from [Gardner et al. (2018)](@cite GGHRUW2018).*

Among the methods tested, `ARS343` achieves the largest maximum stable timestep (450 s, $C_h \approx 1$). IMEX schemes generally recover horizontal acoustic Courant numbers of $C_h \in [0.35, 1.0]$, despite the stiffness of the 3D spherical non-hydrostatic equations. Note that time to solution also depends on the number of stages and the cost of solving the implicit system, which are not captured by $C_h$ alone.
