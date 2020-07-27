# Multirate Runge Kutta

Given a problem with two components that operate at two rates:
```math
\frac{du}{dt} = f_F(u,t) + f_S(u,t)
```
where ``f_F`` is the _fast_ component, and ``f_S`` is the _slow_ component.

Given an outer explicit Runge--Kutta scheme with tableau ``(a,b,c)``

We can define the stage values ``U^{(i)} = v_i(\tau_i)`` as the solution to the _inner_ ODE problem
```math
\frac{dv_i}{d\tau} 
  = \sum_{j=1}^i  \frac{a_{ij} - a_{i-1,j}}{c_i - c_{i-1}}  f_S (U^{(j)}, \tau_j)
    + f_F(v_i, \tau),
\quad \tau \in [\tau_{i-1}, \tau]
```
where ``\tau_i = t + \Delta t c_i``, with initial condition ``v_i(\tau_{i-1}) = U^{(i-1)}``. If ``c_i == c_{i-1}``, we can treat it as a correction step:
```math
U^{(i)} = U^{(i-1)} + \Delta t \frac{\sum_{j=1}^i (a_{ij} - a_{i-1,j})}{c_i - c_{i-1}} f_S (U^{(j)}, \tau_i)
```
The final summation stage treating analogously to ``i=N+1``, with ``a_{N+1,j} = b_j`` and ``c_{N+1} = 1``.

## Low-storage

If using a low-storage Runge--Kutta method is used as an outer solver, then this reduces to
```math
\frac{dv_i}{d\tau} 
  =  \frac{B_{i-1}}{c_i - c_{i-1}} dU_S^{(i-1)}
    + f_F(v_i, \tau),
\quad \tau \in [\tau_{i-1}, \tau]
```
where
```math
dU_S^{(i)} = f_S(U^{(i)}, \tau_i) + A_i dU_S^{(i-1)}
```

## References

* Schlegel, M., Knoth, O., Arnold, M., and Wolke, R. (2012) "Implementation of multirate time integration methods for air pollution modelling", _Geosci. Model Dev._, 5, 1395--1405, doi: [10.5194/gmd-5-1395-2012](https://doi.org/10.5194/gmd-5-1395-2012)