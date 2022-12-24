# Algorithm comparisons

In this section, we compare the solution errors and convergence orders for several test problems and algorithms.

```@example
include("algo_comparisons.jl")
```

Plots for `ark_analytic`:
![](output/solutions_ark_analytic_imex_ark.png)
![](output/errors_ark_analytic_imex_ark.png)
![](output/orders_ark_analytic_imex_ark.png)

Plots for `ark_analytic_nonlin`:
![](output/solutions_ark_analytic_nonlin_imex_ark.png)
![](output/errors_ark_analytic_nonlin_imex_ark.png)
![](output/orders_ark_analytic_nonlin_imex_ark.png)

Plots for `ark_analytic_sys`:
![](output/solutions_ark_analytic_sys_imex_ark.png)
![](output/errors_ark_analytic_sys_imex_ark.png)
![](output/orders_ark_analytic_sys_imex_ark.png)

## References

 - [Example Programs for ARK ode (SUNDIALS)](http://runge.math.smu.edu/ARKode_example.pdf)
