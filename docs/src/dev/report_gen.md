# Report generator

In this section, we create a report comparing the solution errors and convergence orders for several test problems and algorithms.

```@example
include("report_gen.jl")
```

Plots for `IMEXAlgorithm`:
 ![](output/convergence_ark_analytic_nonlin_imex_algorithms.png)
 ![](output/convergence_ark_analytic_sys_imex_algorithms.png)
 ![](output/convergence_ark_analytic_imex_algorithms.png)
 ![](output/convergence_ark_onewaycouple_mri_imex_algorithms.png)
 ![](output/convergence_1d_heat_equation_imex_algorithms.png)
 ![](output/convergence_2d_heat_equation_imex_algorithms.png)

## References

 - [Example Programs for ARK ode (SUNDIALS)](http://runge.math.smu.edu/ARKode_example.pdf)
