# Report generator

In this section, we create a report comparing the solution errors and convergence orders for several test problems and algorithms.

```@example
include("report_gen.jl")
```

Plots for `IMEXAlgorithm{ARK}`:
![](output/convergence_ark_analytic_nonlin_imex_ark.png)
![](output/convergence_ark_analytic_sys_imex_ark.png)
![](output/convergence_ark_analytic_imex_ark.png)
![](output/convergence_ark_onewaycouple_mri_imex_ark.png)
![](output/convergence_1d_heat_equation_imex_ark.png)
![](output/convergence_2d_heat_equation_imex_ark.png)

Plots for `IMEXAlgorithm{SSPRK}`:
![](output/convergence_ark_analytic_nonlin_imex_ssprk.png)
![](output/convergence_ark_analytic_sys_imex_ssprk.png)
![](output/convergence_ark_analytic_imex_ssprk.png)
![](output/convergence_ark_onewaycouple_mri_imex_ssprk.png)
![](output/convergence_1d_heat_equation_imex_ssprk.png)
![](output/convergence_2d_heat_equation_imex_ssprk.png)

## References

 - [Example Programs for ARK ode (SUNDIALS)](http://runge.math.smu.edu/ARKode_example.pdf)
