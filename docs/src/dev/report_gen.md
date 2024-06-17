# Verifying Correctness

The `ARKAlgorithm` supports problems that specify any combination of the following: an implicit tendency `T_imp!`, an explicit tendency `T_exp!`, a limited tendency `T_lim!`, a function `dss!` that applies a direct stiffness summation, and a function `lim!` that applies a monotonicity-preserving limiter.

## Convergence without a Limiter

In order to verify the correctness of our algorithms without a limiter, we compute their convergence orders for a variety of test cases. For each case, we estimate the convergence order of the algorithm over a range of stable timesteps, ensuring that the estimate `computed_order ± order_uncertainty` satisfies `|computed_order - predicted_order| ≤ order_uncertainty` and `order_uncertainty ≤ predicted_order / 10`. We also generate a plot that shows each algorithm's convergence as the timestep is reduced, along with plots that show the norms of each algorithm's solution and error over time (for some stable timestep). In addition, we verify that `SSP` algorithms produce the same results (up to floating-point roundoff error) when run in `Unconstrained` mode (at least, when run without a limiter).

By [Godunov's theorem](https://en.wikipedia.org/wiki/Godunov%27s_theorem), the use of a monotonicity-preserving limiter reduces the convergence order of any algorithm to 1, so we do not include any test cases that use `T_lim!` and `lim!`.

The test cases we use for this analysis are:
    - `ark_analytic`, which uses a nonlinear `T_exp!` and a linear `T_imp!`
    - `ark_analytic_sys` and `ark_onewaycouple_mri`, which use a linear `T_imp!`
    - `ark_analytic_nonlin`, which uses a nonlinear `T_imp!`
    - `1d_heat_equation`, which uses a nonlinear `T_exp!` implemented with `ClimaCore`
    - `2d_heat_equation`, which uses a nonlinear `T_exp!` and `dss!` implemented with `ClimaCore`
    - `1d_heat_equation_implicit`, which uses a nonlinear `T_imp!` implemented with `ClimaCore`

```@example
include("report_gen.jl")
```
 ![](output/convergence_ark_analytic_all_algorithms.png)
 ![](output/convergence_ark_analytic_sys_all_algorithms.png)
 ![](output/convergence_ark_onewaycouple_mri_all_algorithms.png)
 ![](output/convergence_ark_analytic_nonlin_all_algorithms.png)
 ![](output/convergence_1d_heat_equation_all_algorithms.png)
 ![](output/convergence_2d_heat_equation_all_algorithms.png)
 ![](output/convergence_1d_heat_equation_implicit_all_algorithms.png)

## Errors with a Limiter

In order to verify the correctness of our algorithms with a limiter, we recreate Table 1 from ["Optimization-based limiters for the spectral element method" by Guba et al.](https://www.sciencedirect.com/science/article/pii/S0021999114001491) This involves running the `horizontal_deformational_flow` test case (from ["A standard test case suite for two-dimensional linear transport on the sphere" by Lauritzen et al.](https://gmd.copernicus.org/articles/5/887/2012/gmd-5-887-2012.pdf)) with and without a limiter, and also with and without hyperdiffusion. This test case uses a limited tendency `T_lim!` (which consists of advection and, optionally, hyperdiffusion), along with `dss!` and `lim!`. The spatial discretization is implemented using `ClimaCore`. Since this analysis is relatively expensive to run, we only check the results for `SSP333` and `ARS343`. Note that it is possible to limit undershoots and overshoots to 0 (up to floating-point roundoff error) when using the `SSP` `SSP333`, but not when using the `Unconstrained` `ARS343`.

```@example
using ClimaTimeSteppers # hide
ENV["GKSwstype"] = "nul" # hide
include(joinpath(@__DIR__, "..", "plotting_utils.jl")) # hide
include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl")) # hide

# This also runs with num_steps = 1000, but with larger under/overshoots; 4800
# is the value used in the paper.
limiter_summary(Float64, [SSP333(), ARS343()], horizontal_deformational_flow_test, 4800)
```

Plots of the tracer specific humidities that were used to compute this table are shown below.
 ![](output/horizontal_deformational_flow_limiter_summary_SSP333.png)
 ![](output/horizontal_deformational_flow_limiter_summary_ARS343.png)

## References

 - [Example Programs for ARK ode (SUNDIALS)](http://runge.math.smu.edu/ARKode_example.pdf)
 - ["Optimization-based limiters for the spectral element method" by Guba et al.](https://www.sciencedirect.com/science/article/pii/S0021999114001491)
 - ["A standard test case suite for two-dimensional linear transport on the sphere" by Lauritzen et al.](https://gmd.copernicus.org/articles/5/887/2012/gmd-5-887-2012.pdf)
