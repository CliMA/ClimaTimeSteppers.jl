# Developer Guide

This page documents developer tools for generating convergence reports and
testing algorithm correctness beyond the standard CI test suite.

## Convergence Report Scripts

The scripts in `docs/src/dev/` generate detailed, per-algorithm convergence
analyses. During the documentation build, these run automatically and produce
the summary plots shown on the
[Convergence](../algorithm_properties/convergence.md) page. You can also
run them locally for debugging or development.

### Single algorithm

```bash
julia --project=docs -e '
    using ClimaTimeSteppers
    include("docs/src/dev/report_gen_alg.jl")
' -- --alg ARS343
```

This produces a `.jld2` file in `output/` containing errors at each tested
timestep for every test case (analytic nonlinear, analytic system, 1D/2D heat,
stiff linear, etc.).

### All algorithms

```bash
julia --project=docs -e '
    using ClimaTimeSteppers
    include("docs/src/dev/compute_convergence.jl")
    include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl"))
    using InteractiveUtils: subtypes
    function all_subtypes(T)
        result = Type[]
        for s in subtypes(T)
            isabstracttype(s) ? append!(result, all_subtypes(s)) : push!(result, s)
        end
        result
    end
    for alg_type in all_subtypes(AbstractAlgorithmName)
        empty!(ARGS)
        push!(ARGS, "--alg", string(nameof(alg_type)))
        include("docs/src/dev/report_gen_alg.jl")
    end
    include("docs/src/dev/summarize_convergence.jl")
'
```

### Summary plots

After running one or more algorithms, generate the combined plots:

```bash
julia --project=docs -e 'include("docs/src/dev/summarize_convergence.jl")'
```

The resulting PNGs in `output/` contain multi-panel convergence analyses: error
vs. ``\Delta t`` (log-log), RMS solution norm, and RMS error over time, with
all algorithms overlaid.

### Script overview

| Script | Purpose |
|--------|---------|
| `compute_convergence.jl` | Shared infrastructure: imports test problems, defines the doc-specific `test_convergence!` wrapper |
| `report_gen_alg.jl` | Runs convergence tests for a single algorithm (specified via `--alg` CLI flag) and saves results to JLD2 |
| `summarize_convergence.jl` | Reads all JLD2 files in `output/` and generates combined PNG plots |

## Limiter Analysis

`limiter_analysis.jl` provides an in-depth analysis of the `T_lim!`/`lim!`
code path by recreating Table 1 from [GTS2014](@cite). It runs the
`horizontal_deformational_flow` test case from [LNSR2012](@cite) with and
without a limiter and with and without hyperdiffusion, using `ClimaCore` for
the spatial discretization.

Key result: the SSP method `SSP333` limits undershoots and overshoots to
zero (up to floating-point roundoff), while the unconstrained method `ARS343`
cannot — consistent with the
[theoretical analysis](../algorithm_formulations/ode_solvers.md#Adding-a-Limiter).

This analysis is **not** part of CI (it depends on `ClimaCore` and is
relatively expensive). Run it manually:

```bash
julia --project=docs -e 'include("docs/src/dev/limiter_analysis.jl")'
julia --project=docs -e 'include("docs/src/dev/limiter_summary.jl")'
```
