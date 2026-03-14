# Developer Guide

This section describes tools available in `docs/src/dev/` for generating convergence reports and testing algorithm correctness beyond the standard CI suite.

## Running Convergence Reports

The scripts in `docs/src/dev/` allow you to generate detailed, per-algorithm convergence analyses locally. These complement the [Algorithm Convergence Order](../algorithm_properties/convergence.md) page, which describes the test suite and shows the summary plot generated during the CI documentation build.

To run convergence analysis for a specific algorithm:

```bash
julia --project=docs -e '
    using ClimaTimeSteppers
    include("docs/src/dev/report_gen_alg.jl")
' -- --alg ARS343
```

This produces a `.jld2` file in `output/` containing errors at each tested timestep. To generate summary plots for all previously computed algorithms:

```bash
julia --project=docs -e 'include("docs/src/dev/summarize_convergence.jl")'
```

To run all algorithms in sequence:

```julia
include("docs/src/dev/compute_convergence.jl")
for alg_name in all_subtypes(AbstractAlgorithmName)
    empty!(ARGS)
    push!(ARGS, "--alg", string(alg_name))
    include("docs/src/dev/report_gen_alg.jl")
end
include("docs/src/dev/summarize_convergence.jl")
```

The resulting PNG files in `output/` contain multi-panel convergence analyses (one per test case). These plots include all algorithms overlaid and show error vs. $\Delta t$ (log-log), RMS solution norm, and RMS error over time.

## Limiter Analysis

`docs/src/dev/limiter_analysis.jl` provides an in-depth analysis of the `T_lim!`/`lim!` code path, based on the `horizontal_deformational_flow` test from Lauritzen et al. (2012). This is a ClimaCore-dependent analysis that is not part of the standard CI build. See `docs/src/dev/limiter_summary.jl` for the plot generation script.

## Algorithm Type Hierarchy

The algorithm and tableau type trees are listed in the [Types](types.md) page.
