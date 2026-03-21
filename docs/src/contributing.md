# Contributing

Thank you for your interest in contributing to ClimaTimeSteppers.jl!
We welcome pull requests (PRs) of all sizes — from typo fixes to new solver
families. If something seems amiss or you would like to request a feature,
please [open an issue](https://github.com/CliMA/ClimaTimeSteppers.jl/issues/new).

## Getting Started

1. [Fork the repository](https://github.com/CliMA/ClimaTimeSteppers.jl/fork)
   on GitHub, then clone your fork locally:

   ```sh
   git clone https://github.com/YOUR_USERNAME/ClimaTimeSteppers.jl.git
   cd ClimaTimeSteppers.jl
   git remote add upstream https://github.com/CliMA/ClimaTimeSteppers.jl.git
   ```

2. Create a feature branch from the latest `main`:

   ```sh
   git fetch upstream
   git checkout -b my-feature upstream/main
   ```

3. Make your changes. When you are ready for review, rebase against `main`
   and squash into one commit per PR:

   ```sh
   git fetch upstream
   git rebase upstream/main
   ```

## Code Guidelines

- Add tests for new functionality in `test/` and documentation in `docs/`.
- All exported functions and types must have docstrings.
- Keep commits focused — one logical change per commit.

## Formatting

CI enforces consistent style via [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl)
using the settings in `.JuliaFormatter.toml`. Install the formatter once:

```sh
julia -e 'using Pkg; Pkg.add("JuliaFormatter")'
```

Then format the entire repository before committing:

```julia
using JuliaFormatter; format(".")
```

## Continuous Integration

When a PR is created or updated, the following checks run automatically:

| Check | What it verifies |
|-------|-----------------|
| **Unit tests** | All tests in `test/` pass on supported Julia versions |
| **Formatter** | Code matches the project style (`.JuliaFormatter.toml`) |
| **Documentation** | Docs build successfully with no errors |
| **Downstream** | ClimaAtmos, ClimaLand, and ClimaCoupler integration tests (soft-fail) |

If any check fails, click through to the CI log for details.
