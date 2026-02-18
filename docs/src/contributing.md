# Contributing

Thank you for contributing to `ClimaTimeSteppers`! We encourage Pull Requests (PRs).
Please do not hesitate to ask questions, or to open issues if something seems amiss
or you'd like a new feature.

## Some useful tips

- Start by [forking the repository](https://github.com/CliMA/ClimaTimeSteppers.jl/fork)
  on GitHub, then clone your fork locally:
  ```sh
  git clone https://github.com/YOUR_USERNAME/ClimaTimeSteppers.jl.git
  cd ClimaTimeSteppers.jl
  git remote add upstream https://github.com/CliMA/ClimaTimeSteppers.jl.git
  ```
- When developing, work on a branch off of the most recent `main`. Fetch the latest
  changes from upstream first:
  ```sh
  git fetch upstream
  git checkout -b branch_name upstream/main
  ```
- Make sure you add tests for your code in `test/`, appropriate documentation in `docs/`,
  and descriptive inline comments throughout the code.
  All exported functions and structs must have docstrings.
- When your PR is ready for review, clean up your commit history by squashing to 1 commit per PR
  and make sure your code is current with `main` by rebasing against upstream:
  ```sh
  git fetch upstream
  git rebase upstream/main
  ```

## Continuous integration

After rebasing your branch, you can ask for review. Fill out the template and
provide a clear summary of what your PR does. When a PR is created or
updated, a set of automated tests are run on the PR in our continuous
integration (CI) system.

### Formatting check

The `JuliaFormatter` check verifies that the PR is correctly formatted according to
the project's style guidelines (defined in `.JuliaFormatter.toml`).

To format your code, first add JuliaFormatter to your base environment (one-time setup):
```sh
julia -e 'using Pkg; Pkg.add("JuliaFormatter")'
```

Then, in a Julia REPL, run:
```julia
using JuliaFormatter; format(".")
```

### Documentation

The `Documentation` check rebuilds the documentation for the PR and verifies that the
docs are consistent and generate valid output.
