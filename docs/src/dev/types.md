# Type Hierarchies

The type trees below are generated from the loaded package and reflect the
current state of the code. They are useful for understanding how algorithms,
tableau names, constraints, and ODE function types relate to each other.

```@example types
import AbstractTrees as AT
import InteractiveUtils as IU
import ClimaTimeSteppers as CTS
AT.children(x::Type) = IU.subtypes(x)
nothing # hide
```

## Algorithms

All concrete solvers are subtypes of [`ClimaTimeSteppers.TimeSteppingAlgorithm`](@ref):

```@example types
AT.print_tree(CTS.TimeSteppingAlgorithm)
```

## Algorithm Names (Tableaux)

Each named tableau is a subtype of `AbstractAlgorithmName`:

```@example types
AT.print_tree(CTS.AbstractAlgorithmName)
```

## Algorithm Constraints

```@example types
AT.print_tree(CTS.AbstractAlgorithmConstraint)
```

## ODE Function Types

```@example types
AT.print_tree(CTS.AbstractClimaODEFunction)
```

## Internal Types

### `OffsetODEFunction`

Multirate algorithms (MIS, Wicker-Skamarock) solve an "inner ODE" representing the fast tendency ``f_F`` during each outer stage. The inner problem runs over a substep interval ``\tau \in [\tau_{i-1}, \tau_i]`` with the slow tendency ``f_S`` held fixed as a constant forcing term.

To reuse standard timesteppers for the inner problem without allocating new functions, `ClimaTimeSteppers.jl` uses `OffsetODEFunction`. This wrapper dynamically offsets the inner integration time and adds the constant slow forcing:

```math
f_{\text{offset}}(u, p, \tau) = f_F(u, p, \alpha + \beta \tau) + \gamma \cdot x
```

The parameters ``\alpha, \beta, \gamma, x`` are mutated in-place by the outer multirate solver at the start of each stage. This design achieves zero-allocation multirate stepping.
