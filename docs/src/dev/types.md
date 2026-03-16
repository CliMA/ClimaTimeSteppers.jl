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
