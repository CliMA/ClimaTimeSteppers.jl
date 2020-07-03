# Additive Runge--Kutta

ARK methods are IMEX (Implicit-Explicit) methods   based on splitting the ODE function ``f(u) = f_L(u) + f_R(t)`` 
where ``f_L(u) = L u`` is a linear operator which is treated implicitly, and ``f_R(u)`` is the remainder to be
treated explicitly. The value on the ``i``th stage ``U^{(i)}`` is
```math
U^{(i)} = u^n + \Delta t \sum_{j=1}^i \tilde a_{ij} f_L(U^{(j)}) 
              + \Delta t \sum_{j=1}^{i-1} a_{ij} f_R(U^{(j)})
```
which can be written as the solution to the linear problem:
```math
W U^{(i)} = \hat U^{(i)}
```
where
```math
\hat U^{(i)} = u^n + \Delta t \sum_{j=1}^{i-1} \tilde a_{ij} f_L(U^{(j)}) 
                                             + \Delta t \sum_{j=1}^{i-1} a_{ij} f_R(U^{(j)})
```
and
```math
W = (I - \Delta t \tilde a_{ii} L)
```

The next step is then defined as
```math
u^{n+1} = u^n + \Delta t \sum_{i=1}^N b_i [ f_L(U^{(i)}) + f_R(U^{(i)}) ]
```

When an iterative solver is used, an initial value ``\bar U^{(i)}`` can be chosen by an explicit approximation
```math
\bar U^{(i)} = u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} [ f_L(U^{(j)}) + f_R(U^{(j)}) ]
            = \hat U^{(i)} + \Delta t \sum_{j=1}^{i-1} (a_{ij} - \tilde a_{ij})  f_L(U^{(j)}) 
```

By convention, ``\tilde a_{11} = 0``, so that ``U^{(1)} = u^n``, and for all other stages the implicit coefficient ``\tilde a_{ii}`` is the same. Additionally we assume the linear operator ``L`` is time-invariant, which means that when using direct methods ``W`` only needs to be factorized once (assuming a constant ``\Delta t``).


Alternatively if an iterative solver is used used, we can write
```math
\bar U^{(i)} = u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} ( f_L(U^{(j)}) + f_R(U^{(j)}) ) 
            = \hat U^{(i)} + \Delta t  L \sum_{j=1}^{i-1} (a_{ij} - \tilde a_{ij})  U^{(j)}
```
at the cost of one evaluation of ``f_L``.

## Reducing storage
### Remainder form

If we are given ``f_L`` and ``f_R``, we can avoid storing ``f_L(U^{(j)})`` by further defining
```math
\Omega^{(i)} = \sum_{j=1}^{i-1} \frac{\tilde a_{ij}}{\tilde a_{ii}} U^{(j)}
```
and writing
```math
\hat U^{(i)} = u^n + \Delta t \tilde a_{ii} f_L( \Omega^{(i)} ) + \Delta t \sum_{j=1}^{i-1} a_{ij} f_R(U^{(j)})
```
which requires only 1 evaluation of ``f_L`` per stage (plus one extra if we want ``\bar U^{(i)}``).

We can reduce this further evaluation by solving the offset linear problem
```math
W (U^{(i)} + \Omega^{(i)}) 
  = \hat U_{(i)} + (I - \Delta t \tilde a_{ii} L)  \Omega^{(i)} 
  = u^n + \Omega^{(i)} + \Delta t \sum_{j=1}^{i-1} a_{ij} f_R(U^{(j)})
```

### Full form

Similary, if we are given ``f`` and ``f_L``, we can avoid storing ``f_L(U^{(j)}`` by defining
```math
\Omega^{(i)} = \sum_{j=1}^{i-1} \frac{\tilde a_{ij} - a_{ij}}{\tilde a_{ii}} U^{(j)}
```
so that we can write
```math
\hat U^{(i)} = u^n + \Delta t \tilde a_{ii} f_L(\Omega^{(i)}) + \Delta t \sum_{j=1}^{i-1} a_{ij} f(U^{(j)})
```
which only requires one evaluation of ``f_L``. 

As above, we can eliminate this stage by rewriting into an offset linear problem
```math
W (U^{(i)} + \Omega^{(i)}) 
  = \hat U_{(i)} + (I - \Delta t \tilde a_{ii} L)  \Omega^{(i)} 
  = u^n + \Omega^{(i)} + \Delta t \sum_{j=1}^{i-1} a_{ij} f(U^{(j)})
```

## References

* F. X. Giraldo, J. F. Kelly, and E. M. Constantinescu (2013). Implicit-Explicit Formulations of a Three-Dimensional Nonhydrostatic Unified Model of the Atmosphere (NUMA) _SIAM Journal on Scientific Computing_ 35(5), B1162-B1194, doi:[10.1137/120876034](https://doi.org/10.1137/120876034)
