# Additive Runge--Kutta

ARK methods are IMEX (Implicit-Explicit) methods based on splitting the ODE function ``f(u) = f_L(u) + f_R(t)`` 
where ``f_L(u) = L u`` is a linear operator which is treated implicitly, and ``f_R(u)`` is the remainder to be
treated explicitly. Typically we will be given either the pair ``(f_R, f_L)``, which we will term the _remainder form_,
or ``(f, f_L)`` which we will term the _full form_. 

The value on the ``i``th stage ``U^{(i)}`` is
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

By convention, ``\tilde a_{11} = 0``, so that ``U^{(1)} = u^n``, and for all other stages the implicit coefficients ``\tilde a_{ii}`` are the same. 

If the linear operator ``L`` is time-invariant and ``\Delta t`` is constantm, then if using a direct solver, ``W`` only needs to be factorized once.


Alternatively if an iterative solver is used used, we can write
```math
\bar U^{(i)} = u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} [ f_L(U^{(j)}) + f_R(U^{(j)}) ]
            = \hat U^{(i)} + \Delta t  L \sum_{j=1}^{i-1} (a_{ij} - \tilde a_{ij})  U^{(j)}
```
at the cost of one evaluation of ``f_L``.

## Reducing evaluations and storage

If the linear operator ``L`` is constant, then we are able to avoid evaluating the ``f_L`` explicitly.

### Remainder form

If we are given ``f_L`` and ``f_R``, we can avoid storing ``f_L(U^{(j)})`` by further defining
```math
\Omega^{(i)} = \sum_{j=1}^{i-1} \frac{\tilde a_{ij}}{\tilde a_{ii}} U^{(j)}
```
and writing
```math
\hat U^{(i)} = u^n + \Delta t \tilde a_{ii} f_L( \Omega^{(i)} ) + \Delta t \sum_{j=1}^{i-1} a_{ij} f_R(U^{(j)})
```

This can be rewritten into an offset linear problem
```math
W V^{(i)} = \hat V^{(i)}
```
where
```math
V^{(i)} = U^{(i)} + \Omega^{(i)}
```
and
```math
\hat V^{(i)}
  = \hat U_{(i)} + (I - \Delta t \tilde a_{ii} L)  \Omega^{(i)} 
  = u^n + \Omega^{(i)} + \Delta t \sum_{j=1}^{i-1} a_{ij} f_R(U^{(j)})
```
If using an iterative method, an initial guess is
```math
\bar V^{(i)} = \bar U^{(i)} + \Omega^{(i)}
  = u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} f_R(U^{(j)}) + \Omega^{(i)} + \Delta t \sum_{j=1}^{i-1} a_{ij} f_L(U^{(j)})
  = \hat V^{(i)} + \Delta t L \sum_{j=1}^{i-1} a_{ij} U^{(j)}
```

### Full form

Similary, if we are given ``f`` and ``f_L``, we can avoid storing ``f_L(U^{(j)})`` by defining
```math
\Omega^{(i)} = \sum_{j=1}^{i-1} \frac{\tilde a_{ij} - a_{ij}}{\tilde a_{ii}} U^{(j)}
```
so that we can write
```math
\hat U^{(i)} = u^n + \Delta t \tilde a_{ii} f_L(\Omega^{(i)}) + \Delta t \sum_{j=1}^{i-1} a_{ij} f(U^{(j)})
```

As above, we can rewrite into an offset linear problem
```math
W V^{(i)} = \hat V^{(i)}
```
where
```math
V^{(i)} = U^{(i)} + \Omega^{(i)}
```
and
```math
\hat V^{(i)} 
  = \hat U_{(i)} + (I - \Delta t \tilde a_{ii} L)  \Omega^{(i)} 
  = u^n + \Omega^{(i)} + \Delta t \sum_{j=1}^{i-1} a_{ij} f(U^{(j)})
```
If using an iterative method, an initial guess is
```math
\bar V^{(i)} = \bar U^{(i)} + \Omega^{(i)}
  = u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} f(U^{(j)}) 
  = \hat V^{(i)} - \Omega^{(i)}
```


## References

* F. X. Giraldo, J. F. Kelly, and E. M. Constantinescu (2013). Implicit-Explicit Formulations of a Three-Dimensional Nonhydrostatic Unified Model of the Atmosphere (NUMA) _SIAM Journal on Scientific Computing_ 35(5), B1162-B1194, doi:[10.1137/120876034](https://doi.org/10.1137/120876034)
