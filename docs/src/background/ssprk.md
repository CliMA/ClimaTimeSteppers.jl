# Strong Stability Preserving Runge--Kutta methods

SSPRK methods are self-starting, with ``U^{(1)} = u^n``, and subsequent stage updates of the form
```math
\begin{aligned}
U^{(i+1)} &= A_{i,1} u^n + A_{i,2} U^{(i)} + \Delta t B_i f(U^{(i)}, t + c_i \Delta t)
\end{aligned}
```
with the value at the next step being the ``N+1``th stage value ``u^{n+1} = U^{(N+1)})``.

This allows the updates to be performed with only three copies of the state vector (storing ``u^n``, ``U^{(i)}`` and ``f(U^{(i)},t)``).
