# Low-storage Runge--Kutta methods

LSRK methods are a class of explicit Runge--Kutta methods designed to minimize memory usage by storing only two copies of the state vector at any time. The $2N$-storage family was introduced by [Carpenter and Kennedy (1994)](@cite CK1994) and later extended with optimized stability regions by [Niegemann et al. (2012)](@cite NDB2012).

Each timestep begins with $U^{(1)} = u^n$ and proceeds through $s$ stage updates of the form
```math
\begin{aligned}
dU^{(i)} &= f(U^{(i)}, t + c_i \Delta t) + A_i dU^{(i-1)}\\
U^{(i+1)} &= U^{(i)} + \Delta t B_i dU^{(i)}
\end{aligned}
```
where $A_1 = c_1 = 0$ (so the first accumulation reduces to $dU^{(1)} = f(u^n, t)$). The solution at the next timestep is $u^{n+1} = U^{(s+1)}$.

Because each stage only needs $U^{(i)}$ and $dU^{(i-1)}$, the update can be performed with only two state-sized registers (provided $f$ can be evaluated in incrementing form).

It can be written as an RK scheme with Butcher tableau coefficients defined by the recurrences
```math
\begin{aligned}
a_{i,i} &= 0 \\
a_{i,j} &= B_j + A_{j+1} a_{i,j+1}\\
b_N &= B_N \\
b_i &= B_i + A_{i+1} b_{i+1}
\end{aligned}
```
or equivalently
```math
\begin{aligned}
a_{j,j} &= 0 \\
a_{i,j} &= a_{i-1,j} + B_{i-1} \prod_{k=j+1}^{i-1} A_k
\end{aligned}
```
with ``b_j`` treated analogously as ``a_{N+1,j}``.
