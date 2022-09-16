#=
An s-stage diagonally implicit Runge-Kutta (DIRK) method for solving
∂u/∂t = f(u, t) is specified by a lower triangular s×s matrix a (whose values
are called "internal coefficients"), a vector b of length s (whose values are
called "weights"), and a vector c of length s (whose values are called
"abcissae" or "nodes").
Given the state u at time t, the state u_next at time t + Δt is given by
    u_next := u + Δt * ∑_{i=1}^s bᵢ * f(Uᵢ, t + Δt * cᵢ), where
    Uᵢ := u + Δt * ∑_{j=1}^i aᵢⱼ * f(Uⱼ, t + Δt * cⱼ) ∀ i ∈ 1:s.
In order to transform this DIRK method into a Rosenbrock method, we must assume
that it is "internally consistent", which means that
    cᵢ := ∑_{j=1}^i aᵢⱼ ∀ i ∈ 1:s.

First, we will simplify our notation by defining
    Tᵢ := t + Δt * ∑_{j=1}^i aᵢⱼ and
    kᵢ := Δt * f(Uᵢ, Tᵢ) ∀ i ∈ 1:s.
This simplifies our previous expressions to
    u_next = u + ∑_{i=1}^s bᵢ * kᵢ and
    Uᵢ = u + ∑_{j=1}^i aᵢⱼ * kⱼ.
Next, we will define
    Ûᵢ := u + ∑_{j=1}^{i-1} aᵢⱼ * kⱼ and
    T̂ᵢ := t + Δt * ∑_{j=1}^{i-1} aᵢⱼ ∀ i ∈ 1:s.
This implies that
    Uᵢ = Ûᵢ + aᵢᵢ * kᵢ and
    Tᵢ = T̂ᵢ + Δt * aᵢᵢ.
Substituting this into the definition of kᵢ gives us
    kᵢ = Δt * f(Ûᵢ + aᵢᵢ * kᵢ, T̂ᵢ + Δt * aᵢᵢ).
Approximating the value of f with a first-order Taylor series expansion around
Ûᵢ and T̂ᵢ gives us
    kᵢ = Δt * f(Ûᵢ, T̂ᵢ) + Δt * J(Ûᵢ, T̂ᵢ) * aᵢᵢ * kᵢ + Δt^2 * ḟ(Ûᵢ, T̂ᵢ) * aᵢᵢ,
    where J(u, t) := ∂f/∂u(u, t) and ḟ(u, t) := ∂f/∂t(u, t).

If we ignore the ḟ term, this equation for kᵢ can also be obtained by performing
a single Newton iteration to solve for Uᵢ from the starting guess Ûᵢ.
Substituting the definition of kᵢ into the last expression for Uᵢ gives us
    Uᵢ = Ûᵢ + Δt * aᵢᵢ * f(Uᵢ, Tᵢ).
This can be rewritten as
    Fᵢ(Uᵢ) = 0, where
    Fᵢ(u) := Ûᵢ + Δt * aᵢᵢ * f(u, Tᵢ) - u ∀ i ∈ 1:s.
The Jacobian of this new function is
    ∂Fᵢ/∂u(u) = Δt * aᵢᵢ * J(u, Tᵢ) - I.
When solving this equation using Newton's method, we proceed from the n-th
iteration's state Uᵢ_n to the next iteration's state by defining
    Uᵢ_{n+1} := Uᵢ_n + ΔUᵢ_n, where
    ΔUᵢ_n := -∂Fᵢ/∂u(Uᵢ_n) \\ Fᵢ(Uᵢ_n) ∀ n ≥ 0.
So, if Uᵢ_0 = Ûᵢ, this means that
    ΔUᵢ_0 = -∂Fᵢ/∂u(Ûᵢ) \\ Fᵢ(Ûᵢ) =
        = (I - Δt * aᵢᵢ * J(Ûᵢ, Tᵢ)) \\ (Δt * aᵢᵢ * f(Ûᵢ, Tᵢ)).
We can rearrange this equation to find that
    ΔUᵢ_0 = Δt * aᵢᵢ * f(Ûᵢ, Tᵢ) + Δt * J(Ûᵢ, Tᵢ) * aᵢᵢ * ΔUᵢ_0.
Dividing through by aᵢᵢ gives us
    ΔUᵢ_0/aᵢᵢ = Δt * f(Ûᵢ, Tᵢ) + Δt * J(Ûᵢ, Tᵢ) * aᵢᵢ * (ΔUᵢ_0/aᵢᵢ).
We can see that this equation for ΔUᵢ_0/aᵢᵢ is identical to the one for kᵢ,
aside from the missing ḟ term.

In order to get a Rosenbrock method, we must make a modification to the equation
for kᵢ.
By using only a single Newton iteration, we break the assumptions that provide
guarantees about the convergence of the DIRK method.
To remedy this, we introduce an additional lower triangular s×s matrix of
coefficients γ, and we redefine
    kᵢ := Δt * f(Ûᵢ, T̂ᵢ) + Δt * J(Ûᵢ, T̂ᵢ) * ∑_{j=1}^i γᵢⱼ * kⱼ +
          Δt^2 * ḟ(Ûᵢ, T̂ᵢ) * ∑_{j=1}^i γᵢⱼ ∀ i ∈ 1:s.
In other words, we account for the higher-order terms dropped by the first-order
Taylor series expansion by taking carefully chosen linear combinations of kⱼ.
Since this effectively replaces aᵢᵢ with entries in γ, the diagonal entries can
be omitted from a, making it strictly lower triangular.

For a "modified Rosenbrock method", the coefficients are chosen so that
J(Ûᵢ, T̂ᵢ) and ḟ(Ûᵢ, T̂ᵢ) can be approximated by their values at time t, J(u, t)
and ḟ(u, t).
For a "W-method", the approximation can be even coarser; e.g., the values at a
previous timestep, or the values at a precomputed reference state, or even
approximations that do not correspond to any particular state.
So, we will modify the last equation by replacing J(Ûᵢ, T̂ᵢ) and ḟ(Ûᵢ, T̂ᵢ) with
some approximations J and ḟ (whose values should be governed by the constraints
of the Rosenbrock method being used), giving us
    kᵢ := Δt * f(Ûᵢ, T̂ᵢ) + Δt * J * ∑_{j=1}^i γᵢⱼ * kⱼ +
          Δt^2 * ḟ * ∑_{j=1}^i γᵢⱼ ∀ i ∈ 1:s.

Implementing a way to solve the last equation for kᵢ would require evaluating
the matrix-vector product J(Ûᵢ, T̂ᵢ) * ∑_{j=1}^i γᵢⱼ * kⱼ, which is more
computationally expensive than taking linear combinations of vectors.
This can be avoided by introducing the new variables
    Kᵢ := ∑_{j=1}^i γᵢⱼ * kⱼ ∀ i ∈ 1:s.
We can rewrite this, along with the last equations for u_next and Ûᵢ, as matrix
equations:
    K = γ * k,
    u_next = u + bᵀ * k, and
    Û = u + a * k.
In these equations, k, K, and U are matrices whose i-th rows are kᵢ, Kᵢ, and Uᵢ,
respectively; the last equation uses the fact that a is now strictly lower
triangular.
If γᵢᵢ != 0 ∀ i ∈ 1:s, then, γ is invertible, so the first equation implies that
    k = γ⁻¹ * K.
Substituting this into the two remaining matrix equations gives us
    u_next = u + bᵀγ⁻¹ * K and
    U = u + aγ⁻¹ * K.
Since γ is lower triangular, γ⁻¹ is also lower triangular, and, since a is
strictly lower triangular, aγ⁻¹ is also strictly lower triangular, which means
that we can rewrite the last three equations as
    kᵢ = ∑_{j=1}^i (γ⁻¹)ᵢⱼ * Kⱼ,
    u_next = u + ∑_{i=1}^s (bᵀγ⁻¹)ᵢ * Kᵢ, and
    Ûᵢ = u + ∑_{j=1}^{i-1} (aγ⁻¹)ᵢⱼ * Kⱼ.
Also, since γ is lower triangular, (γ⁻¹)ᵢᵢ = 1/γᵢᵢ, which means that
    kᵢ = 1/γᵢᵢ * Kᵢ + ∑_{j=1}^{i-1} (γ⁻¹)ᵢⱼ * Kⱼ.
Using this new expression for kᵢ, we can rewrite the last equation for kᵢ as
    1/γᵢᵢ * Kᵢ + ∑_{j=1}^{i-1} (γ⁻¹)ᵢⱼ * Kⱼ =
        = Δt * f(Ûᵢ, T̂ᵢ) + Δt * J * Kᵢ + Δt^2 * ḟ * ∑_{j=1}^i γᵢⱼ.
Solving for Kᵢ gives us
    Kᵢ = (I - Δt * γᵢᵢ * J) \\
        (
            Δt^2 * ḟ * ∑_{j=1}^i γᵢᵢ * γᵢⱼ - ∑_{j=1}^{i-1} γᵢᵢ * (γ⁻¹)ᵢⱼ * Kⱼ +
            Δt * γᵢᵢ * f(Ûᵢ, T̂ᵢ)
        ).

So, in summary, we have that, for some lower triangular s×s matrix γ with
nonzero diagonal entries, some strictly lower triangular s×s matrix a, some
vector b of length s, and some approximations J and ḟ of ∂f/∂u(Ûᵢ, T̂ᵢ) and
∂f/∂t(Ûᵢ, T̂ᵢ),
    u_next := u + ∑_{i=1}^s (bᵀγ⁻¹)ᵢ * Kᵢ, where
    Ûᵢ := u + ∑_{j=1}^{i-1} (aγ⁻¹)ᵢⱼ * Kⱼ,
    T̂ᵢ := t + Δt * ∑_{j=1}^{i-1} aᵢⱼ, and
    Kᵢ := (I - Δt * γᵢᵢ * J) \\
        (
            Δt^2 * ḟ * ∑_{j=1}^i γᵢᵢ * γᵢⱼ - ∑_{j=1}^{i-1} γᵢᵢ * (γ⁻¹)ᵢⱼ * Kⱼ +
            Δt * γᵢᵢ * f(Ûᵢ, T̂ᵢ)
        ) ∀ i ∈ 1:s.

Now, suppose that, instead of being provided with f(u, t), we are provided with
g(u₊, u, t, Δt), where
    g(u₊, u, t, Δt) := u₊ + Δt * f(u, t).
This is useful if, for example, we have some tendency f̃(u, t), but, instead of
just using it to increment a state, we want to apply a filter/limiter L after
incrementing:
    g(u₊, u, t, Δt) = L(u₊ + Δt * f̃(u, t)).
Rewriting the last definition of Kᵢ in terms of g instead of f gives us
    Kᵢ := (I - Δt * γᵢᵢ * J) \\
        g(
            Δt^2 * ḟ * ∑_{j=1}^i γᵢᵢ * γᵢⱼ - ∑_{j=1}^{i-1} γᵢᵢ * (γ⁻¹)ᵢⱼ * Kⱼ,
            Ûᵢ,
            T̂ᵢ,
            Δt * γᵢᵢ,
        ) ∀ i ∈ 1:s.
=#

struct RosenbrockAlgorithm{γ, a, b, L}
    linsolve:::L
end
RosenbrockAlgorithm{γ, a, b}(linsolve::L) where {γ, a, b, L} =
    RosenbrockAlgorithm{γ, a, b, L}(linsolve)
