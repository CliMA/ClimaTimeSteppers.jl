#=
## Introduction to Rosenbrock Methods

An s-stage diagonally implicit Runge-Kutta (DIRK) method for solving
∂u/∂t = f(u, t) is specified by a lower triangular s×s matrix a (whose values
are called "internal coefficients"), a vector b of length s (whose values are
called "weights"), and a vector c of length s (whose values are called
"abcissae" or "nodes").
Given the state u at time t, the state u_next at time t + Δt is given by
    u_next := u + Δt * ∑_{i=1}^s bᵢ * f(Uᵢ, t + Δt * cᵢ), where
    Uᵢ := u + Δt * ∑_{j=1}^i aᵢⱼ * f(Uⱼ, t + Δt * cⱼ).
In order to transform this DIRK method into a Rosenbrock method, we must assume
that it is "internally consistent", which means that
    cᵢ := ∑_{j=1}^i aᵢⱼ.

First, we will simplify our notation by defining
    Tᵢ := t + Δt * ∑_{j=1}^i aᵢⱼ and
    Fᵢ := f(Uᵢ, Tᵢ).
This simplifies our previous expressions to
    u_next = u + Δt * ∑_{i=1}^s bᵢ * Fᵢ and
    Uᵢ = u + Δt * ∑_{j=1}^i aᵢⱼ * Fⱼ.
Next, we will define
    Ûᵢ := u + Δt * ∑_{j=1}^{i-1} aᵢⱼ * Fⱼ and
    T̂ᵢ := t + Δt * ∑_{j=1}^{i-1} aᵢⱼ.
This implies that
    Uᵢ = Ûᵢ + Δt * aᵢᵢ * Fᵢ and
    Tᵢ = T̂ᵢ + Δt * aᵢᵢ.
Substituting this into the definition of Fᵢ gives us
    Fᵢ = f(Ûᵢ + Δt * aᵢᵢ * Fᵢ, T̂ᵢ + Δt * aᵢᵢ).
Approximating the value of f with a first-order Taylor series expansion around
Ûᵢ and T̂ᵢ gives us
    Fᵢ ≈ f(Ûᵢ, T̂ᵢ) + Δt * J(Ûᵢ, T̂ᵢ) * aᵢᵢ * Fᵢ + Δt * ḟ(Ûᵢ, T̂ᵢ) * aᵢᵢ, where
    J(u, t) := ∂f/∂u(u, t) and
    ḟ(u, t) := ∂f/∂t(u, t).
This is roughly equivalent to running a single Newton iteration to solve for Fᵢ,
starting with an initial guess of f(Ûᵢ, T̂ᵢ) (or, equivalently, to solve for Uᵢ,
starting with an initial guess of Ûᵢ).

In order to obtain a Rosenbrock method, we must modify this equation for Fᵢ.
By using only a single Newton iteration, we break the assumptions that provide
guarantees about the convergence of the DIRK method.
To remedy this, we introduce an additional lower triangular s×s matrix of
coefficients γ, and we redefine Fᵢ to be the solution to
    Fᵢ = f(Ûᵢ, T̂ᵢ) + Δt * J(Ûᵢ, T̂ᵢ) * ∑_{j=1}^i γᵢⱼ * Fⱼ +
         Δt * ḟ(Ûᵢ, T̂ᵢ) * ∑_{j=1}^i γᵢⱼ.
In other words, we account for the higher-order terms dropped by the first-order
Taylor series expansion by taking carefully chosen linear combinations of Fⱼ.
Since this effectively replaces aᵢᵢ with entries in γ, the diagonal entries can
now be omitted from a, making it strictly lower triangular.

For a "modified Rosenbrock method", the coefficients are chosen so that
J(Ûᵢ, T̂ᵢ) and ḟ(Ûᵢ, T̂ᵢ) can be approximated by their values at time t, J(u, t)
and ḟ(u, t).
For a "W-method", the approximation can be even coarser; e.g., the values at a
previous timestep, or the values at a precomputed reference state, or even
approximations that do not correspond to any particular state.
So, we will modify the last equation by replacing J(Ûᵢ, T̂ᵢ) and ḟ(Ûᵢ, T̂ᵢ) with
some approximations J and ḟ (whose values must satisfy the constraints of the
specific Rosenbrock method being used), giving us
    Fᵢ = f(Ûᵢ, T̂ᵢ) + Δt * J * ∑_{j=1}^i γᵢⱼ * Fⱼ + Δt * ḟ * ∑_{j=1}^i γᵢⱼ.
Solving this equation for Fᵢ lets us redefine
    Fᵢ := (I - Δt * γᵢᵢ * J)⁻¹ * (
              f(Ûᵢ, T̂ᵢ) + Δt * J * ∑_{j=1}^{i-1} γᵢⱼ * Fⱼ +
              Δt * ḟ * ∑_{j=1}^i γᵢⱼ
          ).
Since multiplying ∑_{j=1}^{i-1} γᵢⱼ * Fⱼ by J is more computationally expensive
than subtracting it from another value, we will rewrite this as
    Fᵢ := (I - Δt * γᵢᵢ * J)⁻¹ * (
              f(Ûᵢ, T̂ᵢ) + γᵢᵢ⁻¹ * ∑_{j=1}^{i-1} γᵢⱼ * Fⱼ +
              Δt * ḟ * ∑_{j=1}^i γᵢⱼ
          ) - γᵢᵢ⁻¹ * ∑_{j=1}^{i-1} γᵢⱼ * Fⱼ.

So, in summary, a Rosenbrock method is defined by some lower triangular s×s
matrix γ, some strictly lower triangular s×s matrix a, some vector b of length
s, and some approximations J and ḟ of ∂f/∂u(Ûᵢ, T̂ᵢ) and ∂f/∂t(Ûᵢ, T̂ᵢ), which
are all used to compute
    u_next := u + Δt * ∑_{i=1}^s bᵢ * Fᵢ, where
    Ûᵢ := u + Δt * ∑_{j=1}^{i-1} aᵢⱼ * Fⱼ,
    T̂ᵢ := t + Δt * ∑_{j=1}^{i-1} aᵢⱼ, and
    Fᵢ := (I - Δt * γᵢᵢ * J)⁻¹ * (
              f(Ûᵢ, T̂ᵢ) + γᵢᵢ⁻¹ * ∑_{j=1}^{i-1} γᵢⱼ * Fⱼ +
              Δt * ḟ * ∑_{j=1}^i γᵢⱼ
          ) - γᵢᵢ⁻¹ * ∑_{j=1}^{i-1} γᵢⱼ * Fⱼ.

## Converting to Matrix Form

To simplify our further reformulations, we will convert our definitions to
matrix form.

First, we will reduce the number of matrix equations we need to analyze by
defining
    âᵢⱼ := i < s ? a₍ᵢ₊₁₎ⱼ : bⱼ and
    Û⁺ᵢ := i < s ? Ûᵢ₊₁ : u_next.
We can then redefine
    u_next := Û⁺ₛ and
    Ûᵢ := i == 1 ? u : Û⁺ᵢ₋₁, where
    Û⁺ᵢ := u + Δt * ∑_{j=1}^i âᵢⱼ * Fⱼ.

The equations we will convert to matrix form are
    Û⁺ᵢ = u + Δt * ∑_{j=1}^i âᵢⱼ * Fⱼ and
    Fᵢ = f(Ûᵢ, T̂ᵢ) + Δt * J * ∑_{j=1}^i γᵢⱼ * Fⱼ + Δt * ḟ * ∑_{j=1}^i γᵢⱼ.
Rewriting these with an explicit element index n ∈ 1:N (where N = length(u))
gives us
    Û⁺ᵢ[n] = u[n] + Δt * ∑_{j=1}^i âᵢⱼ * Fⱼ[n] and
    Fᵢ[n] = f(Ûᵢ, T̂ᵢ)[n] + Δt * ∑_{m=1}^N J[n,m] * ∑_{j=1}^i γᵢⱼ * Fⱼ[m] +
            Δt * ḟ[n] * ∑_{j=1}^i γᵢⱼ.
Now, we will formally define the vectors and matrices
    𝟙 ∈ ℝˢ     | 𝟙ᵢ := 1,
    u ∈ ℝᴺ     | uₙ := u[n],
    ḟ ∈ ℝᴺ     | ḟₙ := ḟ[n], and
    Û⁺ ∈ ℝᴺ×ℝˢ | Û⁺ₙᵢ := Û⁺ᵢ[n],
    F ∈ ℝᴺ×ℝˢ  | Fₙᵢ := Fᵢ[n],
    F̂ ∈ ℝᴺ×ℝˢ  | F̂ₙᵢ := f(Ûᵢ, T̂ᵢ)[n],
    J ∈ ℝᴺ×ℝᴺ  | Jₙₘ := J[n,m].
(If J and ḟ are different for each stage, they may be replaced with tensors in
all of the following manipulations.)
We then have that
    Û⁺ₙᵢ = uₙ * 𝟙ᵢ + Δt * ∑_{j=1}^i Fₙⱼ * âᵢⱼ and
    Fₙᵢ = F̂ₙᵢ + Δt * ∑_{m=1}^N Jₙₘ * ∑_{j=1}^i Fₘⱼ * γᵢⱼ +
          Δt * ḟₙ * ∑_{j=1}^i 𝟙ⱼ * γᵢⱼ.
We can rewrite this as
    Û⁺ₙᵢ = uₙ * (𝟙ᵀ)ᵢ + Δt * ∑_{j=1}^s Fₙⱼ * (âᵀ)ⱼᵢ and
    Fₙᵢ = F̂ₙᵢ + Δt * ∑_{m=1}^N Jₙₘ * ∑_{j=1}^s Fₘⱼ * (γᵀ)ⱼᵢ +
          Δt * ḟₙ * ∑_{j=1}^s (𝟙ᵀ)ⱼ * (γᵀ)ⱼᵢ.
Combining matrix-matrix and matrix-vector products gives us
    Û⁺ₙᵢ = uₙ * (𝟙ᵀ)ᵢ + Δt * (F * âᵀ)ₙᵢ and
    Fₙᵢ = F̂ₙᵢ + Δt * (J * F * γᵀ)ₙᵢ + Δt * ḟₙ * (γ * 𝟙)ᵢ.
Dropping the indices then tells us that
    Û⁺ = u ⊗ 𝟙ᵀ + Δt * F * âᵀ and
    F = F̂ + Δt * J * F * γᵀ + Δt * ḟ ⊗ (γ * 𝟙)ᵀ.
To rewrite this in a way that more directly corresponds to our last formulation,
we will define the functions diag() and lower(), such that, for any lower
triangular matrix M,
    M = diag(M) + lower(M), where
    diag(M)ᵢⱼ := i == j ? Mᵢⱼ : 0 and
    lower(M)ᵢⱼ := i > j ? Mᵢⱼ : 0.
This lets us rewrite the equation for F as
    F * (I + diag(γ)⁻¹ * lower(γ))ᵀ -
    Δt * J * F * (I + diag(γ)⁻¹ * lower(γ))ᵀ * diag(γ)ᵀ =
        = F̂ + F * (diag(γ)⁻¹ * lower(γ))ᵀ + Δt * ḟ ⊗ (γ * 𝟙)ᵀ.

We will now use these matrix equations to define two reformulations: one that
optimizes performance by eliminating the subtraction after the linear solve, and
one that enables limiters by appropriately modifying the value being subtracted.

## Optimizing Performance

Let us define a new N×s matrix
    K := F * γᵀ.
We then have that
    F = K * (γ⁻¹)ᵀ.
This allows us to rewrite the matrix equations as
    Û⁺ = u ⊗ 𝟙ᵀ + Δt * K * (â * γ⁻¹)ᵀ and
    K * (γ⁻¹)ᵀ = F̂ + Δt * J * K + Δt * ḟ ⊗ (γ * 𝟙)ᵀ.
Since γ is lower triangular,
    γ⁻¹ = diag(γ⁻¹) + lower(γ⁻¹) = diag(γ)⁻¹ + lower(γ⁻¹).
This lets us rewrite the equation for K as
    K * (diag(γ)⁻¹ + lower(γ⁻¹))ᵀ = F̂ + Δt * J * K + Δt * ḟ ⊗ (γ * 𝟙)ᵀ.
Multiplying through on the right by diag(γ)ᵀ and rearranging gives us
    K - Δt * J * K * diag(γ)ᵀ =
        (F̂ - K * lower(γ⁻¹)ᵀ + Δt * ḟ ⊗ (γ * 𝟙)ᵀ) * diag(γ)ᵀ.
Taking this and the equation for Û⁺ back out of matrix form gives us
    Û⁺ᵢ = u + Δt * ∑_{j=1}^i (â * γ⁻¹)ᵢⱼ * Kⱼ and
    Kᵢ - Δt * γᵢᵢ * J * Kᵢ =
        = γᵢᵢ *
          (f(Ûᵢ, T̂ᵢ) - ∑_{j=1}^{i-1} (γ⁻¹)ᵢⱼ * Kⱼ + Δt * ḟ * ∑_{j=1}^i γᵢⱼ).
Thus, to optimize performance, we can redefine
    Û⁺ᵢ := u + Δt * ∑_{j=1}^i (â * γ⁻¹)ᵢⱼ * Kⱼ, where
    Kᵢ := (I - Δt * γᵢᵢ * J)⁻¹ * γᵢᵢ * (
              f(Ûᵢ, T̂ᵢ) - ∑_{j=1}^{i-1} (γ⁻¹)ᵢⱼ * Kⱼ + Δt * ḟ * ∑_{j=1}^i γᵢⱼ
          ).

## Enabling Limiters

In the previous section, we changed the temporary variable from Fᵢ, which is an
approximation of f(Uᵢ, Tᵢ), to Kᵢ, which is a linear combination of previously
computed values of Fᵢ.
Now, we will change the temporary variable to Vᵢ, so that the approximations of
Uᵢ and u_next will be linear combinations of previously computed values of Vᵢ.
In other words, we will make it so that Û⁺ is a linear transformation of the
temporary variable, rather than an affine transformation, absorbing u into the
temporary variable.
So, consider a lower triangular s×s matrix β and an N×s matrix V such that
    Û⁺ = V * βᵀ.
We can then rewrite the matrix equations as
    V * βᵀ = u ⊗ 𝟙ᵀ + Δt * F * âᵀ and
    F = F̂ + Δt * J * F * γᵀ + Δt * ḟ ⊗ (γ * 𝟙)ᵀ.
Solving the first equation for F tells us that
    F = Δt⁻¹ * V * (â⁻¹ * β)ᵀ - Δt⁻¹ * u ⊗ (â⁻¹ * 𝟙)ᵀ.
Substituting this into the second equation gives us
    Δt⁻¹ * V * (â⁻¹ * β)ᵀ - Δt⁻¹ * u ⊗ (â⁻¹ * 𝟙)ᵀ =
        = F̂ + J * V * (γ * â⁻¹ * β)ᵀ - J * u ⊗ (γ * â⁻¹ * 𝟙)ᵀ +
          Δt * ḟ ⊗ (γ * 𝟙)ᵀ.
Multiplying through on the right by Δt * (β⁻¹ * â * γ⁻¹)ᵀ and rearranging
results in
    V * (β⁻¹ * â * γ⁻¹ * â⁻¹ * β)ᵀ - Δt * J * V =
        = u ⊗ (β⁻¹ * â * γ⁻¹ * â⁻¹ * 𝟙)ᵀ - Δt * J * u ⊗ (β⁻¹ * 𝟙)ᵀ +
          Δt * F̂ * (β⁻¹ * â * γ⁻¹)ᵀ + Δt² * ḟ ⊗ (β⁻¹ * â * 𝟙)ᵀ.
Since γ, â, and β are all lower triangular, β⁻¹ * â * γ⁻¹ * â⁻¹ * β is also
lower triangular, which means that
    β⁻¹ * â * γ⁻¹ * â⁻¹ * β =
        diag(β⁻¹ * â * γ⁻¹ * â⁻¹ * β) + lower(β⁻¹ * â * γ⁻¹ * â⁻¹ * β).
In addition, we have that
    diag(β⁻¹ * â * γ⁻¹ * â⁻¹ * β) =
        = diag(β⁻¹) * diag(â) * diag(γ⁻¹) * diag(â⁻¹) * diag(β) =
        = diag(β)⁻¹ * diag(â) * diag(γ)⁻¹ * diag(â)⁻¹ * diag(β) =
        = diag(γ)⁻¹.
Combining the last two expressions gives us
    β⁻¹ * â * γ⁻¹ * â⁻¹ * β = diag(γ)⁻¹ + lower(β⁻¹ * â * γ⁻¹ * â⁻¹ * β).
Substituting this into the last equation for V gives us
    V * (diag(γ)⁻¹)ᵀ + V * lower(β⁻¹ * â * γ⁻¹ * â⁻¹ * β)ᵀ - Δt * J * V =
        = u ⊗ (β⁻¹ * â * γ⁻¹ * â⁻¹ * 𝟙)ᵀ - Δt * J * u ⊗ (β⁻¹ * 𝟙)ᵀ +
          Δt * F̂ * (β⁻¹ * â * γ⁻¹)ᵀ + Δt² * ḟ ⊗ (β⁻¹ * â * 𝟙)ᵀ.
Multiplying through on the right by diag(γ)ᵀ and rearranging tells us that
    V - Δt * J * V * diag(γ)ᵀ =
        = u ⊗ (diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ * 𝟙)ᵀ -
          Δt * J * u ⊗ (β⁻¹ * 𝟙)ᵀ * diag(γ)ᵀ +
          Δt * F̂ * (diag(γ) * β⁻¹ * â * γ⁻¹)ᵀ +
          Δt² * ḟ ⊗ (diag(γ) * β⁻¹ * â * 𝟙)ᵀ -
          V * (diag(γ) * lower(β⁻¹ * â * γ⁻¹ * â⁻¹ * β))ᵀ.
Since lower() preserves multiplication by a diagonal matrix, we can rewrite
this as
    V - Δt * J * V * diag(γ)ᵀ =
        = u ⊗ (diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ * 𝟙)ᵀ -
          Δt * J * u ⊗ (β⁻¹ * 𝟙)ᵀ * diag(γ)ᵀ +
          Δt * F̂ * (diag(γ) * β⁻¹ * â * γ⁻¹)ᵀ +
          Δt² * ḟ ⊗ (diag(γ) * β⁻¹ * â * 𝟙)ᵀ -
          V * lower(diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ * β)ᵀ.

We will now show that this reformulation will not allow us to eliminate
multiplications by J, as the previous ones did.
If we wanted to factor out all multiplications by J when we convert back out of
matrix form, we would rearrange the last equation to get
    (V - u ⊗ (β⁻¹ * 𝟙)ᵀ) - Δt * J * (V - u ⊗ (β⁻¹ * 𝟙)ᵀ) * diag(γ)ᵀ =
        = u ⊗ ((diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ - β⁻¹) * 𝟙)ᵀ +
          Δt * F̂ * (diag(γ) * β⁻¹ * â * γ⁻¹)ᵀ +
          Δt² * ḟ ⊗ (diag(γ) * β⁻¹ * â * 𝟙)ᵀ -
          V * lower(diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ * β)ᵀ.
In order to apply limiters on an unscaled state, we would then require that the
coefficient of u on the right-hand side of this equation be 𝟙ᵀ; i.e., that
    (diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ - β⁻¹) * 𝟙 = 𝟙.
In general, this equation does not have a solution β; e.g., if γ = â = d * I for
some scalar constant d, then the equation simplifies to
    (β⁻¹ - β⁻¹) * 𝟙 = 𝟙.
Even if we were to use more complex transformations, it would still be
impossible to eliminate multiplications by J while preserving an unscaled state
on the right-hand side.
For example, if we were to instead set Û⁺ = u ⊗ δᵀ + V * βᵀ for some vector δ of
length s, the above equation would become
    (diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ - β⁻¹) * (δ - 𝟙) = 𝟙.
This also does not have a general solution.

So, we must proceed without rearranging the last equation for V.
In order to apply limiters on an unscaled state, we must require that the
coefficient of u in that equation be 𝟙ᵀ, which implies that
    diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ * 𝟙 = 𝟙.

We will now show that we cannot also make the coefficient of the J term on the
right-hand side be the same as the one on the left-hand side.
If we wanted this to be the case, we would also have to satisfy the equation
    β⁻¹ * 𝟙 = 𝟙.
In general, we cannot simultaneously satisfy both of the last two equations;
e.g., if γ = d * I for some scalar constant d, we can rearrange the equations to
get that
    β * 𝟙 = d * â * γ⁻¹ * â⁻¹ * 𝟙 and β * 𝟙 = 𝟙.
Unless d * â * γ⁻¹ * â⁻¹ * 𝟙 = 𝟙 (which will not be the case in general), this
system of equations cannot be satisfied.

So, we will only require that β satisfies the equation
    diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ * 𝟙 = 𝟙.
This equation has infinitely many solutions; the easiest way to obtain a
solution is to set
    diag(γ) * β⁻¹ * â * γ⁻¹ * â⁻¹ = I.
This implies that
    β⁻¹ = diag(γ)⁻¹ * â * γ * â⁻¹ and
    β = â * γ⁻¹ * â⁻¹ * diag(γ).
Substituting this into the last equation for V gives us
    V - Δt * J * V * diag(γ)ᵀ =
        = u ⊗ 𝟙ᵀ - Δt * J * u ⊗ (β⁻¹ * 𝟙)ᵀ * diag(γ)ᵀ + Δt * F̂ * âᵀ +
          Δt² * ḟ ⊗ (â * γ * 𝟙)ᵀ - V * lower(β)ᵀ.
Taking this and the equation Û⁺ = V * βᵀ back out of matrix form gives us
    Û⁺ᵢ = ∑_{j=1}^i βᵢⱼ * Vᵢ and
    Vᵢ - Δt * γᵢᵢ * J * Vᵢ =
        = u - Δt * γᵢᵢ * J * u * ∑_{j=1}^i (β⁻¹)ᵢⱼ +
          Δt * ∑_{j=1}^i âᵢⱼ * f(Ûⱼ, T̂ⱼ) + Δt² * ḟ * ∑_{j=1}^i (â * γ)ᵢⱼ -
          ∑_{j=1}^{i-1} βᵢⱼ * Vⱼ.
Thus, to enable the use of limiters, we can redefine
    Û⁺ᵢ := ∑_{j=1}^i βᵢⱼ * Vᵢ, where
    Vᵢ := (I - Δt * γᵢᵢ * J)⁻¹ * (
              u - Δt * γᵢᵢ * J * u * ∑_{j=1}^i (β⁻¹)ᵢⱼ +
              Δt * ∑_{j=1}^i âᵢⱼ * f(Ûⱼ, T̂ⱼ) + Δt² * ḟ * ∑_{j=1}^i (â * γ)ᵢⱼ -
              ∑_{j=1}^{i-1} βᵢⱼ * Vⱼ
          ).

To actually use a limiter, we must replace the use of f(u, t) in the equation
for Vᵢ with a new function g(u₊, u, t, Δt), where
    g(u₊, u, t, Δt) := u₊ + Δt * f(u, t).
Internally, g can use a different tendency function f̃(u, t), and it can apply a
limiter L after incrementing a state with the output of f̃, so that
    g(u₊, u, t, Δt) = L(u₊ + Δt * f̃(u, t)).
Rewriting the last definition of Vᵢ in terms of g instead of f gives us
    Vᵢ := (I - Δt * γᵢᵢ * J)⁻¹ * g(
              u - Δt * γᵢᵢ * J * u * ∑_{j=1}^i (β⁻¹)ᵢⱼ +
              Δt * ∑_{j=1}^{i-1} âᵢⱼ * f(Ûⱼ, T̂ⱼ) +
              Δt² * ḟ * ∑_{j=1}^i (â * γ)ᵢⱼ - ∑_{j=1}^{i-1} βᵢⱼ * Vⱼ,
              Ûᵢ,
              T̂ᵢ
              Δt * âᵢᵢ,
          ).

## Negating Wfact

For some reason, OrdinaryDiffEq defines Wfact as Δt * γᵢᵢ * J - I, so we must
negate all of our temporary variables.

For the original formulation, this means that
    Ûᵢ := u - Δt * ∑_{j=1}^{i-1} aᵢⱼ * Fⱼ, where
    Fᵢ := (Δt * γᵢᵢ * J - I)⁻¹ * (
              f(Ûᵢ, T̂ᵢ) - γᵢᵢ⁻¹ * ∑_{j=1}^{i-1} γᵢⱼ * Fⱼ +
              Δt * ḟ * ∑_{j=1}^i γᵢⱼ
          ) + γᵢᵢ⁻¹ * ∑_{j=1}^{i-1} γᵢⱼ * Fⱼ.
For the performance formulation, this means that
    Û⁺ᵢ := u - Δt * ∑_{j=1}^i (â * γ⁻¹)ᵢⱼ * Kⱼ, where
    Kᵢ := (Δt * γᵢᵢ * J - I)⁻¹ * γᵢᵢ * (
              f(Ûᵢ, T̂ᵢ) + ∑_{j=1}^{i-1} (γ⁻¹)ᵢⱼ * Kⱼ + Δt * ḟ * ∑_{j=1}^i γᵢⱼ
          ).
For the limiters formulation, this means that
    Û⁺ᵢ := -∑_{j=1}^i βᵢⱼ * Vᵢ, where
    Vᵢ := (Δt * γᵢᵢ * J - I)⁻¹ * g(
              u - Δt * γᵢᵢ * J * u * ∑_{j=1}^i (β⁻¹)ᵢⱼ +
              Δt * ∑_{j=1}^{i-1} âᵢⱼ * f(Ûⱼ, T̂ⱼ) +
              Δt² * ḟ * ∑_{j=1}^i (â * γ)ᵢⱼ + ∑_{j=1}^{i-1} βᵢⱼ * Vⱼ,
              Ûᵢ,
              T̂ᵢ
              Δt * âᵢᵢ,
          ).
=#
import LinearAlgebra
import StaticArrays: SUnitRange, SOneTo
import Base: broadcasted, materialize!

struct RosenbrockAlgorithm{γ, a, b, U, L, M, S} <: DistributedODEAlgorithm
    update_jac::U
    linsolve::L
    multiply!::M
    set_Δtγ!::S
end
RosenbrockAlgorithm{γ, a, b}(;
    update_jac::U = UpdateEvery(NewStep()),
    linsolve::L,
    multiply!::M = nothing,
    set_Δtγ!::S = nothing,
) where {γ, a, b, U, L, M, S} =
    RosenbrockAlgorithm{γ, a, b, U, L, M, S}(
        update_jac,
        linsolve,
        multiply!,
        set_Δtγ!,
    )

@generated foreachval(f::F, ::Val{N}) where {F, N} =
    quote
        Base.@nexprs $N i -> f(Val(i))
        return nothing
    end
triangular_inv(matrix::T) where {T} =
    T(inv(LinearAlgebra.LowerTriangular(matrix)))
lower_plus_diag(matrix::T) where {T} = T(LinearAlgebra.LowerTriangular(matrix))
diag(matrix::T) where {T} = T(LinearAlgebra.Diagonal(matrix))
lower(matrix) = lower_plus_diag(matrix) - diag(matrix)
to_enumerated_rows(v::AbstractVector) = v
function to_enumerated_rows(m::AbstractMatrix)
    rows = tuple(1:size(m, 1)...)
    nonzero_indices = map(i -> findall(m[i, :] .!= 0), rows)
    enumerated_rows = map(
        i -> tuple(zip(nonzero_indices[i], m[i, nonzero_indices[i]])...),
        rows,
    )
    return enumerated_rows
end
linear_combination(enumerated_row, vectors) =
    map(((j, val),) -> broadcasted(*, val, vectors[j]), enumerated_row)
function scaled_linear_combination(enumerated_row, vectors, scale)
    unscaled_terms = linear_combination(enumerated_row, vectors)
    length(unscaled_terms) == 0 && return ()
    return (broadcasted(*, scale, broadcasted(+, unscaled_terms...)),)
end

num_stages(::Type{<:RosenbrockAlgorithm{γ}}) where {γ} = size(γ, 1)

function check_valid_parameters(
    ::Type{<:RosenbrockAlgorithm{γ, a, b, U}},
) where {γ, a, b, U}
    γ === lower_plus_diag(γ) ||
        error("γ must be a lower triangular matrix")
    a === lower(a) ||
        error("a must be a strictly lower triangular matrix")
    LinearAlgebra.det(γ) != 0 ||
        error("non-invertible matrices γ are not currently supported")
    if U != UpdateEvery{NewStage}
        diag(γ) === typeof(γ)(γ[1, 1] * I) ||
            error("γ must have a uniform diagonal when \
                   update_jac != UpdateEvery(NewStage())")
    end
    can_handle(U, NewStep()) || can_handle(U, NewStage()) ||
        error("update_jac must be able to handle NewStep() or NewStage()")
end
function check_valid_parameters(
    alg_type::Type{<:RosenbrockAlgorithm{γ, a, b, U, L, M, S}},
    ::Type{<:ForwardEulerODEFunction},
) where {γ, a, b, U, L, M, S}
    check_valid_parameters(alg_type)
    â = vcat(a[SUnitRange(2, length(b)), SOneTo(length(b))], transpose(b))
    LinearAlgebra.det(â) != 0 ||
        error("non-invertible matrices â are not currently supported when \
               using ForwardEulerODEFunction")
    M != Nothing ||
        error("multiply! must be specified when using ForwardEulerODEFunction")
    S != Nothing ||
        error("set_Δtγ! must be specified when using ForwardEulerODEFunction")
end
check_valid_parameters(alg_type, _) = check_valid_parameters(alg_type)

struct RosenbrockCache{C, U, L}
    _cache::C
    update_jac_cache::U
    linsolve!::L
end

# TODO: Minimize allocations by reusing temporary variables after they are no
# longer needed.
function cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::RosenbrockAlgorithm;
    kwargs...
)
    check_valid_parameters(typeof(alg), typeof(prob.f))

    s = num_stages(typeof(alg))
    u_prototype = prob.u0
    W_prototype = prob.f.jac_prototype
    increment_mode = prob.f isa ForwardEulerODEFunction

    _cache = NamedTuple((
        :Û⁺ᵢ => similar(u_prototype),
        (increment_mode ? (:Fs => map(i -> similar(u_prototype), 1:s),) : ())...,
        (increment_mode ? :Vs : :Ks) => map(i -> similar(u_prototype), 1:s),
        :W => similar(W_prototype),
        :ḟ => similar(u_prototype),
    ))

    update_jac_cache = allocate_cache(alg.update_jac)

    linsolve! = alg.linsolve(Val{:init}, W_prototype, u_prototype)

    return RosenbrockCache(_cache, update_jac_cache, linsolve!)
end

step_u!(integrator, cache::RosenbrockCache) =
    rosenbrock_step_u!(integrator, cache, integrator.prob.f)

# The precomputed values are too complicated for constant propagation, so we use
# @generated to force the values to be compile-time constants.
@generated function precomputed_values(
    ::Type{<:RosenbrockAlgorithm{γ, a, b}},
    _,
) where {γ, a, b}
    â = vcat(a[2:end, :], transpose(b))
    γ⁻¹ = triangular_inv(γ)
    lowerγ⁻¹ = lower(γ⁻¹)
    âγ⁻¹ = â * γ⁻¹
    diagγ𝟙 = vec(sum(diag(γ), dims = 2))
    γ𝟙 = vec(sum(γ, dims = 2))
    â𝟙 = vec(sum(â, dims = 2))
    values = map(to_enumerated_rows, (; lowerγ⁻¹, âγ⁻¹, diagγ𝟙, γ𝟙, â𝟙))
    return :($values)
end
@generated function precomputed_values(
    ::Type{<:RosenbrockAlgorithm{γ, a, b}},
    ::Type{<:ForwardEulerODEFunction},
) where {γ, a, b}
    â = vcat(a[2:end, :], transpose(b))
    âγ = â * γ
    β = â * triangular_inv(âγ) * diag(γ)
    lowerâ = lower(â)
    lowerβ = lower(β)
    diagγ𝟙 = vec(sum(diag(γ), dims = 2))
    â𝟙 = vec(sum(â, dims = 2))
    β⁻¹𝟙 = vec(sum(triangular_inv(β), dims = 2))
    âγ𝟙 = vec(sum(âγ, dims = 2))
    diagâ𝟙 = vec(sum(diag(â), dims = 2))
    values = map(
        to_enumerated_rows,
        (; lowerâ, lowerβ, β, diagγ𝟙, â𝟙, β⁻¹𝟙, âγ𝟙, diagâ𝟙),
    )
    return :($values)
end

function rosenbrock_step_u!(integrator, cache, g::ForwardEulerODEFunction)
    (; u, p, t, dt, alg) = integrator
    (; update_jac, multiply!, set_Δtγ!) = alg
    (; update_jac_cache, linsolve!) = cache
    (; Û⁺ᵢ, Vs, Fs, W, ḟ) = cache._cache
    (; lowerâ, lowerβ, β, diagγ𝟙, â𝟙, β⁻¹𝟙, âγ𝟙, diagâ𝟙) =
        precomputed_values(typeof(alg), typeof(g))
    function jac_func(Ûᵢ, T̂ᵢ, γᵢᵢ)
        g.Wfact(W, Ûᵢ, p, dt * γᵢᵢ, T̂ᵢ)
        !isnothing(g.tgrad) && g.tgrad(ḟ, Ûᵢ, p, T̂ᵢ)
    end
    function stage_func(::Val{i}) where {i}
        γᵢᵢ = diagγ𝟙[i]
        Ûᵢ = i == 1 ? u : Û⁺ᵢ
        T̂ᵢ = i == 1 ? t : t + dt * â𝟙[i]

        run!(update_jac, update_jac_cache, NewStage(), jac_func, Ûᵢ, T̂ᵢ, γᵢᵢ)

        # Vᵢ = (Δt * γᵢᵢ * J - I)⁻¹ * g(
        #     u - Δt * γᵢᵢ * J * u * ∑_{j=1}^i (β⁻¹)ᵢⱼ +
        #     Δt * ∑_{j=1}^{i-1} âᵢⱼ * f(Ûⱼ, T̂ⱼ) +
        #     Δt² * ḟ * ∑_{j=1}^i (â * γ)ᵢⱼ + ∑_{j=1}^{i-1} βᵢⱼ * Vⱼ,
        #     Ûᵢ,
        #     T̂ᵢ
        #     Δt * âᵢᵢ,
        # )
        set_Δtγ!(W, dt * γᵢᵢ * β⁻¹𝟙[i], dt * γᵢᵢ)
        multiply!(Vs[i], W, u)
        set_Δtγ!(W, dt * γᵢᵢ, dt * γᵢᵢ * β⁻¹𝟙[i])
        Vs[i] .= broadcasted(
            +,
            broadcasted(-, Vs[i]),
            scaled_linear_combination(lowerâ[i], Fs, dt)...,
            (isnothing(g.tgrad) ? () : (broadcasted(*, dt^2 * âγ𝟙[i], ḟ),))...,
            linear_combination(lowerβ[i], Vs)...,
        )
        Fs[i] .= Vs[i]
        g(Vs[i], Ûᵢ, p, T̂ᵢ, dt * diagâ𝟙[i])
        Fs[i] .= (Fs[i] .- Vs[i]) ./ (dt * diagâ𝟙[i])
        linsolve!(Vs[i], W, Vs[i]) # assume that linsolve! can handle aliasing

        # Û⁺ᵢ = -∑_{j=1}^i βᵢⱼ * Vᵢ
        Û⁺ᵢ .= scaled_linear_combination(β[i], Vs, -1)[1]
    end

    run!(update_jac, update_jac_cache, NewStep(), jac_func, u, t, diagγ𝟙[1])
    foreachval(stage_func, Val(num_stages(typeof(alg))))
    u .= Û⁺ᵢ
end

function rosenbrock_step_u!(integrator, cache, f)
    (; u, p, t, dt, alg) = integrator
    (; update_jac) = alg
    (; update_jac_cache, linsolve!) = cache
    (; Û⁺ᵢ, Ks, W, ḟ) = cache._cache
    (; lowerγ⁻¹, âγ⁻¹, diagγ𝟙, γ𝟙, â𝟙) =
        precomputed_values(typeof(alg), typeof(f))
    function jac_func(Ûᵢ, T̂ᵢ, γᵢᵢ)
        f.Wfact(W, Ûᵢ, p, dt * γᵢᵢ, T̂ᵢ)
        !isnothing(f.tgrad) && f.tgrad(ḟ, Ûᵢ, p, T̂ᵢ)
    end
    function stage_func(::Val{i}) where {i}
        γᵢᵢ = diagγ𝟙[i]
        Ûᵢ = i == 1 ? u : Û⁺ᵢ
        T̂ᵢ = i == 1 ? t : t + dt * â𝟙[i]

        run!(update_jac, update_jac_cache, NewStage(), jac_func, Ûᵢ, T̂ᵢ, γᵢᵢ)

        # Kᵢ = (Δt * γᵢᵢ * J - I)⁻¹ * γᵢᵢ *
        #      (f(Ûᵢ, T̂ᵢ) + ∑_{j=1}^{i-1} (γ⁻¹)ᵢⱼ * Kⱼ + Δt * ḟ * ∑_{j=1}^i γᵢⱼ)
        f(Ks[i], Ûᵢ, p, T̂ᵢ)
        Ks[i] .= γᵢᵢ .* broadcasted(
            +,
            Ks[i],
            linear_combination(lowerγ⁻¹[i], Ks)...,
            (isnothing(f.tgrad) ? () : (broadcasted(*, dt * γ𝟙[i], ḟ),))...,
        )
        linsolve!(Ks[i], W, Ks[i]) # assume that linsolve! can handle aliasing

        # Û⁺ᵢ = u - Δt * ∑_{j=1}^i (â * γ⁻¹)ᵢⱼ * Kⱼ
        Û⁺ᵢ .= broadcasted(+, u, scaled_linear_combination(âγ⁻¹[i], Ks, -dt)...)
    end

    run!(update_jac, update_jac_cache, NewStep(), jac_func, u, t, diagγ𝟙[1])
    foreachval(stage_func, Val(num_stages(typeof(alg))))
    u .= Û⁺ᵢ
end
