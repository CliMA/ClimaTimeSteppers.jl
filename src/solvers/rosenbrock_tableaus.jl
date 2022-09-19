export Rosenbrock23, SSPKnoth, ROS3, RODAS3

using StaticArrays: @SArray

#=
Rosenbrock23 in OrdinaryDiffEq and ode23s in MATLAB (2 stages, 3rd order
convergence)

From Section 4.1 of "The MATLAB ODE Suite" by L. F. Shampine and M. W. Reichelt

The paper does not directly give the method coefficients, but, converting to our
notation, it specifies that
    Û₁ = u, T̂₁ = t,
    F₁ = (I - Δt * 1/(2+√2) * J)⁻¹ * (f(Û₁, T̂₁) + Δt * 1/(2+√2) * ḟ),
    Û₂ = u + Δt * 1/2 * F₁, T̂₂ = t + Δt * 1/2,
    F₂ = (I - Δt * 1/(2+√2) * J)⁻¹ * (f(Û₂, T̂₂) - F₁ + Δt * 0 * ḟ) + F₁, and
    u_next = u + Δt * (0 * F₁ + 1 * F₂)
This implies the following table of coefficients:
=#
const Rosenbrock23 = RosenbrockAlgorithm{
    @SArray([
        1/(2+√2)  0;
        -1/(2+√2) 1/(2+√2);
    ]),
    @SArray([
        0   0;
        1/2 0;
    ]),
    @SArray([0, 1]),
}

#=
Custom Rosenbrock scheme from Oswald Knoth
=#
const SSPKnoth = RosenbrockAlgorithm{
    @SArray([
        1    0    0;
        0    1    0;
        -3/4 -3/4 1;
    ]),
    @SArray([
        0   0   0;
        1   0   0;
        1/4 1/4 0;
    ]),
    @SArray([1/6, 1/6, 2/3]),
}

#=
3 stages, 3rd order convergence

From Section 4 of "Benchmarking Stiff ODE Solvers" by A. Sandu et. al.
=#
const ROS3 = begin
    γᵢᵢ =  0.43586652150845899941601945119356
    γ₂₁ = -0.19294655696029095575009695436041
    γ₃₂ =  1.74927148125794685173529749738960
    b₁  = -0.75457412385404315829818998646589
    b₂  =  1.94100407061964420292840123379419
    b₃  = -0.18642994676560104463021124732829
    RosenbrockAlgorithm{
        @SArray([
            γᵢᵢ 0   0;
            γ₂₁ γᵢᵢ 0;
            0   γ₃₂ γᵢᵢ;
        ]),
        @SArray([
            0   0   0;
            γᵢᵢ 0   0;
            γᵢᵢ 0   0;
        ]),
        @SArray([b₁, b₂, b₃]),
    }
end

#=
4 stages, 3rd order convergence

From Section 4 of "Benchmarking Stiff ODE Solvers" by A. Sandu et. al.
=#
const RODAS3 = RosenbrockAlgorithm{
    @SArray([
        1/2  0    0    0;
        1    1/2  0    0;
        -1/4 -1/4 1/2  0;
        1/12 1/12 -2/3 1/2;
    ]),
    @SArray([
        0    0    0    0;
        0    0    0    0;
        1    0    0    0;
        3/4  -1/4 1/2  0;
    ]),
    @SArray([5/6, -1/6, -1/6, 1/2]),
}
