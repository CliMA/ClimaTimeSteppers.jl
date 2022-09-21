export Rosenbrock23, SSPKnoth, RODASP2
export ROS3w, ROS3Pw, ROS34PW1a, ROS34PW1b, ROS34PW2, ROS34PW3

using StaticArrays: @SArray

#=
Rosenbrock23 in OrdinaryDiffEq.jl and ode23s in MATLAB

Rosenbrock-W method with 2 stages and 2nd order convergence

From Section 4.1 of "The MATLAB ODE Suite" by L. F. Shampine and M. W. Reichelt

The paper does not directly provide the method coefficients, but, converting to
our notation, it specifies that
    Û₁ = u, T̂₁ = t,
    F₁ = (I - Δt * 1/(2+√2) * J)⁻¹ * (f(Û₁, T̂₁) + Δt * 1/(2+√2) * ḟ),
    Û₂ = u + Δt * 1/2 * F₁, T̂₂ = t + Δt * 1/2,
    F₂ = (I - Δt * 1/(2+√2) * J)⁻¹ * (f(Û₂, T̂₂) - F₁ + Δt * 0 * ḟ) + F₁, and
    u_next = u + Δt * (0 * F₁ + 1 * F₂)
This implies the following table of coefficients:
=#
const Rosenbrock23 = RosenbrockAlgorithm{
    @SArray([
         1/(2+√2) 0;
        -1/(2+√2) 1/(2+√2);
    ]),
    @SArray([
        0   0;
        1/2 0;
    ]),
    @SArray([0, 1]),
}

#=
Rosenbrock-W method with 3 stages and 2nd order convergence from Oswald Knoth
=#
const SSPKnoth = RosenbrockAlgorithm{
    @SArray([
         1    0    0;
         0    1    0;
        -3/4 -3/4  1;
    ]),
    @SArray([
        0   0   0;
        1   0   0;
        1/4 1/4 0;
    ]),
    @SArray([1/6, 1/6, 2/3]),
}

#=
An improved version of RODASP, which is itself an improved version of RODAS

Rosenbrock-W method with 6 stages and 4th order convergence (reduces to 2nd
order for inexact Jacobians)

From Table 3 of "Improvement of Rosenbrock-Wanner Method RODASP" by G.
Steinebach

The paper calls a "α" and specifies β = α + γ instead of γ.
=#
const RODASP2 = begin
    α = @SArray([
         0                     0                     0                     0                     0                     0;
         3/4                   0                     0                     0                     0                     0;
         3.688749816109670e-1 -4.742684759792117e-2  0                     0                     0                     0;
         4.596170083041160e-1  2.724432453018110e-1 -2.123145213282008e-1  0                     0                     0;
         2.719770298548111e+0  1.358873794835473e+0 -2.838824065018641e+0 -2.398200283649438e-1  0                     0;
        -6.315720511779362e-1 -3.326966988718489e-1  1.154688683864918e+0  5.595800661848674e-1  1/4                   0;
    ])
    β = @SArray([
         1/4                   0                     0                     0                     0                     0;
         0                     1/4                   0                     0                     0                     0;
        -9.184372116108780e-2 -2.624106318888223e-2  1/4                   0                     0                     0;
        -5.817702768270960e-2 -1.382129630513952e-1  5.517478318046004e-1  1/4                   0                     0;
        -6.315720511779359e-1 -3.326966988718489e-1  1.154688683864917e+0  5.595800661848674e-1  1/4                   0;
         1.464968119068509e-1  8.896159691002870e-2  1.648843942975147e-1  4.568000540284631e-1 -1.071428571428573e-1  1/4;
    ])
    RosenbrockAlgorithm{
        β - α,
        α,
        vec(β[end, :]),
    }
end

################################################################################

#=
Methods from "New Rosenbrock W-Methods of Order 3 for Partial Differential
Algebraic Equations of Index 1" by J Rang and L. Angermann

Each method is named ROS3[4][P][w/W][...], where "ROS3" indicates a Rosenbrock
method of order 3, "4" indicates that the method has 4 stages (otherwise, it has
3 stages), "P" indicates that the method is suited for parabolic problems, "w"
or "W" indicates that the method can handle an approximate Jacobian, and some of
the names end with additional identifying numbers and symbols.

ROS3w and ROS3Pw both reduce to order 2 for inexact Jacobians.
ROS34PW3 is actually of order 4 but reduces to order 3 for inexact Jacobians.
=#
const ROS3w = RosenbrockAlgorithm{
    @SArray([
         4.358665215084590e-1  0                     0;
         3.635068368900681e-1  4.358665215084590e-1  0;
        −8.996866791992636e-1 −1.537997822626885e-1  4.358665215084590e-1;
    ]),
    @SArray([
        0   0   0;
        2/3 0   0;
        2/3 0   0;
    ]),
    @SArray([1/4, 1/4, 1/2]),
}
const ROS3Pw = RosenbrockAlgorithm{
    @SArray([
         7.8867513459481287e-1  0                      0;
        −1.5773502691896257e+0  7.8867513459481287e-1  0;
        −6.7075317547305480e-1 −1.7075317547305482e-1  7.8867513459481287e-1;
    ]),
    @SArray([
        0                     0                     0;
        1.5773502691896257e+0 0                     0;
        1/2                   0                     0;
    ]),
    @SArray([1.0566243270259355e-1, 4.9038105676657971e-2, 8.4529946162074843e-1]),
}
const ROS34PW1a = RosenbrockAlgorithm{
    @SArray([
         4.358665215084590e-1  0                     0                     0;
        −2.218787467653286e+0  4.358665215084590e-1  0                     0;
        −9.461966143940745e-2 −7.913526735718213e-3  4.358665215084590e-1  0;
        −1.870323744195384e+0 −9.624340112825115e-2  2.726301276675511e-1  4.358665215084590e-1;
    ]),
    @SArray([
        0                    0                    0                    0;
        2.218787467653286e+0 0                    0                    0;
        0                    0                    0                    0;
        1.208587690772214e+0 7.511610241919324e-2 1/2                  0;
    ]),
    @SArray([3.285609536316354e-1, −5.785609536316354e-1, 1/4, 1]),
}
const ROS34PW1b = RosenbrockAlgorithm{
    @SArray([
         4.358665215084590e-1  0                     0                     0;
        −2.218787467653286e+0  4.358665215084590e-1  0                     0;
        −2.848610224639349e+0 −5.267530183845237e-2  4.358665215084590e-1  0;
        −1.128167857898393e+0 −1.677546870499461e-1  5.452602553351021e-2  4.358665215084590e-1;
    ]),
    @SArray([
        0                    0                    0                    0;
        2.218787467653286e+0 0                    0                    0;
        2.218787467653286e+0 0                    0                    0;
        1.453923375357884e+0 0                    1/10                 0;
    ]),
    @SArray([5.495647928937977e-1, −5.507258170857301e-1, 1/4, 7.511610241919324e-1]),
}
const ROS34PW2 = RosenbrockAlgorithm{
    @SArray([
         4.3586652150845900e-1  0                      0                      0;
        −8.7173304301691801e-1  4.3586652150845900e-1  0                      0;
        −9.0338057013044082e-1  5.4180672388095326e-2  4.3586652150845900e-1  0;
         2.4212380706095346e-1 −1.2232505839045147e+0  5.4526025533510214e-1  4.3586652150845900e-1;
    ]),
    @SArray([
        0                      0                      0                      0;
        8.7173304301691801e-1  0                      0                      0;
        8.4457060015369423e-1 −1.1299064236484185e-1  0                      0;
        0                      0                      1                      0;
    ]),
    @SArray([2.4212380706095346e-1, −1.2232505839045147e+0, 1.5452602553351020e+0, 4.3586652150845900e-1]),
}
const ROS34PW3 = RosenbrockAlgorithm{
    @SArray([
         1.0685790213016289e+0  0                      0                      0;
        −2.5155456020628817e+0  1.0685790213016289e+0  0                      0;
        −8.7991339217106512e-1 −9.6014187766190695e-1  1.0685790213016289e+0  0;
        −4.1731389379448741e-1  4.1091047035857703e-1 −1.3558873204765276e+0  1.0685790213016289e+0;
    ]),
    @SArray([
        0                      0                      0                      0;
        2.5155456020628817e+0  0                      0                      0;
        5.0777280103144085e-1  3/4                    0                      0;
        1.3959081404277204e-1 −3.3111001065419338e-1  8.2040559712714178e-1  0;
    ]),
    @SArray([2.2047681286931747e-1, 2.7828278331185935e-3, 7.1844787635140066e-3, 7.6955588053404989e-1]),
}
