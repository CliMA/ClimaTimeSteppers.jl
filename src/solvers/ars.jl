export ARS111, ARS121, ARS232, ARS343

struct ARSAlgorithm{name,L} <: DistributedODEAlgorithm
    linsolve::L
end
ARSAlgorithm{name}(;linsolve) where {name} = ARSAlgorithm{name,typeof(linsolve)}(linsolve)



struct ARSTableau{RT}
    a::Matrix{RT}
    ahat::Matrix{RT}
    c::Vector{RT}
    chat::Vector{RT}
    Q0::Vector{RT}
    Q::Matrix{RT}
    Qhat::Matrix{RT}
    γ::RT
end
function ARSTableau(a::AbstractMatrix{RT}, ahat::AbstractMatrix{RT}) where {RT}
    #=
    utilde[i] = u0 +
    dt * sum(j -> ahat[i+1,j] * g(U[j-1]), 1:i-1) +
    dt * sum(j -> a[i,j]      * f(U[j]), 1:i-1)

    Uhat[i] = utilde[i] + dt * ahat[i+1,i] g(U[i-1])
    U[i] = Uhat[i] + dt * a[i,i] f(U[i])
        ≈ Uhat[i] + dt * a[i,i] (I - dt * a[i,i] * J) \ f(U[i])

    we don't store g or f, or utilde

    g(U[i-1]) = (Uhat[i] - utilde[i]) / (dt * ahat[i+1,i])
    f(U[i]) = (U[i] - Uhat[i]) / (dt * a[i,i])

    utilde[i] = u0 +
    sum(j -> (ahat[i+1,j]/ahat[j+1,j]) * (Uhat[j] - utilde[j]), 1:i-1) +
    sum(j -> (a[i,j]/a[j,j]) (U[j] - Uhat[j]), 1:i-1)

    write this as

    utilde[i] = Q0[i] +
    sum(k -> Qhat[i,k] * Uhat[k], 1:i-1) +
    sum(k -> Q[i,k] * U[k], 1:i-1)

    =#

    N = size(a,2)

    c = vec(sum(a, dims=2))
    chat = vec(sum(ahat, dims=2))

    Q = zeros(RT, N+1,N)
    Q0 = zeros(RT,N+1)
    Qhat = zeros(RT, N+1,N)

    for i = 1:N+1
      Q0[i] = 1 - sum(j -> ahat[i+1,j] / ahat[j+1,j] * Q0[j], 1:i-1;init=0.0)
      for k = 1:i-1
        Q[i, k] = a[i,k]/a[k,k] - sum(j -> ahat[i+1,j] / ahat[j+1,j] * Q[j,k], 1:i-1)
        Qhat[i,k] = ahat[i+1,k]/ahat[k+1,k] - a[i,k]/a[k,k] - sum(j -> ahat[i+1,j] / ahat[j+1,j] * Qhat[j,k], 1:i-1)
      end
    end

    for i = 1:N+1
      @assert Q0[i] + sum(Q[i,:]) + sum(Qhat[i,:]) ≈ 1
    end

    γ = a[1,1]
    for i = 2:N
      @assert a[i,i] == γ
    end
    ARSTableau(a, ahat, c, chat, Q0, Q, Qhat, γ)
end


"""
    ARS111

The Forward-Backward (1,1,1) implicit-explicit (IMEX) Runge-Kutta scheme of
[ARS1997](@cite), section 2.1.

This is equivalent to the `OrdinaryDiffEq.IMEXEuler` algorithm.
"""
const ARS111 = ARSAlgorithm{:ARS111}
function tableau(::ARS111, RT)
    # https://github.com/SciML/OrdinaryDiffEq.jl/issues/1590
    a = RT[1;
           1;;]
    ahat = RT[0 0;
              1 0;
              1 0]
    return ARSTableau(a, ahat)
end

"""
    ARS121

The Forward-Backward (1,2,1) implicit-explicit (IMEX) Runge-Kutta scheme of
[ARS1997](@cite), section 2.2.

This is equivalent to the `OrdinaryDiffEq.IMEXEulerARK` algorithm.
"""
const ARS121 = ARSAlgorithm{:ARS121}
function tableau(::ARS121, RT)
    a = RT[1;
           1;;]
    ahat = RT[0 0;
              1 0;
              0 1]
    return ARSTableau(a, ahat)
end

"""
    ARS232

The Forward-Backward (2,3,2) implicit-explicit (IMEX) Runge-Kutta scheme of
[ARS1997](@cite), section 2.5.
"""
const ARS232 = ARSAlgorithm{:ARS232}
function tableau(::ARS232, RT)
    γ = (2 - sqrt(2))/2
    δ = -2*sqrt(2)/3
    # implicit
    a = RT[γ 0;
        1-γ γ;
        1-γ γ];
    # explicit
    ahat = RT[ 0 0 0;
            γ 0 0;
            δ 1-δ 0;
            0 1-γ γ];
    return ARSTableau(a, ahat)
end

"""
    ARS343

The L-stable, third-order (3,4,3) implicit-explicit (IMEX) Runge-Kutta scheme of
[ARS1997](@cite), section 2.7.
"""
const ARS343 = ARSAlgorithm{:ARS343}
function tableau(::ARS343, RT)
    N = 3
    γ = 0.4358665215084590
    b1 = -3/2 * γ^2 + 4 * γ - 1/4
    b2 =  3/2 * γ^2 - 5 * γ + 5/4
    # implicit tableau
    a = RT[γ 0 0;
        (1-γ)/2 γ 0;
        b1 b2 γ;
        b1 b2 γ]

    dA42 = 0.5529291480359398;
    dA43 = 0.5529291480359398;
    dA31 =(
        (1.0 - 4.5 * γ + 1.5 * γ * γ) * dA42
          + (2.75 - 10.5 * γ + 3.75 * γ * γ) * dA43
          - 3.5 + 13 * γ - 4.5 * γ * γ)
    dA32 = (
        (-1.0 + 4.5 * γ - 1.5 * γ * γ) * dA42
          + (-2.75 + 10.5 * γ - 3.75 * γ * γ) * dA43
          + 4.0 - 12.5 * γ + 4.5 * γ * γ)
    dA41 = 1.0 - dA42 - dA43;
    # explicit tableau
    ahat = RT[
      0 0 0 0;
      γ 0 0 0;
      dA31 dA32 0 0;
      dA41 dA42 dA43 0;
      0 b1 b2 γ
    ]
    ARSTableau(a, ahat)
end

struct ARSCache{Nstages, RT, A}
    tableau::ARSTableau{RT}
    U::NTuple{Nstages, A}
    Uhat::NTuple{Nstages, A}
    idu::A
    W
    linsolve!
end

function cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::ARSAlgorithm; kwargs...)

    tab = tableau(alg, eltype(prob.u0))
    Nstages = length(tab.c) - 1
    U = ntuple(i -> similar(prob.u0), Nstages)
    Uhat = ntuple(i -> similar(prob.u0), Nstages)
    idu = similar(prob.u0)
    W = prob.f.f1.jac_prototype
    linsolve! = alg.linsolve(Val{:init}, W, prob.u0; kwargs...)

    return ARSCache(tab, U, Uhat, idu, W, linsolve!)
end

function step_u!(int, cache::ARSCache{Nstages}) where {Nstages}

    f = int.sol.prob.f
    f1! = f.f1
    f2! = f.f2
    Wfact! = f1!.Wfact

    u = int.u
    p = int.p
    t = int.t
    dt = int.dt

    tab = cache.tableau
    W = cache.W
    U = cache.U
    Uhat = cache.Uhat
    idu = cache.idu
    linsolve! = cache.linsolve!

    # Update W
    # Wfact!(W, u, p, dt*tab.γ, t)


    # implicit eqn:
    #   ux = u + dt * f(ux, p, t)
    # Newton iteration:
    #   ux <- ux + (I - dt J) \ (u + dt f(ux, p, t) - ux)
    # initial iteration
    #   ux <- u + dt (I - dt J) \ f(u, p, t)

    function implicit_step!(ux, u, p, t, dt)
        Wfact!(W, u, p, dt, t)
        # currently this does just a single Newton iteration
        # and assumes we compute f(u, p, t) directly
        # need to figure out how we would do multiple iterations
        # du = f(u, p, t)
        f1!(ux, u, p, t)
        # @show ux
        # solve for idu = (I - dt J) \ du
        linsolve!(idu, W, ux)
        # @show W
        # @show idu
        @. ux = u + dt * idu
        # @show ux
    end

    #### stage 1
    # explicit
    Uhat[1] .= u # utilde[i],  Q0[1] == 1
    f2!(Uhat[1], u, p, t+dt*tab.chat[1], dt*tab.ahat[2,1])

    # implicit
    implicit_step!(U[1], Uhat[1], p, t+dt*tab.c[1], dt*tab.a[1,1])
    if Nstages == 1
        u .= tab.Q0[2] .* u .+
            tab.Qhat[2,1] .* Uhat[1] .+ tab.Q[2,1] .* U[1] # utilde[2]
        f2!(u, U[1], p, t+dt*tab.chat[2], dt*tab.ahat[3,2])
        return
    end

    #### stage 2
    Uhat[2] .= tab.Q0[2] .* u .+
            tab.Qhat[2,1] .* Uhat[1] .+ tab.Q[2,1] .* U[1] # utilde[2]
    f2!(Uhat[2], U[1], p, t+dt*tab.chat[2], dt*tab.ahat[3,2])

    implicit_step!(U[2], Uhat[2], p, t+dt*tab.c[2], dt*tab.a[2,2])

    if Nstages == 2
        u .= tab.Q0[3] .* u .+
            tab.Qhat[3,1] .* Uhat[1] .+ tab.Q[3,1] .* U[1] .+
            tab.Qhat[3,2] .* Uhat[2] .+ tab.Q[3,2] .* U[2] # utilde[3]
        f2!(u, U[2], p, t+dt*tab.chat[3], dt*tab.ahat[4,3])
        return
    end

    #### stage 3
    Uhat[3] .= tab.Q0[3] .* u .+
            tab.Qhat[3,1] .* Uhat[1] .+ tab.Q[3,1] .* U[1] .+
            tab.Qhat[3,2] .* Uhat[2] .+ tab.Q[3,2] .* U[2] # utilde[3]
    f2!(Uhat[3], U[2], p, t+dt*tab.chat[3], dt*tab.ahat[4,3])
    # @show Uhat[3] t+dt*tab.chat[3]

    implicit_step!(U[3], Uhat[3], p, t+dt*tab.c[3], dt*tab.a[3,3])
    # @show U[3] t+dt*tab.c[3]

    ### final update
    u .= tab.Q0[4] .* u .+
    tab.Qhat[4,1] .* Uhat[1] .+ tab.Q[4,1] .* U[1] .+
    tab.Qhat[4,2] .* Uhat[2] .+ tab.Q[4,2] .* U[2] .+
    tab.Qhat[4,3] .* Uhat[3] .+ tab.Q[4,3] .* U[3]

    # @show u
    f2!(u, U[3], p, t+dt*tab.chat[4], dt*tab.ahat[5,4])
    # @show u t+dt*tab.chat[4]
    return
end
