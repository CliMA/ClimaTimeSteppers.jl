export ARS343

struct ARS{name,L} <: DistributedODEAlgorithm
    linsolve::L
end
ARS{name}(;linsolve) where {name} = ARS343{name,typeof(linsolve)}(linsolve)



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

#=
# 232
γ = (2 - sqrt(2))
δ = -2*sqrt(2)/3
# implicit
a = [γ 0;
     1-γ γ];
# explicit
ahat = [ 0 0 0;
         γ 0 0;
        δ 1-δ 0]

=#

const ARSIMEXEuler = ARS{:IMEXEuler}
function tableau(::ARSIMEXEuler, RT)
    # this is the same as used in OrdinaryDiffEq
    # https://github.com/SciML/OrdinaryDiffEq.jl/issues/1590
    #  Ascher, Ruuth, and Wetton (1995)
    a = RT[1;
           1;;]
    ahat = RT[0 0;
              1 0;
              1 0]
    return ARSTableau(a, ahat)


    # the method of  Araújo, Murua, Sanz-Serna (1997).
    # would be
    #=
    a = RT[1;
           1;;]
    ahat = RT[0 0;
              0 0;
              1 0]
    =#
end

const ARS343 = ARS{:ARS343}
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

function step_u!(int, cache::ARSCache)

    f = int.prob.f
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

    Nstages = length(tab.c) - 1

    # Update W
    # @show tab.γ dt
    Wfact!(W, u, p, dt*tab.γ, t)
    # @show W

    # implicit eqn:
    #   ux = u + dt * f(ux, p, t)
    # Newton iteration:
    #   ux <- ux + (I - dt J) \ (u + dt f(ux, p, t) - ux)
    # initial iteration
    #   ux <- u + dt (I - dt J) \ f(u, p, t)

    function implicit_step!(ux, u, p, t, dt)

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
    if tab.ahat[2,1] != 0
        f2!(Uhat[1], u, p, t+dt*tab.chat[1], dt*tab.ahat[2,1])
    end
    # implicit
    implicit_step!(U[1], Uhat[1], p, t+dt*tab.c[1], dt*tab.a[1,1])

    #### stage 2
    uhat = Nstages == 1 ? u : Uhat[2]
    uhat .= tab.Q0[2] .* u .+
            tab.Qhat[2,1] .* Uhat[1] .+ tab.Q[2,1] .* U[1] # utilde[2]
    if tab.ahat[3,2] != 0
        f2!(uhat, U[1], p, t+dt*tab.chat[2], dt*tab.ahat[3,2])
    end
    Nstages == 1 && return

    implicit_step!(U[2], Uhat[2], p, t+dt*tab.c[2], dt*tab.a[2,2])

    #### stage 3
    uhat = Nstages == 2 ? u : Uhat[3]
    uhat .= tab.Q0[3] .* u .+
            tab.Qhat[3,1] .* Uhat[1] .+ tab.Q[3,1] .* U[1] .+
            tab.Qhat[3,2] .* Uhat[2] .+ tab.Q[3,2] .* U[2] # utilde[3]
    if tab.ahat[4,3] != 0
        f2!(uhat, U[2], p, t+dt*tab.chat[3], dt*tab.ahat[4,3])
    end
    Nstages == 2 && return

    implicit_step!(U[3], Uhat[3], p, t+dt*tab.c[3], dt*tab.a[3,3])
    # @show U[3] t+dt*tab.c[3]

    ### final update
    @assert Nstages == 3
    uhat = u
    uhat .= tab.Q0[4] .* u .+
    tab.Qhat[4,1] .* Uhat[1] .+ tab.Q[4,1] .* U[1] .+
    tab.Qhat[4,2] .* Uhat[2] .+ tab.Q[4,2] .* U[2] .+
    tab.Qhat[4,3] .* Uhat[3] .+ tab.Q[4,3] .* U[3]

    if tab.ahat[5,4] != 0
        f2!(uhat, U[3], p, t+dt*tab.chat[4], dt*tab.ahat[5,4])
    end
    return
end
