export RKAlgorithm, ARKAlgorithm

#= Derivation of ARK timestepper formulation (TODO: Move to docs)

Coefficient Definitions:
    a_exp, a_imp ∈ R^{s×s}
    b_exp, b_imp, c_exp, c_imp ∈ R^s
    a_imp is lower triangular, and a_exp is strictly lower triangular

    A_exp := vcat(a_exp, b_exp')
    A_imp := vcat(a_imp, b_imp')

    Γ := diag(A_imp)
    DA_imp := vcat(Diagonal(A_imp), zeros(s)')
    LA_imp := A_imp - DA_imp

Original Algorithm:
for i ∈ 1:(s + 1)
    U_L[i] = u + Δt * (dot(A_exp[i, :], F_exp) + dot(LA_imp[i, :], F_imp))
    if i == s + 1
        u_next = U_L[i]
    else
        t_exp = t + Δt * c_exp[i]
        t_imp = t + Δt * c_imp[i]
        U[i] = findmin(x -> |U_L[i] - x + Δt * Γ[i] * f_imp(x, t_imp)|)
        F_exp[i] = f_exp(U[i], t_exp)
        F_imp[i] = f_imp(U[i], t_imp)
        # This value of F_imp[i] is inconsistent with U[i] if findmin is not
        # exact, so we replaced it with (U[i] - U_L[i]) / (Δt * Γ[i]) when
        # Γ[i] is not 0.
    end
end

Reformulation:
    In order to avoid inconsistencies between U[i] and F_imp[i], we want to
    avoid computing F_imp[i] on stages with implicit solves (i.e., stages where
    Γ[i] is not 0).

    S_z := findall(iszero, Γ) # stages without implicit solves
    S_nz := findall(!iszero, Γ) # stages with implicit solves

    F_imp_z := (V = zero(F_imp); V[S_z] = F_imp[S_z]; V)
    F_imp_nz := (V = zero(F_imp); V[S_nz] = F_imp[S_nz]; V)
    I_z := (M = zero(a_imp); M[S_z, S_z] = I; M)
    A_imp_z := (M = zero(A_imp); M[:, S_z] = A_imp[:, S_z]; M)
    A_imp_nz := A_imp - A_imp_z
    LA_imp_nz := A_imp_nz - DA_imp
    a_imp_nz := A_imp_nz[1:s, :]
    ΔU_imp_nz := Δt * a_imp_nz * F_imp_nz

    I_z * F_imp_nz = 0
    A_imp_z * F_imp_nz = 0
    A_imp_nz * F_imp_z = 0
    DA_imp * F_imp_z = 0
    I_z * ΔU_imp_nz = 0

    Since U_L = u .+ Δt * (A_exp * F_exp + LA_imp * F_imp), we want to express
    Δt * LA_imp * F_imp in terms of F_imp_z and ΔU_imp_nz. To do this, we must
    assume that I_z + a_imp_nz is invertible.

    Δt * F_imp_nz =
        = Δt * inv(I_z + a_imp_nz) * (I_z + a_imp_nz) * F_imp_nz
        = inv(I_z + a_imp_nz) * Δt * a_imp_nz * F_imp_nz
        = inv(I_z + a_imp_nz) * ΔU_imp_nz
        = (inv(I_z + a_imp_nz) - I_z) * ΔU_imp_nz

    G_imp_nz := LA_imp_nz * (inv(I_z + a_imp_nz) - I_z)

    Δt * LA_imp * F_imp =
        = Δt * LA_imp * (F_imp_z + F_imp_nz)
        = Δt * (A_imp_z * F_imp_z + LA_imp_nz * F_imp_nz)
        = Δt * A_imp_z * F_imp_z + G_imp_nz * ΔU_imp_nz

Reformulated Algorithm:
for i ∈ 1:(s + 1)
    U_L[i] =
        u + Δt * (dot(A_exp[i, :], F_exp) + dot(A_imp_z[i, :], F_imp_z)) +
        dot(G_imp_nz[i, :], ΔU_imp_nz)
    if i == s + 1
        u_next = U_L[i]
    else
        t_exp = t + Δt * c_exp[i]
        t_imp = t + Δt * c_imp[i]
        if i ∈ S_z
            U[i] = U_L[i] # no need to use findmin, since Γ[i] is 0
        else
            U[i] = findmin(x -> |U_L[i] - x + Δt * Γ[i] * f_imp(x, t_imp)|)
        end
        F_exp[i] = f_exp(U[i], t_exp)
        if i ∈ S_z
            F_imp_z[i] = f_imp(U[i], t_imp) # never inconsistent with U[i]
        else
            ΔU_imp_nz[i] = U[i] - U_L[i] + dot(G_imp_nz[i, :], ΔU_imp_nz)
        end
    end
end
=#

"""
    RKAlgorithm(tableau)
    RKAlgorithm(name)

Constructs a Runge-Kutta (RK) algorithm for solving ODEs. The first constructor
accepts any `RKTableau` and leaves the algorithm unnamed, while the second
determines the tableau from an `RKAlgorithmName`. Each of these constructors
just makes an `ARKAlgorithm` with identical `lim`, `exp`, and `imp` tableaus.
"""
RKAlgorithm(tableau::RKTableau) = ARKAlgorithm(ARKTableau(tableau))
RKAlgorithm(name::RKAlgorithmName) = ARKAlgorithm(name, ARKTableau(RKTableau(name)), nothing)

"""
    ARKAlgorithm(tableau, [newtons_method])
    ARKAlgorithm(name, [newtons_method])

Constructs an additive Runge-Kutta (ARK) algorithm for solving ODEs. The first
constructor accepts any `ARKTableau` and leaves the algorithm unnamed, while the
second determines the tableau from an `ARKAlgorithmName`. If the specified `imp`
tableau necessitates the use of an implicit solver for the problem that this
algorithm will solve, a `NewtonsMethod` must also be specified.
"""
struct ARKAlgorithm{N <: Union{Nothing, AbstractAlgorithmName}, T <: ARKTableau, NM <: Union{Nothing, NewtonsMethod}} <:
       DistributedODEAlgorithm
    name::N
    tableau::T
    newtons_method::NM
end
ARKAlgorithm(tableau_or_name) = ARKAlgorithm(tableau_or_name, nothing)
ARKAlgorithm(tableau::ARKTableau, newtons_method) = ARKAlgorithm(nothing, tableau, newtons_method)
ARKAlgorithm(name::ARKAlgorithmName, newtons_method) = ARKAlgorithm(name, ARKTableau(name), newtons_method)
ARKAlgorithm(name::ARKRosenbrockAlgorithmName, _) = ARKAlgorithm(name, ARKTableau(name), nothing)

has_jac(T_imp!) =
    hasfield(typeof(T_imp!), :Wfact) &&
    hasfield(typeof(T_imp!), :jac_prototype) &&
    !isnothing(T_imp!.Wfact) &&
    !isnothing(T_imp!.jac_prototype)

imp_error(name) = error("$(isnothing(name) ? "The given ARKTableau" : name) \
                         has implicit stages that require a nonlinear solver, \
                         so NewtonsMethod must be specified alongside T_imp!.")

sdirk_error(name) = error("$(isnothing(name) ? "The given ARKTableau" : name) \
                           has implicit stages with distinct coefficients (it \
                           is not SDIRK), and an update is required whenever a \
                           stage has a different coefficient from the previous \
                           stage. Do not update on the NewTimeStep signal when \
                           using $(isnothing(name) ? "this tableau" : name).")

function fix_float_error(value)
    FT = typeof(value)
    FT <: AbstractFloat || return value
    atol = 100 * eps(FT)
    abs(value) > atol || return 0
    for denominator in 1:100, numerator in (-10 * denominator):(10 * denominator)
        abs(value - numerator // denominator) > atol || return numerator // denominator
    end
    return value
end

struct ARKAlgorithmCache{T, N}
    timestepper_cache::T
    newtons_method_cache::N
end

function init_cache(prob::DiffEqBase.AbstractODEProblem, alg::ARKAlgorithm; kwargs...)
    (; u0) = prob
    (; T_lim!, T_exp!, T_exp_T_lim!, T_imp!) = prob.f
    (; name, tableau, newtons_method) = alg

    FT = float_type(eltype(u0))
    isconcretetype(FT) || error("floating point type of initial state is not concrete: $FT")

    s = length(tableau.imp.b) # number of internal stages

    A_lim = vcat(tableau.lim.a, tableau.lim.b')
    A_exp = vcat(tableau.exp.a, tableau.exp.b')
    A_imp = vcat(tableau.imp.a, tableau.imp.b')
    Γ_imp = vcat(tableau.imp.Γ, tableau.imp.b')

    DΓ = diag(tableau.imp.Γ)
    DΓ_imp = vcat(Diagonal(DΓ), zeros(s)')
    LΓ_imp = Γ_imp - DΓ_imp

    z_stages = findall(iszero, DΓ) # stages without implicit solves
    nz_stages = findall(!iszero, DΓ) # stages with implicit solves

    I_z = zeros(s, s)
    I_z[z_stages, z_stages] = Matrix(I, length(z_stages), length(z_stages))

    A_imp_z = zeros(s + 1, s)
    A_imp_z[:, z_stages] = A_imp[:, z_stages]

    A_imp_nz = A_imp - A_imp_z
    DA = diag(tableau.imp.a)
    DA_imp = vcat(Diagonal(DA), zeros(s)')
    LA_imp = A_imp - DA_imp
    LA_imp_nz = A_imp_nz - DA_imp

    Γ_imp_z = zeros(s + 1, s)
    Γ_imp_z[:, z_stages] = Γ_imp[:, z_stages]

    Γ_imp_nz = Γ_imp - Γ_imp_z
    LΓ_imp_nz = Γ_imp_nz - DΓ_imp
    γ_imp_nz = Γ_imp_nz[1:s, :]

    G_imp_nz = fix_float_error.(LA_imp_nz * (inv(I_z + γ_imp_nz) - I_z))
    @assert all(iszero, G_imp_nz[:, z_stages])
    @assert all(iszero, UpperTriangular(G_imp_nz[1:s, :]))
    @assert all(value -> value == 0 || abs(value) > 100 * eps(), G_imp_nz)

    empty_matrix_rows = ntuple(_ -> SparseTuple(), s + 1)
    empty_vector = SparseTuple()

    if isnothing(T_lim!) && isnothing(T_exp_T_lim!)
        r_lim = nothing
        A_lim_rows = empty_matrix_rows
        T_lim_sparse = empty_vector
    else
        r_lim = if isnothing(tableau.lim.α)
            @warn "a canonical Shu-Osher formulation is required in order to \
                   use limiters in a way that exactly preserves monotonicity"
            nothing
        else
            # Use the Shu-Osher formulation to get each stage's SSP coefficient.
            β_lim = A_lim - tableau.lim.α * A_lim[1:s, :]
            @assert all(>=(0), β_lim)
            @assert all(iszero, UpperTriangular(β_lim[1:s, :]))
            @assert issubset(findall(iszero, tableau.lim.α), findall(iszero, β_lim))
            map(eachcol(tableau.lim.α), eachcol(β_lim)) do α_column, β_column
                nonzero_stages = findall(!iszero, β_column)
                isempty(nonzero_stages) ? FT(0) : FT(minimum(α_column[nonzero_stages] ./ β_column[nonzero_stages]))
            end
        end
        # Use the Butcher formulation for other computations involving T_lim.
        A_lim_rows = sparse_matrix_rows(FT.(A_lim))
        T_lim_sparse = SparseTuple(_ -> similar(u0), findall(!iszero, eachcol(A_lim)))
    end

    if isnothing(T_exp!) && isnothing(T_exp_T_lim!)
        A_exp_rows = empty_matrix_rows
        T_exp_sparse = empty_vector
    else
        # Use the Butcher formulation for T_exp.
        A_exp_rows = sparse_matrix_rows(FT.(A_exp))
        T_exp_sparse = SparseTuple(_ -> similar(u0), findall(!iszero, eachcol(A_exp)))
    end

    if isnothing(T_imp!)
        LA_imp_rows = G_imp_rows = empty_matrix_rows
        T_imp_sparse = ΔU_imp_nz_sparse = empty_vector
    elseif count(!iszero, LA_imp_nz) <= count(!iszero, G_imp_nz)
        # Use the Butcher formulation for T_imp if its matrix is sparser, or if
        # both formulations have the same sparsity.
        LA_imp_rows = sparse_matrix_rows(FT.(LA_imp))
        G_imp_rows = empty_matrix_rows
        T_imp_sparse = SparseTuple(_ -> similar(u0), findall(!iszero, eachcol(LA_imp)))
        ΔU_imp_nz_sparse = empty_vector
    else
        # Use the increment formulation for T_imp if its matrix is sparser.
        LA_imp_rows = sparse_matrix_rows(FT.(A_imp_z))
        G_imp_rows = sparse_matrix_rows(FT.(G_imp_nz))
        T_imp_sparse = SparseTuple(_ -> similar(u0), findall(!iszero, eachcol(A_imp_z)))
        ΔU_imp_nz_sparse = SparseTuple(_ -> similar(u0), findall(!iszero, eachcol(G_imp_nz)))
    end

    timestepper_cache = (;
        internal_and_final_stages = ntuple(identity, s + 1),
        γ = length(unique(DΓ[nz_stages])) == 1 ? FT(DΓ[nz_stages[1]]) : nothing,
        DΓ = FT.(DΓ),
        Γ_imp = FT.(Γ_imp),
        c_lim = FT.(tableau.lim.c),
        c_exp = FT.(tableau.exp.c),
        c_imp = FT.(tableau.imp.c),
        r_lim,
        A_lim_rows,
        A_exp_rows,
        LA_imp_rows,
        G_imp_rows,
        T_lim_sparse,
        T_exp_sparse,
        T_imp_sparse,
        ΔU_imp_nz_sparse,
        T_lim = dense_tuple(T_lim_sparse, s + 1, nothing),
        T_exp = dense_tuple(T_exp_sparse, s + 1, nothing),
        T_imp = dense_tuple(T_imp_sparse, s + 1, nothing),
        ΔU_imp_nz = dense_tuple(ΔU_imp_nz_sparse, s + 1, nothing),
        u_on_stage = similar(u0),
        u_plus_Δu_lim = similar(u0),
    )

    newtons_method_cache = if !iszero(DΓ) && !isnothing(T_imp!)
        (isnothing(newtons_method) && !(name isa ClimaTimeSteppers.ARKRosenbrockAlgorithmName)) && imp_error(name)
        j = has_jac(T_imp!) ? T_imp!.jac_prototype : nothing
        allocate_cache(newtons_method, u0, j)
    else
        nothing
    end

    return ARKAlgorithmCache(timestepper_cache, newtons_method_cache)
end

function step_implicit!(name,
                        newtons_method,
                        newtons_method_cache,
                        u_on_stage,
                        T_imp!,
                        p,
                        t_imp,
                        u_minus_Δu_imp_from_solve,
                        Δtγ,
                        pre_implicit_solve!,
                        post_stage!
                        )
    # Solve u′ ≈ u_minus_Δu_imp_from_solve + Δtγ * T_imp(u′, p, t_imp).
    solve_newton!(
        newtons_method,
        newtons_method_cache,
        u_on_stage,
        (residual, u′) -> begin
            T_imp!(residual, u′, p, t_imp)
            @. residual = u_minus_Δu_imp_from_solve + Δtγ * residual - u′
        end,
        (jacobian, u′) -> T_imp!.Wfact(jacobian, u′, p, Δtγ, t_imp),
        u′ -> pre_implicit_solve!(u′, p, t_imp),
        u′ -> post_stage!(u′, p, t_imp),
    )
end

function step_implicit!(name::ClimaTimeSteppers.ARKRosenbrockAlgorithmName,
                        newtons_method,
                        newtons_method_cache,
                        u_on_stage,
                        T_imp!,
                        p,
                        t_imp,
                        u_minus_Δu_imp_from_solve,
                        Δtγ,
                        pre_implicit_solve!,
                        post_stage!
                        )
end

function step_u!(integrator, cache::ARKAlgorithmCache)
    (; u, p, t, alg) = integrator
    (; T_lim!, T_exp!, T_exp_T_lim!, T_imp!) = integrator.sol.prob.f
    (; lim!, dss!, post_stage!, pre_implicit_solve!) = integrator.sol.prob.f
    (; name, newtons_method) = alg
    (; newtons_method_cache) = cache
    (;
        internal_and_final_stages,
        γ,
        DΓ,
        c_lim,
        c_exp,
        c_imp,
        r_lim,
        A_lim_rows,
        A_exp_rows,
        LA_imp_rows,
        G_imp_rows,
        T_lim_sparse,
        T_exp_sparse,
        T_imp_sparse,
        ΔU_imp_nz_sparse,
        T_lim,
        T_exp,
        T_imp,
        ΔU_imp_nz,
        u_on_stage,
        u_plus_Δu_lim,
    ) = cache.timestepper_cache

    Δt = integrator.dt
    s = length(internal_and_final_stages) - 1

    if !isnothing(newtons_method_cache)
        (; update_j) = newtons_method
        (; j) = newtons_method_cache
        if !isnothing(j) && needs_update!(update_j, NewTimeStep(t))
            isnothing(γ) && sdirk_error(name)
            T_imp!.Wfact(j, u, p, Δt * γ, t)
        end
    end

    u_on_stage .= u

    # Use unrolled_foreach instead of a regular loop to ensure type stability
    # with nonuniform tuples.
    unrolled_foreach(
        internal_and_final_stages,
        A_lim_rows,
        A_exp_rows,
        LA_imp_rows,
        G_imp_rows,
        T_lim,
        T_exp,
        T_imp,
        ΔU_imp_nz,
    ) do stage, A_lim_row, A_exp_row, LA_imp_row, G_imp_row, T_lim_on_stage, T_exp_on_stage, T_imp_on_stage, Δu_imp_nz
        Δu_lim_over_Δt = broadcasted_dot(A_lim_row, T_lim_sparse)
        Δu_exp_over_Δt = broadcasted_dot(A_exp_row, T_exp_sparse)
        Δu_imp_from_T_over_Δt = broadcasted_dot(LA_imp_row, T_imp_sparse)
        Δu_imp_from_ΔU = broadcasted_dot(G_imp_row, ΔU_imp_nz_sparse)

        # Get a lazy representation of the state on the current stage, but
        # without the contribution from the implicit solve (if there is one).
        u_minus_Δu_imp_from_solve = if isempty(A_lim_row) || !isnothing(r_lim)
            Δu_from_T_over_Δt = Base.broadcasted(+, Δu_lim_over_Δt, Δu_exp_over_Δt, Δu_imp_from_T_over_Δt)
            Base.broadcasted(+, u, Base.broadcasted(*, Δt, Δu_from_T_over_Δt), Δu_imp_from_ΔU)
        else
            # The following operations do not make sense in the context of a
            # Runge-Kutta method, so the time passed to lim and dss is nothing.
            @. u_plus_Δu_lim = u + Δt * Δu_lim_over_Δt
            lim!(u_plus_Δu_lim, p, nothing, u)
            stage < s + 1 && dss!(u_plus_Δu_lim, p, nothing)

            Δu_from_T_over_Δt = Base.broadcasted(+, Δu_exp_over_Δt, Δu_imp_from_T_over_Δt)
            Base.broadcasted(+, u_plus_Δu_lim, Base.broadcasted(*, Δt, Δu_from_T_over_Δt), Δu_imp_from_ΔU)
        end

        # If we are past the last stage, compute the final state and apply dss!
        # and post_stage!.
        if stage == s + 1
            @. u = u_minus_Δu_imp_from_solve
            dss!(u, p, t + Δt)
            post_stage!(u, p, t + Δt)
            return
        end

        Δtγ = Δt * DΓ[stage]
        t_lim = t + Δt * c_lim[stage]
        t_exp = t + Δt * c_exp[stage]
        t_imp = t + Δt * c_imp[stage]

        # Compute the state on the current stage. Apply post_stage! if it is
        # different from the previous state.
        if !isnothing(T_imp!) && !iszero(Δtγ)
            @. u_on_stage = u_minus_Δu_imp_from_solve
            # TODO: Is u_minus_Δu_imp_from_solve a good initial guess?
            # Alternatives include u and u_on_stage + Δtγ * T_imp(u_on_stage).

            pre_implicit_solve!(u_on_stage, p, t_imp)

            step_implicit!(name,
                           newtons_method,
                           newtons_method_cache,
                           u_on_stage,
                           T_imp!,
                           p,
                           t_imp,
                           u_minus_Δu_imp_from_solve,
                           Δtγ,
                           pre_implicit_solve!,
                           post_stage!)
        else
            @. u_on_stage = u_minus_Δu_imp_from_solve
            if !isempty(A_lim_row) || !isempty(A_exp_row) || !isempty(LA_imp_row) || !isempty(G_imp_row)
                post_stage!(u_on_stage, p, t_imp)
            end
        end

        # Compute the limited and/or explicit tendencies for the current stage.
        # Apply the limiter if SSP coefficients are available. Apply DSS on all
        # but the last stage.
        if !isnothing(T_lim_on_stage) || !isnothing(T_exp_on_stage)
            !isnothing(T_lim!) && T_lim!(T_lim_on_stage, u_on_stage, p, t_lim)
            !isnothing(T_exp!) && T_exp!(T_exp_on_stage, u_on_stage, p, t_exp)
            if !isnothing(T_exp_T_lim!)
                @assert t_lim == t_exp
                T_exp_T_lim!(T_exp_on_stage, T_lim_on_stage, u_on_stage, p, t_exp)
            end
            if !isnothing(r_lim)
                Δt_SSP = Δt / r_lim[stage]
                @. u_plus_Δu_lim = u_on_stage + Δt_SSP * T_lim_on_stage
                lim!(u_plus_Δu_lim, p, t_lim, u_on_stage)
                @. T_lim_on_stage = (u_plus_Δu_lim - u_on_stage) / Δt_SSP
            end
            if stage < s
                !isnothing(T_lim!) && dss!(T_lim_on_stage, p, t_lim)
                !isnothing(T_exp!) && dss!(T_exp_on_stage, p, t_exp)
                if !isnothing(T_exp_T_lim!) # TODO: Add support for fusing DSS.
                    dss!(T_lim_on_stage, p, t_lim)
                    dss!(T_exp_on_stage, p, t_exp)
                end
            end
        end

        # Compute the implicit tendency or increment for the current stage.
        if !isnothing(T_imp_on_stage) && iszero(Δtγ)
            T_imp!(T_imp_on_stage, u_on_stage, p, t_imp)
        elseif !isnothing(T_imp_on_stage)
            @. T_imp_on_stage = (u_on_stage - u_minus_Δu_imp_from_solve) / Δtγ
        elseif !isnothing(Δu_imp_nz)
            @. Δu_imp_nz = u_on_stage - u_minus_Δu_imp_from_solve + Δu_imp_from_ΔU
        end
    end
end
