export MIS2, MIS3C, MIS4, MIS4a, TVDMISA, TVDMISB

"""
    MultirateInfinitesimalStep

Multirate Infinitesimal Step (MIS) methods of [WKG2009](@cite) and [KW2014](@cite).

The available implementations are:

 - [`MIS2`](@ref)
 - [`MIS3C`](@ref)
 - [`MIS4`](@ref)
 - [`MIS4a`](@ref)
 - [`TVDMISA`](@ref)
 - [`TVDMISB`](@ref)
"""
abstract type MultirateInfinitesimalStep <: DistributedODEAlgorithm end


struct MultirateInfinitesimalStepTableau{Nstages, Nstages², RT}
  α::SArray{NTuple{2, Nstages}, RT, 2, Nstages²}
  β::SArray{NTuple{2, Nstages}, RT, 2, Nstages²}
  γ::SArray{NTuple{2, Nstages}, RT, 2, Nstages²}
  d::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
  c::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
  c̃::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
end
function MultirateInfinitesimalStepTableau(α,β,γ)
  d = SVector(sum(β, dims = 2)) # KW2014 (2)
  # c = (I - α - γ) \ d
  c = similar(d)
  for i in eachindex(c)
      c[i] = d[i]
      for j = 1:i-1
        c[i] += (α[i, j] + γ[i, j]) * c[j]
      end
  end
  c = SArray(c)
  c̃ = α * c
  MultirateInfinitesimalStepTableau(α,β,γ,d,c,c̃)
end


struct MultirateInfinitesimalStepCache{Nstages, A, T<:MultirateInfinitesimalStepTableau}
  "difference between stage and initial value ``U^{(i)} - u``"
  ΔU::NTuple{Nstages,A}
  "evaluated slow part of each stage ``f_slow(U^{(i)})``"
  F::NTuple{Nstages,A}
  tableau::T
end

nstages(::MultirateInfinitesimalStepCache{Nstages}) where {Nstages} = Nstages

function cache(
  prob::DiffEqBase.AbstractODEProblem{uType, tType, true},
  alg::MultirateInfinitesimalStep; kwargs...) where {uType,tType}

  tab = tableau(alg,eltype(prob.u0))
  N = length(tab.d)
  ΔU = ntuple(n -> similar(prob.u0), N)
  # at time i, contains
  #   ΔU[j] = U[j] - u    j < i
  #   ΔU[i] = U[i]
  #   ΔU[N] = offset vec
  F = ntuple(n -> similar(prob.u0), N)
  return MultirateInfinitesimalStepCache(ΔU, F, tab)
end


function init_inner(prob, outercache::MultirateInfinitesimalStepCache, dt)
  OffsetODEFunction(prob.f.f1, zero(dt), one(dt),
    (Polynomial(one(dt)),),
    (outercache.ΔU[end],),
    )
end

function update_inner!(innerinteg, outercache::MultirateInfinitesimalStepCache,
  f_slow, u, p, t, dt, i)

  f_offset = innerinteg.prob.f
  tab = outercache.tableau
  N = nstages(outercache)

  F = outercache.F
  ΔU = outercache.ΔU

  # F[i] = f_slow(U[i-1], p, t + c[i-1]*dt)
  u0 = i == 1 ? u : ΔU[i-1]
  t0 = i == 1 ? t : t + tab.c[i-1]*dt
  f_slow(F[i], u0, p, t0)

  # the (i+1)th stage of the paper
  innerinteg.u = i == N ? u : ΔU[i]

  groupsize = 256
  event = Event(array_device(u))
  event = mis_update!(array_device(u), groupsize)(
    u,
    ΔU,
    F,
    innerinteg.u,
    f_offset.x[1],
    tab,
    i,
    N,
    dt;
    ndrange = length(u),
    dependencies = (event,),
  )
  wait(array_device(u), event)

  # KW2014 (9)
  # evaluate f_fast(z(τ), p, t + c̃[i]*dt + (c[i]-c̃[i])/d[i] * τ)
  f_offset.α = t + tab.c̃[i] * dt
  f_offset.β = (tab.c[i] - tab.c̃[i]) / tab.d[i]

  innerinteg.t = zero(t)
  innerinteg.tstop = tab.d[i] * dt
end

@kernel function mis_update!(
  u,
  ΔU,
  F,
  innerinteg_u,
  f_offset_x,
  tab,
  i,
  N,
  dt,
)
  e = @index(Global, Linear)
  @inbounds begin
    if i > 1
      ΔU[i-1][e] -= u[e]
    end

    # KW2014 (1a)
    if i < N
      innerinteg_u[e] = u[e]
    end
    for j = 1:i-1
      innerinteg_u[e] += tab.α[i,j] * ΔU[j][e]
    end

    # KW2014 (1b) / (9)
    f_offset_x[e] = tab.β[i,i]/tab.d[i] .* F[i][e]
    for j = 1:i-1
      f_offset_x[e] += (tab.γ[i,j]/(tab.d[i]*dt)) * ΔU[j][e] + tab.β[i,j]/tab.d[i]  * F[j][e]
    end
  end
end


"""
    MIS2()

The MIS2 Multirate Infinitesimal Step (MIS) method of [KW2014](@cite).
"""
struct MIS2 <: MultirateInfinitesimalStep
end
function tableau(mis::MIS2, RT)
  α = @SArray RT[
    0 0 0
    0.536946566710 0 0
    0.480892968551 0.500561163566 0
  ]
  β = @SArray RT[
    0.126848494553 0 0
    -0.784838278826 1.37442675268 0
    -0.0456727081749 -0.00875082271190 0.524775788629
  ]
  γ = @SArray RT[
    0 0 0
    0.652465126004 0 0
    -0.0732769849457 0.144902430420 0
  ]
  MultirateInfinitesimalStepTableau(α,β,γ)
end

"""
    MIS3C()

The MIS3C Multirate Infinitesimal Step (MIS) method of [KW2014](@cite).
"""
struct MIS3C <: MultirateInfinitesimalStep
end
function tableau(mis::MIS3C, RT)
  α = @SArray RT[
    0 0 0
    0.589557277145 0 0
    0.544036601551 0.565511042564 0
  ]
  β = @SArray RT[
    0.397525189225 0 0
    -0.227036463644 0.624528794618 0
    -0.00295238076840 -0.270971764284 0.671323159437
  ]
  γ = @SArray RT[
    0 0 0
    0.142798786398 0 0
    -0.0428918957402 0.0202720980282 0
  ]
  MultirateInfinitesimalStepTableau(α,β,γ)
end

"""
    MIS4()

The MIS4 Multirate Infinitesimal Step (MIS) method of [KW2014](@cite).
"""
struct MIS4 <: MultirateInfinitesimalStep
end
function tableau(mis::MIS4, RT)
  α = @SArray RT[
    0 0 0 0
    0.914092810304 0 0 0
    1.14274417397 -0.295211246188 0 0
    0.112965282231 0.337369411296 0.503747183119 0
  ]
  β = @SArray RT[
    0.136296478423 0 0 0
    0.280462398979 -0.0160351333596 0 0
    0.904713355208 -1.04011183154 0.652337563489 0
    0.0671969845546 -0.365621862610 -0.154861470835 0.970362444469
  ]
  γ = @SArray RT[
    0 0 0 0
    0.678951983291 0 0 0
    -1.38974164070 0.503864576302 0 0
    -0.375328608282 0.320925021109 -0.158259688945 0
  ]
  MultirateInfinitesimalStepTableau(α,β,γ)
end

"""
    MIS4a()

The MIS4a Multirate Infinitesimal Step (MIS) method of [KW2014](@cite).
"""
struct MIS4a <: MultirateInfinitesimalStep
end
function tableau(mis::MIS4a, RT)
  α = @SArray RT[
    0 0 0 0
    0.52349249922385610 0 0 0
    1.1683374366893629 -0.75762080241712637 0 0
    -0.036477233846797109 0.56936148730740477 0.47746263002599681 0
  ]
  # β₅₁ in the paper is incorrect
  # the correct value (β[4,1]) is used below (from authors)
  β = @SArray RT[
    0.38758444641450318 0 0 0
    -0.025318448354142823 0.38668943087310403 0 0
    0.20899983523553325 -0.45856648476371231 0.43423187573425748 0
    -0.10048822195663100 -0.46186171956333327 0.83045062122462809 0.27014914900250392
  ]
  γ = @SArray RT[
    0 0 0 0
    0.13145089796226542 0 0 0
    -0.36855857648747881 0.33159232636600550 0 0
    -0.065767130537473045 0.040591093109036858 0.064902111640806712 0
  ]
  MultirateInfinitesimalStepTableau(α,β,γ)
end

"""
    TVDMISA()

The TVDMISA Total Variation Diminishing (TVD) Multirate Infinitesimal Step (MIS)
method of [KW2014](@cite).
"""
struct TVDMISA <: MultirateInfinitesimalStep
end
function tableau(mis::TVDMISA, RT)
  α = @SArray RT[
    0 0 0
    0.1946360605647457 0 0
    0.3971200136786614 0.2609434606211801 0
  ]
  β = @SArray RT[
    2//3 0 0
    -0.28247174703488398 4//9 0
    -0.31198081960042401 0.18082737579913699 9//16
  ]
  γ = @SArray RT[
    0 0 0
    0.5624048933209129 0 0
    0.4408467475713277 -0.2459300561692391 0
  ]
  MultirateInfinitesimalStepTableau(α,β,γ)
end

"""
    TVDMISB()

The TVDMISB Total Variation Diminishing (TVD) Multirate Infinitesimal Step (MIS)
method of [KW2014](@cite).
"""
struct TVDMISB <: MultirateInfinitesimalStep
end
function tableau(mis::TVDMISB, RT)
  α = @SArray RT[
    0 0 0
    0.42668232863311001 0 0
    0.26570779016173801 0.41489966891866698 0
  ]
  β = @SArray RT[
    2//3 0 0
    -0.25492859100078202 4//9 0
    -0.26452517179288798 0.11424084424766399 9//16
  ]
  γ = @SArray RT[
    0 0 0
    0.28904389120139701 0 0
    0.45113560071334202 -0.25006656847591002 0
  ]
  MultirateInfinitesimalStepTableau(α,β,γ)
end
