# EKD


function.. update_inner!(innerinteg, outercache::ETDCache,
  f_slow, u, p, t, dt, i)


  f_offset = innerinteg.prob.f
  tab = outercache.tableau
  U = outercache.U
  N = nstages(outercache)

  f_slow(f_offset.x[i], i == 1 ? u : U, p, t + tab.c[i]*dt)

  # τ ∈ [0,c[i+1]]
  #

  for j = 1:i-1
    γ[j] = Polynomial()

  βrow = ntuple(k -> ntuple(j -> βs[k][iStage+1, j], iStage), Nβ)

  if i < N
    U .= u
    innerinteg.u = U
  else
    innerinteg.u = u
  end

  innerinteg.t = t
  innerinteg.tstop = i == N ? t+dt : t+tab.c[i+1]*dt
end


# ETDRK4-B from https://doi.org/10.1016/j.jcp.2004.08.006
# Stage 1 (0)

# Stage 2
#γ_21(t) * v_1
(
  Polynomial(1//2),
)

# Stage 3
#γ_31(t) * v_1 + γ_32(t) * v_2

(
  Polynomial(RT(1 // 2), RT(-1)),
  Polynomial(0,1),
 )

 # Stage 4
 (
   Polynomial(1,-2, 0),
   Polynomial(0, 0, 0),
   Polynomial(0, 2, 0)
 )

 # Stage 5 (final)
 (
  Polynomial(1,-3, 4),
  Polymomial(0, 2,-4),
  Polymomial(0, 2,-4),
  Polynomial(0,-1, 4)
 )


 Polynomial(a0,a1,a2) => Polynomial(
   a0 / (factorial(0) * c[i]),
   a1 / (factorial(1) * c[i]),
   a2 / (factorial(2) * c[i]),
 )

 a0 + a1*ξ + a2*ξ^2

 ξ = t/c[i]

 for i = 2:Nstages
  for j = 1:i-1
      kFac = 1
      for k = 1:nPhi
          kFac = kFac * max(k - 1, 1)
          βs[k][i, j] = β[k][i, j] / (kFac * c[i])
          β[k][i, j] /= c[i]
      end
  end
end




 c = RT[0 1//2 1//2 1 1]

β0 = [
  RT(0) RT(0) RT(0) RT(0) RT(0)
  RT(1 // 2) RT(0) RT(0) RT(0) RT(0)
  RT(1 // 2) RT(0) RT(0) RT(0) RT(0)
  RT(1) RT(0) RT(1) RT(0) RT(0)
  RT(1) RT(0) RT(0) RT(0) RT(0)
]
βs0 = zeros(RT, 5, 5)
β1 = [
  RT(0) RT(0) RT(0) RT(0) RT(0)
  RT(0) RT(0) RT(0) RT(0) RT(0)
  RT(-1) RT(1) RT(0) RT(0) RT(0)
  RT(-2) RT(0) RT(2) RT(0) RT(0)
  RT(-3) RT(2) RT(2) RT(-1) RT(0)
]
βs1 = zeros(RT, 5, 5)
β2 = [
  RT(0) RT(0) RT(0) RT(0) RT(0)
  RT(0) RT(0) RT(0) RT(0) RT(0)
  RT(0) RT(0) RT(0) RT(0) RT(0)
  RT(0) RT(0) RT(0) RT(0) RT(0)
  RT(4) RT(-4) RT(-4) RT(4) RT(0)
]