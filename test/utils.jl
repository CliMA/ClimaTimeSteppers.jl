"""
    convergence_rates(problem, solution, method, dts; kwargs...)

Compute the errors rates of `method` on `problem` by comparing to `solution`
on the set of `dt` values in `dts`. Extra `kwargs` are passed to `solve`

`solution` should be a function with a method `solution(u0, p, t)`.
"""
function convergence_errors(prob, sol, method, dts; kwargs...)
  errs = map(dts) do dt
       # copy the problem so we don't mutate u0
      prob_copy = deepcopy(prob)
      u = solve(prob_copy, method; dt=dt, kwargs...)
      norm(u .- sol(prob.u0, prob.p, prob.tspan[end]))
  end
  return errs
end


"""
  convergence_order(problem, solution, method, dts; kwargs...)

Estimate the order of the rate of convergence of `method` on `problem` by comparing to
`solution` the set of `dt` values in `dts`.

`solution` should be a function with a method `solution(u0, p, t)`.
"""
function convergence_order(prob, sol, method, dts; kwargs...)
  errs = convergence_errors(prob, sol, method, dts; kwargs...)
  # find slope coefficient in log scale
  _,order_est = hcat(ones(length(dts)), log2.(dts)) \ log2.(errs)
  return order_est
end



"""
    DirectSolver

A linear solver which forms the full matrix of a linear operator and its LU factorization.
"""
struct DirectSolver end

DirectSolver(args...) = DirectSolver()

function (::DirectSolver)(x,A,b,matrix_updated; kwargs...)
  n = length(x)
  M = mapslices(y -> mul!(similar(y), A, y), Matrix{eltype(x)}(I,n,n), dims=1)
  x .= M \ b
end
