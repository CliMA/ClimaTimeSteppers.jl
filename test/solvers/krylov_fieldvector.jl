#=
julia --project=test
using Revise; include("test/solvers/krylov_fieldvector.jl")

Newton-Krylov solves with a ClimaCore FieldVector state. Exercises the flat
Krylov workspace (Krylov.ktypeof from ClimaCore's KrylovExt) and the
KrylovVectorAdapter block copies from ClimaTimeSteppersClimaCoreExt, including
the FlatPreconditioner path.
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import ClimaComms
import Krylov
using ClimaCore: Geometry, Domains, Meshes, Spaces, Fields

FT = Float64
domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0),
    Geometry.ZPoint{FT}(1);
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain, nelems = 10)
space = Spaces.CenterFiniteDifferenceSpace(ClimaComms.device(), mesh)

# Solve f(x) = x^2 - c = 0 elementwise, with exact solution x = sqrt(c).
c = Fields.FieldVector(a = 4 .* ones(space), b = 9 .* ones(space))
x_exact = Fields.FieldVector(a = 2 .* ones(space), b = 3 .* ones(space))
x_init = Fields.FieldVector(a = ones(space), b = ones(space))
f!(f, x) = @. f = x^2 - c

# M approximates j = diag(2x); ldiv!(y, M, b) computes y = b ./ (2x).
struct DiagonalPreconditioner{X}
    x::X
end
LinearAlgebra.ldiv!(y, M::DiagonalPreconditioner, b) = @. y = b / (2 * M.x)

@testset "Newton-Krylov with FieldVector state" begin
    # The Krylov workspace is flat (CuVector for CuArray-backed states), not a
    # FieldVector, so the solve avoids scalar indexing on GPUs.
    @test Krylov.ktypeof(x_init) == ClimaComms.array_type(x_init){FT, 1}

    step_sizes = (
        ForwardDiffStepSize1(),
        ForwardDiffStepSize2(),
        ForwardDiffStepSize3(),
    )
    preconditioners = (nothing, DiagonalPreconditioner(x_init))
    for step_size in step_sizes, preconditioner in preconditioners
        alg = NewtonsMethod(;
            max_iters = 20,
            krylov_method = KrylovMethod(;
                jacobian_free_jvp = ForwardDiffJVP(; default_step = step_size),
                forcing_term = ConstantForcing(FT(1e-6)),
                preconditioner,
            ),
        )
        x = copy(x_init)
        cache = CTS.allocate_cache(alg, x, nothing)
        @test cache.krylov_method_cache.adapter isa CTS.KrylovVectorAdapter
        CTS.solve_newton!(alg, cache, x, f!, nothing)
        @test norm(x .- x_exact) / norm(x_exact) < 1e-6
    end
end
