module ClimaTimeSteppersClimaCoreExt

import ClimaTimeSteppers as CTS
import ClimaCore.Fields as Fields

# With ClimaCore's KrylovExt, Krylov.ktypeof(::FieldVector) is the state's flat
# device vector type (e.g. CuVector for CuArray-backed states), so KrylovMethod
# keeps its workspace on the GPU. This adapter supplies the block-wise copies
# between the flat workspace vectors and the FieldVector layout; see
# CTS.KrylovVectorAdapter.
CTS.krylov_adapter(x_prototype::Fields.FieldVector, ::Type{S}) where {S} =
    CTS.KrylovVectorAdapter(
        zero(x_prototype),
        zero(x_prototype),
        S(undef, length(x_prototype)),
        Fields.fieldvector2array!,
        Fields.array2fieldvector!,
    )

end
