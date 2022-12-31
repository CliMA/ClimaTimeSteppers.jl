#####
##### 1st order
#####

# TODO: is it better to use `first_order_tableau = Union{ARS111,ARS121}`? to
#       reduce the number of methods?
const first_order_tableau = [ARS111, ARS121]

#####
##### 2nd order
#####

const second_order_tableau = [
    ARS122,
    ARS232,
    ARS222,
    IMKG232a,
    IMKG232b,
    IMKG242a,
    IMKG242b,
    IMKG252a,
    IMKG252b,
    IMKG253a,
    IMKG253b,
    IMKG254a,
    IMKG254b,
    IMKG254c,
    HOMMEM1,
]

#####
##### 3rd order
#####
const third_order_tableau = [ARS233, ARS343, ARS443, IMKG342a, IMKG343a, DBM453]

import OrdinaryDiffEq as ODE
import ClimaTimeSteppers as CTS
ODE.alg_order(alg::CTS.IMEXARKAlgorithm) = ODE.alg_order(alg.tab)

for m in first_order_tableau
    @eval ODE.alg_order(::$m) = 1
end
for m in second_order_tableau
    @eval ODE.alg_order(::$m) = 2
end
for m in third_order_tableau
    @eval ODE.alg_order(::$m) = 3
end
