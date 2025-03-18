var documenterSearchIndex = {"docs":
[{"location":"background/AdditiveRungeKutta/#Additive-Runge–Kutta","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"","category":"section"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"ARK methods are IMEX (Implicit-Explicit) methods based on splitting the ODE function f(u) = f_L(u) + f_R(t)  where f_L(u) = L u is a linear operator which is treated implicitly, and f_R(u) is the remainder to be treated explicitly. Typically we will be given either the pair (f_R f_L), which we will term the remainder form, or (f f_L) which we will term the full form. ","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"The value on the ith stage U^(i) is","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"U^(i) = u^n + Delta t sum_j=1^i tilde a_ij f_L(U^(j)) \n              + Delta t sum_j=1^i-1 a_ij f_R(U^(j))","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"which can be written as the solution to the linear problem:","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"W U^(i) = hat U^(i)","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"where","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"hat U^(i) = u^n + Delta t sum_j=1^i-1 tilde a_ij f_L(U^(j)) \n                                             + Delta t sum_j=1^i-1 a_ij f_R(U^(j))","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"and","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"W = (I - Delta t tilde a_ii L)","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"The next step is then defined as","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"u^n+1 = u^n + Delta t sum_i=1^N b_i  f_L(U^(i)) + f_R(U^(i)) ","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"When an iterative solver is used, an initial value bar U^(i) can be chosen by an explicit approximation","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"bar U^(i) = u^n + Delta t sum_j=1^i-1 a_ij  f_L(U^(j)) + f_R(U^(j)) \n            = hat U^(i) + Delta t sum_j=1^i-1 (a_ij - tilde a_ij)  f_L(U^(j)) ","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"By convention, tilde a_11 = 0, so that U^(1) = u^n, and for all other stages the implicit coefficients tilde a_ii are the same. ","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"If the linear operator L is time-invariant and Delta t is constantm, then if using a direct solver, W only needs to be factorized once.","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"Alternatively if an iterative solver is used used, we can write","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"bar U^(i) = u^n + Delta t sum_j=1^i-1 a_ij  f_L(U^(j)) + f_R(U^(j)) \n            = hat U^(i) + Delta t  L sum_j=1^i-1 (a_ij - tilde a_ij)  U^(j)","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"at the cost of one evaluation of f_L.","category":"page"},{"location":"background/AdditiveRungeKutta/#Reducing-evaluations-and-storage","page":"Additive Runge–Kutta","title":"Reducing evaluations and storage","text":"","category":"section"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"If the linear operator L is constant, then we are able to avoid evaluating the f_L explicitly.","category":"page"},{"location":"background/AdditiveRungeKutta/#Remainder-form","page":"Additive Runge–Kutta","title":"Remainder form","text":"","category":"section"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"If we are given f_L and f_R, we can avoid storing f_L(U^(j)) by further defining","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"Omega^(i) = sum_j=1^i-1 fractilde a_ijtilde a_ii U^(j)","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"and writing","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"hat U^(i) = u^n + Delta t tilde a_ii f_L( Omega^(i) ) + Delta t sum_j=1^i-1 a_ij f_R(U^(j))","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"This can be rewritten into an offset linear problem","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"W V^(i) = hat V^(i)","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"where","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"V^(i) = U^(i) + Omega^(i)","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"and","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"hat V^(i)\n  = hat U_(i) + (I - Delta t tilde a_ii L)  Omega^(i) \n  = u^n + Omega^(i) + Delta t sum_j=1^i-1 a_ij f_R(U^(j))","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"If using an iterative method, an initial guess is","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"bar V^(i) = bar U^(i) + Omega^(i)\n  = u^n + Delta t sum_j=1^i-1 a_ij f_R(U^(j)) + Omega^(i) + Delta t sum_j=1^i-1 a_ij f_L(U^(j))\n  = hat V^(i) + Delta t L sum_j=1^i-1 a_ij U^(j)","category":"page"},{"location":"background/AdditiveRungeKutta/#Full-form","page":"Additive Runge–Kutta","title":"Full form","text":"","category":"section"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"Similary, if we are given f and f_L, we can avoid storing f_L(U^(j)) by defining","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"Omega^(i) = sum_j=1^i-1 fractilde a_ij - a_ijtilde a_ii U^(j)","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"so that we can write","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"hat U^(i) = u^n + Delta t tilde a_ii f_L(Omega^(i)) + Delta t sum_j=1^i-1 a_ij f(U^(j))","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"As above, we can rewrite into an offset linear problem","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"W V^(i) = hat V^(i)","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"where","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"V^(i) = U^(i) + Omega^(i)","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"and","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"hat V^(i) \n  = hat U_(i) + (I - Delta t tilde a_ii L)  Omega^(i) \n  = u^n + Omega^(i) + Delta t sum_j=1^i-1 a_ij f(U^(j))","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"If using an iterative method, an initial guess is","category":"page"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"bar V^(i) = bar U^(i) + Omega^(i)\n  = u^n + Delta t sum_j=1^i-1 a_ij f(U^(j)) \n  = hat V^(i) - Omega^(i)","category":"page"},{"location":"background/AdditiveRungeKutta/#References","page":"Additive Runge–Kutta","title":"References","text":"","category":"section"},{"location":"background/AdditiveRungeKutta/","page":"Additive Runge–Kutta","title":"Additive Runge–Kutta","text":"F. X. Giraldo, J. F. Kelly, and E. M. Constantinescu (2013). Implicit-Explicit Formulations of a Three-Dimensional Nonhydrostatic Unified Model of the Atmosphere (NUMA) SIAM Journal on Scientific Computing 35(5), B1162-B1194, doi:10.1137/120876034","category":"page"},{"location":"#TimeMachine.jl","page":"TimeMachine.jl","title":"TimeMachine.jl","text":"","category":"section"},{"location":"","page":"TimeMachine.jl","title":"TimeMachine.jl","text":"TimeMachine.jl is a suite of ordinary differential equation (ODE) solvers for use as time-stepping methods in a partial differential equation (PDE) solver, such as ClimateMachine.jl. They are specifically written to support distributed and GPU computation, while minimising the memory footprint.","category":"page"},{"location":"","page":"TimeMachine.jl","title":"TimeMachine.jl","text":"TimeMachine.jl is built on top of DiffEqBase.jl, and aims to be compatible with the DifferentialEquations.jl ecosystem.","category":"page"},{"location":"algorithms/#Algorithms","page":"Algorithms","title":"Algorithms","text":"","category":"section"},{"location":"algorithms/#Low-Storage-Runge–Kutta-(LSRK)-methods","page":"Algorithms","title":"Low-Storage Runge–Kutta (LSRK) methods","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"Low-storage Runger–Kutta methods reduce the number stages that need to be stored. The methods below require only one additional storage vector.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"An IncrementingODEProblem must be used.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"LSRK54CarpenterKennedy\nLSRK144NiegemannDiehlBusch\nLSRKEulerMethod","category":"page"},{"location":"algorithms/#TimeMachine.LSRK54CarpenterKennedy","page":"Algorithms","title":"TimeMachine.LSRK54CarpenterKennedy","text":"LSRK54CarpenterKennedy()\n\nThe fourth-order, 5-stage low storage Runge–Kutta scheme from Carpenter and Kennedy (1994). The coefficients are those from Solution 3 in the paper.\n\nReferences\n\nCarpenter, M.H.; Kennedy, C.A. (1994) \"Fourth-order 2N-storage Runge–Kutta schemes\", NASA Technical Memorandum 109112. http://hdl.handle.net/2060/19940028444\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#TimeMachine.LSRK144NiegemannDiehlBusch","page":"Algorithms","title":"TimeMachine.LSRK144NiegemannDiehlBusch","text":"LSRK144NiegemannDiehlBusch()\n\nThe fourth-order, 14-stage, low-storage, Runge–Kutta scheme of Niegemann, Diehl, and Busch (2012) with optimized stability region\n\nReferences\n\nNiegemann, J.; Diehl, R.; Busch, K. (2012)  \"Efficient low-storage Runge–Kutta schemes with optimized stability regions\", Journal of Computational Physics 231(2): 364–372. doi: 10.1016/j.jcp.2011.09.003\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#TimeMachine.LSRKEulerMethod","page":"Algorithms","title":"TimeMachine.LSRKEulerMethod","text":"LSRKEulerMethod()\n\nAn implementation of explicit Euler method using LowStorageRungeKutta2N infrastructure.  This is mainly for debugging.\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#Strong-Stability-Preserving-Runge–Kutta-(SSPRK)-methods","page":"Algorithms","title":"Strong Stability-Preserving Runge–Kutta (SSPRK) methods","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"SSPRK22Heuns\nSSPRK22Ralstons\nSSPRK33ShuOsher\nSSPRK34SpiteriRuuth","category":"page"},{"location":"algorithms/#TimeMachine.SSPRK22Heuns","page":"Algorithms","title":"TimeMachine.SSPRK22Heuns","text":"SSPRK22Heuns()\n\nThe second-order, 2-stage, strong-stability-preserving, Runge–Kutta scheme of Shu and Osher (1988) (also known as Heun's method.) Exact choice of coefficients from wikipedia page for Heun's method :)\n\nReferences\n\n@article{shu1988efficient,\n  title={Efficient implementation of essentially non-oscillatory shock-capturing schemes},\n  author={Shu, Chi-Wang and Osher, Stanley},\n  journal={Journal of computational physics},\n  volume={77},\n  number={2},\n  pages={439--471},\n  year={1988},\n  publisher={Elsevier}\n}\n@article {Heun1900,\n   title = {Neue Methoden zur approximativen Integration der\n   Differentialgleichungen einer unabh\"{a}ngigen Ver\"{a}nderlichen}\n   author = {Heun, Karl},\n   journal = {Z. Math. Phys},\n   volume = {45},\n   pages = {23--38},\n   year = {1900}\n}\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#TimeMachine.SSPRK22Ralstons","page":"Algorithms","title":"TimeMachine.SSPRK22Ralstons","text":"SSPRK22Ralstons()\n\nThe second-order, 2-stage, strong-stability-preserving, Runge–Kutta scheme of Shu and Osher (1988) (also known as Ralstons's method.) Exact choice of coefficients from wikipedia page for Heun's method :)\n\nReferences\n\n@article{shu1988efficient,\n  title={Efficient implementation of essentially non-oscillatory shock-capturing schemes},\n  author={Shu, Chi-Wang and Osher, Stanley},\n  journal={Journal of computational physics},\n  volume={77},\n  number={2},\n  pages={439--471},\n  year={1988},\n  publisher={Elsevier}\n}\n@article{ralston1962runge,\n  title={Runge-Kutta methods with minimum error bounds},\n  author={Ralston, Anthony},\n  journal={Mathematics of computation},\n  volume={16},\n  number={80},\n  pages={431--437},\n  year={1962},\n  doi={10.1090/S0025-5718-1962-0150954-0}\n}\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#TimeMachine.SSPRK33ShuOsher","page":"Algorithms","title":"TimeMachine.SSPRK33ShuOsher","text":"SSPRK33ShuOsher()\n\nThe third-order, 3-stage, strong-stability-preserving, Runge–Kutta scheme of Shu and Osher (1988)\n\nReferences\n\n@article{shu1988efficient,\n  title={Efficient implementation of essentially non-oscillatory shock-capturing schemes},\n  author={Shu, Chi-Wang and Osher, Stanley},\n  journal={Journal of computational physics},\n  volume={77},\n  number={2},\n  pages={439--471},\n  year={1988},\n  publisher={Elsevier}\n}\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#TimeMachine.SSPRK34SpiteriRuuth","page":"Algorithms","title":"TimeMachine.SSPRK34SpiteriRuuth","text":"SSPRK34SpiteriRuuth()\n\nThe third-order, 4-stage, strong-stability-preserving, Runge–Kutta scheme of Spiteri and Ruuth (1988)\n\nReferences\n\n@article{spiteri2002new,\n  title={A new class of optimal high-order strong-stability-preserving time discretization methods},\n  author={Spiteri, Raymond J and Ruuth, Steven J},\n  journal={SIAM Journal on Numerical Analysis},\n  volume={40},\n  number={2},\n  pages={469--491},\n  year={2002},\n  publisher={SIAM}\n}\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#Additive-Runge–Kutta-(ARK)-methods","page":"Algorithms","title":"Additive Runge–Kutta (ARK) methods","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"ARK methods are IMEX (Implicit-Explicit) methods based on splitting the ODE function into a linear and remainder components:","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"fracdudt = Lu + f_R(ut)","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"where the linear part is solved implicitly. All the algorithms below take a linsolve argument to specify the linear solver to be used. See the linsolve specification of DifferentialEquations.jl.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"Currently ARK methods require a SplitODEProblem.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"ARK1ForwardBackwardEuler\nARK2ImplicitExplicitMidpoint\nARK2GiraldoKellyConstantinescu\nARK437L2SA1KennedyCarpenter\nARK548L2SA2KennedyCarpenter","category":"page"},{"location":"algorithms/#TimeMachine.ARK1ForwardBackwardEuler","page":"Algorithms","title":"TimeMachine.ARK1ForwardBackwardEuler","text":"ARK1ForwardBackwardEuler(linsolve)\n\nA first-order-accurate two-stage additive Runge–Kutta scheme, combining a forward Euler explicit step with a backward Euler implicit correction.\n\nReferences\n\nAscher, U.M.; Ruuth, S.J. and Spiteri, R.S. (1997) \"Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations\", Applied Numerical Mathematics, 25(2-3): 151–167. doi: 10.1016/S0168-9274(97)00056-1\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#TimeMachine.ARK2ImplicitExplicitMidpoint","page":"Algorithms","title":"TimeMachine.ARK2ImplicitExplicitMidpoint","text":"ARK2ImplicitExplicitMidpoint(linsolve)\n\nA second-order, two-stage additive Runge–Kutta scheme, combining the implicit and explicit midpoint methods.\n\nReferences\n\nAscher, U.M.; Ruuth, S.J. and Spiteri, R.S. (1997) \"Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations\", Applied Numerical Mathematics, 25(2-3): 151–167. doi: 10.1016/S0168-9274(97)00056-1\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#TimeMachine.ARK2GiraldoKellyConstantinescu","page":"Algorithms","title":"TimeMachine.ARK2GiraldoKellyConstantinescu","text":"ARK2GiraldoKellyConstantinescu(linsolve; paperversion=false)\n\nThe second-order, 3-stage additive Runge–Kutta scheme of Giraldo, Kelly and Constantinescu (2013).\n\nIf the keyword paperversion=true is used, the coefficients from the paper are used. Otherwise it uses coefficients that make the scheme (much) more stable but less accurate\n\nReferences\n\nGiraldo, F.X.; Kelly, J.F. and Constantinescu, E.M. (2013) \"Implicit-explicit formulations of a three-dimensional nonhydrostatic unified model of the atmosphere (NUMA)\", SIAM Journal on Scientific Computing, 35(5): B1162–B1194 doi: 10.1137/120876034\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#TimeMachine.ARK437L2SA1KennedyCarpenter","page":"Algorithms","title":"TimeMachine.ARK437L2SA1KennedyCarpenter","text":"ARK437L2SA1KennedyCarpenter(linsolve)\n\nThe fourth-order, 7-stage additive Runge–Kutta scheme ARK4(3)7L[2]SA₁ of  Kennedy and Carpenter (2019).\n\nReferences\n\nKennedy, C.A. and Carpenter, M.H. (2019) \"Higher-order additive Runge–Kutta schemes for ordinary differential equations\" Applied Numerical Mathematics. 136: 183–205. doi: 10.1016/j.apnum.2018.10.007\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#TimeMachine.ARK548L2SA2KennedyCarpenter","page":"Algorithms","title":"TimeMachine.ARK548L2SA2KennedyCarpenter","text":"ARK548L2SA2KennedyCarpenter(linsolve)\n\nThe fifth-order, 8-stage additive Runge–Kutta scheme ARK5(4)8L[2]SA₂ of  Kennedy and Carpenter (2019).\n\nReferences\n\nKennedy, C.A. and Carpenter, M.H. (2019) \"Higher-order additive Runge–Kutta schemes for ordinary differential equations\" Applied Numerical Mathematics. 136: 183–205. doi: 10.1016/j.apnum.2018.10.007\n\n\n\n\n\n","category":"type"}]
}
