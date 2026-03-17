ClimaTimeSteppers.jl Release Notes
========================

Main
-------
- ![][badge-🔥behavioralΔ] Add keyword argument `constrain_state!`, which can be used
    to apply state constraints. Currently, these are always applied immediately before `dss!`.

v0.8.7
------
- Added a simple line search algorithm to Newtons method.

v0.8.5
------
- Removed SciMLBase` type `restriction on `step_u!`.

v0.8.2
------
- ![][badge-💥breaking] If saveat is a number, then it does not automatically expand to `tspan[1]:saveat:tspan[2]`. To fix this, update
`saveat`, which is a keyword in the integrator, to be an array. For example, if `saveat` is a scalar, replace it with
`[tspan[1]:saveat:tspan[2]..., tspan[2]]` to achieve the same behavior as before.
- IMEXAlgorithms and SSPKnoth are compatible with ITime. See ClimaUtilities for more information about ITime.

v0.7.18
-------
- ![][badge-🚀performance] Added a fused `T_exp_T_lim!`, so that we can fuse DSS calls that previously lived in `T_exp!` and `T_lim!`.

v0.7.17
-------
- Started NEWS
- ![][badge-🤖precisionΔ]![][badge-🚀performance] increments are now fused in the imex ARK method. This should yield better GPU performance, and may result in machine-precision changes.

<!--
Contributors are welcome to begin the description of changelog items with badge(s) below. Here is a brief description of when to use badges for a particular pull request / set of changes:
 - 🔥behavioralΔ - behavioral changes. For example: a new model is used, yielding more accurate results.
 - 🤖precisionΔ - machine-precision changes. For example, swapping the order of summed arguments can result in machine-precision changes.
 - 💥breaking - breaking changes. For example: removing deprecated functions/types, removing support for functionality, API changes.
 - 🚀performance - performance improvements. For example: improving type inference, reducing allocations, or code hoisting.
 - ✨feature - new feature added. For example: adding support for a cubed-sphere grid
 - 🐛bugfix - bugfix. For example: fixing incorrect logic, resulting in incorrect results, or fixing code that otherwise might give a `MethodError`.
-->

[badge-🔥behavioralΔ]: https://img.shields.io/badge/🔥behavioralΔ-orange.svg
[badge-🤖precisionΔ]: https://img.shields.io/badge/🤖precisionΔ-black.svg
[badge-💥breaking]: https://img.shields.io/badge/💥BREAKING-red.svg
[badge-🚀performance]: https://img.shields.io/badge/🚀performance-green.svg
[badge-✨feature/enhancement]: https://img.shields.io/badge/feature/enhancement-blue.svg
[badge-🐛bugfix]: https://img.shields.io/badge/🐛bugfix-purple.svg
