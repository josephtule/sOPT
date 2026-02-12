# Repository Map

Top-level layout:

- [`include/sOPT/`](../include/sOPT): header-only library source
- [`examples/`](../examples): runnable example binaries
- [`tests/`](../tests): ctest test binaries
- [`python/`](../python): symbolic derivation helper scripts (SymPy)
- [`docs/`](.): maintained documentation
- [`external/eigen/`](../external/eigen): vendored Eigen (third-party)

## `include/sOPT/` layout

- `core/`: shared types, options, callbacks, result/status/trace.
- `problem/`: objective traits and Oracle.
- `finite_diff/`: gradient/Hessian/Hv finite-difference routines.
- `step_size/`: fixed and line-search step strategies.
- `algorithms/`: solver algorithm implementations.
- `bench/`: benchmark objective families 
- `sOPT.hpp`: umbrella include.

## Design summary

- Build system exports `sOPT` as an `INTERFACE` target.
- Solvers are templated on objective and step strategy.
- `Oracle<Obj>` facilitates function and derivative calls, counters, limits, and cache.
- `validate_options(opt)` ensures values in options are within bounds.
- Shared solver lifecycle/status logic is in `algorithms/detail/solver_common.hpp`.
