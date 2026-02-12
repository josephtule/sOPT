# Benchmark Objectives and Harness

Benchmark objective functions from: @LVtestproblems, most of the gradients are taken by hand or left empty.

## Objective families

- [`quadratic_spd.hpp`](../../include/sOPT/bench/quadratic_spd.hpp): SPD quadratic with generated spectrum/conditioning
- [`lasso_diag.hpp`](../../include/sOPT/bench/lasso_diag.hpp): diagonal least-squares smooth term for L1-composite (`lasso`-style) benchmarks
- [`sparse_logistic.hpp`](../../include/sOPT/bench/sparse_logistic.hpp): sparse-feature logistic smooth term for L1-composite benchmarks
- [`rosenbrock.hpp`](../../include/sOPT/bench/rosenbrock.hpp): chained Rosenbrock in $\R^n$
- [`powell_singular.hpp`](../../include/sOPT/bench/powell_singular.hpp): chained Powell singular objective
- [`wood.hpp`](../../include/sOPT/bench/wood.hpp): chained Wood objective in even dimensions
- [`cragg_levy.hpp`](../../include/sOPT/bench/cragg_levy.hpp): chained Cragg-Levy objective
- [`broyden.hpp`](../../include/sOPT/bench/broyden.hpp): generalized Broyden variants
- [`nazareth.hpp`](../../include/sOPT/bench/nazareth.hpp): Nazareth variants and TointTrig objective
- [`augmented_lagrangian.hpp`](../../include/sOPT/bench/augmented_lagrangian.hpp): synthetic augmented-Lagrangian objective

