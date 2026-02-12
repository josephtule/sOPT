# Oracle API and Behavior

Defined in [`include/sOPT/problem/oracle.hpp`](../../include/sOPT/problem/oracle.hpp).

## Main responsibilities

- Expose uniform `func` / `gradient` / `hessian` / `hv` operations.
- Dispatch to analytic or finite-difference derivative paths.
- Maintain evaluation counters.
- Enforce per-type eval limits.
- Provide optional LRU caching for `f`/`g`/`H`.

## Eval semantics

- `try_*` methods return `bool` success and do not throw
- Counters increment only on concrete evaluations (cache hits do not increment).

## Fallback order

- Gradient: analytic gradient, else FD gradient.
- Hessian: analytic Hessian, else FD Hessian.
- Hv: analytic Hv, else Hessian-times-vector if Hessian exists, else FD Hv.

## Cache keying

- Checks exact equality on `x` entries (`(a.array() == b.array()).all()`).
- Per-quantity caches use independent LRU replacement.
- Hessian cache may be disabled at runtime by memory guard (for large systems).
