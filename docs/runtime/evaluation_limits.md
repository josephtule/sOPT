# Evaluation Limits

Evaluation budgets are configured by:

- `opt.limits.max_f_evals`
- `opt.limits.max_g_evals`
- `opt.limits.max_h_evals`

Semantics:

- $v \ge 0$ means limit enabled
- $v < 0$ means unlimited

## Oracle Budget Checks

Before evaluating:

$$
\begin{aligned}
\mathtt{can\_eval\_f}:&\ f_\mathtt{evals} < \mathtt{max\_f\_evals}\quad(\text{if enabled}), \\
\mathtt{can\_eval\_g}:&\ g_\mathtt{evals} < \mathtt{max\_g\_evals}\quad(\text{if enabled}), \\
\mathtt{can\_eval\_h}:&\ h_\mathtt{evals} < \mathtt{max\_h\_evals}\quad(\text{if enabled}).
\end{aligned}
$$

`try_*` returns `false` if budget is exhausted.

## Mapping to Solver Status

In common helper wrappers:

- failed `try_func` with f-limit reached -> `max_evals`
- failed `try_gradient` with g-limit reached -> `max_evals`
- failed `try_hessian` with h-limit reached -> `max_evals`

Otherwise failures map to `eval_failed`.

## See related docs:

- [solver_flow_and_status.md](solver_flow_and_status.md)
- [oracle_cache.md](oracle_cache.md)
