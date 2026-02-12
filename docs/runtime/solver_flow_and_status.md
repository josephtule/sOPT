# Solver Common Flow and Status Precedence

This reflects `include/sOPT/algorithms/detail/solver_common.hpp`.

## Initialization (`init_common`)

At $x_0$:

0. Validate options (`validate_options(opt)`) if `opt.validate_options = true`.
1. Evaluate $f_0 = f(x_0)$.
2. Evaluate $g_0 = \nabla f(x_0)$.
3. Push initial trace sample with `step_norm=0`, `alpha=0`.
4. Call iteration callback at `iter=0`.

Immediate exits:

- invalid options -> `invalid_input` (before any oracle evaluations)
- size mismatch -> `invalid_input`
- eval failure -> `eval_failed` or `max_evals` (mapped by budget state)
- non-finite $f_0$ or $\norm{g_0}$ -> `nan_detected`
- stop callback true -> `user_terminated`

## Per-Iteration General Order

For each iteration:

1. `pre_step_checks(f_k, grad_norm_k)`.
2. Build direction $p_k$ (solver-specific).
3. Run step strategy via `run_step(...)`.
4. If accepted: accept step + refresh gradient + trace + callbacks.
5. Check post-accept termination (step/objective-change criteria).

## Pre-Step Checks and Status

`pre_step_checks` order:

1. non-finite $f$ or grad norm -> `nan_detected`
2. $\norm{g_k} \le \max(\tau_g,\ \tau_{g,\mathrm{rel}}\cdot\max(1,\norm{g_0}))$
   -> `converged_grad`

with $\tau_g$ = `opt.term.grad_tol` and
$\tau_{g,\mathrm{rel}}$ = `opt.term.grad_tol_rel`.

## Step Attempt Status

Step attempt maps to:

- `accepted` -> continue with iteration.
- `eval_failed` -> `eval_failed` unless f/g budget reached, then `max_evals`.
- `line_search_failed` -> `line_search_failed` unless f/g budget reached, then `max_evals`.

Note: step-status mapping checks only function/gradient budgets (not Hessian).

## Post-Accept Checks and Status

After accepted step:

$$
\norm{x_{k+1}-x_k} \le \max(\tau_s,\ \tau_{s,\mathrm{rel}} \cdot \max(1,\norm{x_0}))
\Rightarrow \mathtt{converged\_step}.
$$

with $\tau_s$ = `opt.term.step_tol` and
$\tau_{s,\mathrm{rel}}$ = `opt.term.step_tol_rel`.

Gradient is always refreshed at the accepted iterate before this check.

If step convergence does not trigger and `f_tol > 0`, objective-change
termination is checked:

$$
|f_k-f_{k-1}| \le \tau_f\cdot\max(1,|f_{k-1}|)\Rightarrow\mathtt{success}.
$$

with $\tau_f$ = `opt.term.f_tol`.

## Finalization

Before exiting:

- Store final $f$, final $\norm{g}$.
- Sync eval counters from oracle.
- If no terminal status was set during loop, status becomes `max_iters`.

Related docs:

- [evaluation_limits.md](evaluation_limits.md)
