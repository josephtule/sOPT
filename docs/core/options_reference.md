# Options Reference

Defined in [`include/sOPT/core/options.hpp`](../../include/sOPT/core/options.hpp).
Validation utilities are in
[`include/sOPT/core/options_validation.hpp`](../../include/sOPT/core/options_validation.hpp)
via `validate_options(const Options&)`.

## `TerminationOptions` (`opt.term`)

- `max_iters`: hard solver iteration cap.
- `grad_tol`: absolute gradient threshold.
- `grad_tol_rel`: relative gradient threshold multiplier.
- effective gradient threshold:
    - $\norm{g_k} \le \max(\mathtt{grad\_tol},\ \mathtt{grad\_tol\_rel}\cdot \max(1,\norm{g_0}))$
- `step_tol`: absolute step thresholdl
- `step_tol_rel`: relative step threshold multiplier.
- effective step threshold:
    - $\norm{\Delta x_k} \le \max(\mathtt{step\_tol},\ \mathtt{step\_tol\_rel}\cdot \max(1,\norm{x_0}))$
- `f_tol`: relative objective-change threshold (`0` disables)
    - criterion: $|f_k-f_{k-1}| \le \mathtt{f\_tol}\cdot\max(1,|f_{k-1}|)$

Validation:

- all tolerances must be finite and nonnegative
- `max_iters > 0`

## `DiagnosticsOptions` (`opt.diag`)

- `enabled`: enable diagnostic recording.
- `record_directional_derivative`: record $g^T p$.
- `record_qn_curvature`: record $y^T s$ and cosine alignment $\frac{y^T s}{\norm{y}\norm{s}}$.
- `record_hessian_diag_bounds`: record Hessian diagonal min/max.
- `cond_mode`: `off` | `diagonal_proxy` | `power_iteration` determines the condition of the Hessian matrix (or approximate) via a diagonal proxy approximation (cheap) or eigenvalue power iteration (expensive).
- `cond_power_iters`: power-iteration count for condition estimate.
- `cond_eps`: numerical guard epsilon for condition estimation.

Validation:

- `cond_power_iters >= 0`
- `cond_eps > 0`

## Trace settings on `Options`

- `trace_level`: `off` | `basic` | `full`.
- `trace_reserve`: reserve capacity for trace vectors (`0` -> `max_iters + 1`).

## `EvalLimitOptions` (`opt.limits`)

- `max_f_evals`, `max_g_evals`, `max_h_evals`
- semantics: $v \ge 0$ means enabled limit, $v < 0$ means unlimited.

## `CacheOptions` (`opt.cache`)

- `enabled`: master cache switch.
- `f_slots`, `g_slots`, `h_slots`: per-quantity LRU (least recently used) cache sizes.
- `enforce_max_bytes`: enable Hessian-cache memory guard.
- `max_bytes`: cap used by Hessian cache guard.

Validation:

- `f_slots, g_slots, h_slots >= 0`

## `FDOptions` (`opt.fd`)

- `fallback_grad`: `fd_forward` | `fd_backward` | `fd_central`.
- `fallback_hess`: `fd_forward` | `fd_backward` | `fd_central`.
- `fallback_hv`: `fd_forward` | `fd_backward` | `fd_central`.
- Append with `_2` for 2nd (and 4th) order finite differences.
- `eps`: base FD step for gradient/Hessian fallback.
- `hv_eps`: base FD step for Hv fallback.

## `LineSearchOptions` (`opt.ls`)

- `try_full_step`: wrap strategy with `TryFull`.
- `alpha_fixed`: fixed step size for `FixedStep`.
- `alpha0`: initial trial step.
- `alpha_max`: expansion cap in Wolfe-like searches.
- `rho`: backtracking contraction factor.
- `c1`: Armijo/Wolfe sufficient decrease constant.
- `c2`: Wolfe curvature constant.
- `max_iters`: max iterations for the step strategy algorithm.

Validation:

- `alpha_fixed > 0`
- `alpha0 > 0`
- `alpha_max >= alpha0`
- `0 < rho < 1`
- `0 < c1 < c2 < 1`
- `max_iters > 0`

## `NewtonOptions` (`opt.newton`)

- `damping0`: first damping value when Hessian is not SPD.
- `damping_scale`: multiplicative damping increase per retry.
- `damping_max_tries`: number of damped LLT attempts before LU fallback.

Validation:

- `damping0 > 0`
- `damping_scale > 0`
- `damping_max_tries >= 0`

## `LBFGSOptions` (`opt.lbfgs`)

- `memory`: number of $(s, y)$ pairs stored.
- `h0_auto_scale`: enable initial inverse-Hessian scaling.
- `h0_scale_min`, `h0_scale_max`: clamp range for auto scale factor.

Validation:

- `memory >= 0`
- `h0_scale_min > 0`
- `h0_scale_max > 0`
- `h0_scale_min <= h0_scale_max`
