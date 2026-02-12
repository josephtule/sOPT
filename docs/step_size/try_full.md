# Try-Full Wrapper

`TryFull<InnerStep>` is a meta-strategy: first try $\alpha=1$, then fallback to
inner strategy if needed.

## Full-step test

Let $\vecb{x},\vecb{p},\vecb{g}_0\in\R^n$.

Only if direction is descent:

$$
\vecb{g}_0^\top \vecb{p}<0.
$$

Trial:

$$
\alpha=1,
\qquad
\vecb{x}_{\text{trial}}=\vecb{x}+\vecb{p}.
$$

Accept if Armijo-style check holds:

$$
f(\vecb{x}_{\text{trial}}) \le f_0 + c_1\alpha \vecb{g}_0^\top \vecb{p},
$$

with $c_1=\mathtt{opt.ls.c1}$.

## Purpose

Near a minimizer, good directions (Newton/quasi-Newton) often allow full steps.
Trying $\alpha=1$ first can reduce line-search computations and preserve fast local
convergence.

## Fallback behavior

If full step fails by rule, call inner strategy. If full-step function evaluation fails, return `eval_failed`.

## Usage note

Enable via `opt.ls.try_full_step` in options. Though allowed, using `TryFull{StepStrategy{}}` works, but is discouraged.
