# Armijo Backtracking

## Setup

At current state $(\vecb{x},f_0,\vecb{g})$ and direction
$\vecb{p}$ with $\vecb{x},\vecb{g},\vecb{p}\in\R^n$, require
descent:

$$
\vecb{g}^\top \vecb{p} < 0.
$$

Start from

$$
\alpha_0 = \mathtt{opt.ls.alpha0},
$$

with parameters

$$
0<\rho<1,\quad \rho=\mathtt{opt.ls.rho},
\qquad
0<c_1<1,\quad c_1=\mathtt{opt.ls.c1}.
$$

## Acceptance condition

Armijo (sufficient decrease):

$$
f(\vecb{x}+\alpha \vecb{p}) \le f_0 + c_1\alpha \vecb{g}^\top \vecb{p}.
$$

If condition fails, shrink:

$$
\alpha \leftarrow \rho\alpha.
$$

Repeat up to `opt.ls.max_iters`.


## Parameter effects

- Smaller $c_1$ (e.g. $10^{-4}$): easier acceptance, larger steps.
- Larger $c_1$: stricter decrease, more conservative steps.
- Smaller $\rho$ (e.g. $0.1$): aggressive shrinking.
- Larger $\rho$ (e.g. $0.8$): gentle shrinking, more line-search iterations.