# Damped Newton (Globalized Newton)

## Problem Setup

For smooth unconstrained minimization,

$$
\min_{\vecb{x}\in\R^n} f(\vecb{x}),
$$

Newton methods use second-order local models.

Notation:

- $\vecb{x}_k,\vecb{g}_k,\vecb{p}_k \in \R^n$
- $\vecb{H}_k \in \R^{n\times n}$

## Update Rule

Quadratic model at $\vecb{x}_k$:

$$
q_k(\vecb{p})=f(\vecb{x}_k)+\vecb{g}_k^\top \vecb{p}+\tfrac12 \vecb{p}^\top \vecb{H}_k \vecb{p},
\quad
\begin{aligned}
\vecb{g}_k&=\nabla f(\vecb{x}_k),\\ \vecb{H}_k&=\nabla^2 f(\vecb{x}_k).
\end{aligned}
$$

Exact Newton step solves:

$$
\vecb{H}_k \vecb{p}_k = -\vecb{g}_k.
$$

This implementation uses damped system:

$$
(\vecb{H}_k+\lambda_k I)\vecb{p}_k=-\vecb{g}_k,
$$

then line-search update:

$$
\vecb{x}_{k+1}=\vecb{x}_k+\alpha_k \vecb{p}_k.
$$

## Damping

If $\vecb{H}_k$ is indefinite or poorly conditioned, pure Newton directions can be
unstable or non-descent. Adding $\lambda_k I$ shifts eigenvalues and promotes
SPD structure for Cholesky solve.

## Damping strategy in this code

Repeated LLT tries:

$$
\begin{aligned}
\lambda&=0\;(t=0),\\
\lambda&=\mathtt{damping0}\;(t=1),\\
\lambda&\leftarrow\lambda\cdot\mathtt{damping\_scale}\;(t>1),
\end{aligned}
$$

with

$$
\vecb{H}_{\text{mod}}=\vecb{H}+\lambda \vecb{I}.
$$

If LLT keeps failing, fallback to LU solve on $H$.
If solve fails or yields non-finite direction, return `linear_solve_failed`.

## Practical notes

- Fast local convergence near a well-behaved minimizer.
- Higher per-iteration cost than first-order methods.
- Usually strongest when analytic gradients/Hessians are available.
