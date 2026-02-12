# BFGS (Inverse-Hessian Form)

## Problem Setup

For smooth unconstrained minimization,

$$
\min_{\vecb{x}\in\R^n} f(\vecb{x}),
$$

BFGS builds an approximation to the inverse Hessian and uses it to precondition
the gradient direction.

Notation:

- $\vecb{x}_k,\vecb{g}_k,\vecb{p}_k,\vecb{s}_k,\vecb{y}_k \in \R^n$
- $\vecb{B}_k,\vecb{H}_k,\vecb{V}_k \in \R^{n\times n}$

## Update Rule

$$
\begin{aligned}
\vecb{g}_k &= \nabla f(\vecb{x}_k), \\
\vecb{p}_k &= -\vecb{B}_k \vecb{g}_k, \\
\vecb{x}_{k+1} &= \vecb{x}_k + \alpha_k \vecb{p}_k.
\end{aligned}
$$

Here $\vecb{B}_k\approx \vecb{H}_k^{-1}$.

Define secant pair:

$$
\begin{aligned}
\vecb{s}_k &= \vecb{x}_{k+1}-\vecb{x}_k, \\
\vecb{y}_k &= \vecb{g}_{k+1}-\vecb{g}_k, \\
\rho_k &= \frac{1}{\vecb{y}_k^\top \vecb{s}_k}.
\end{aligned}
$$

Update used in this implementation:

$$
\begin{aligned}
\vecb{V}_k &= \vecb{I}-\rho_k \vecb{s}_k \vecb{y}_k^\top, \\
\vecb{B}_{k+1} &= \vecb{V}_k \vecb{B}_k \vecb{V}_k^\top + \rho_k \vecb{s}_k \vecb{s}_k^\top.
\end{aligned}
$$

## Curvature condition and SPD behavior

To keep $B_{k+1}$ well-behaved, require positive curvature:

$$
\vecb{y}_k^\top \vecb{s}_k > 0.
$$

Strong Wolfe line search typically helps satisfy this, which is why it is the
default for BFGS.

## Practical notes

- Usually much faster than GD on smooth problems.
- For larger $n$, L-BFGS is usually preferred.
