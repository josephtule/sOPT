# Gradient Descent

## Problem Setup

For unconstrained smooth minimization,

$$
\min_{\vecb{x}\in\R^n} f(\vecb{x}),
$$

gradient descent uses only first-order information.

Notation:

- $\vecb{x}_k,\vecb{g}_k,\vecb{p}_k \in \R^n$

## Update Rule

At iterate $\vecb{x}_k$:

$$
\begin{aligned}
\vecb{g}_k &= \nabla f(\vecb{x}_k), \\
\vecb{p}_k &= -\vecb{g}_k, \\
\vecb{x}_{k+1} &= \vecb{x}_k + \alpha_k \vecb{p}_k.
\end{aligned}
$$

$\alpha_k>0$ is selected by a step strategy (Armijo/Wolfe/Goldstein/fixed).

## Practical notes

- Cheap iteration cost, but often many iterations.
- Strongly affected by scaling/conditioning and step strategy quality.
