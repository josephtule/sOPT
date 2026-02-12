# DFP (Davidon-Fletcher-Powell)

Implementation:

- [`include/sOPT/algorithms/dfp.hpp`](../../include/sOPT/algorithms/dfp.hpp)

## Update Rule

Given $(\vecb{x}_k, \vecb{g}_k)$ and inverse-Hessian approximation $\vecb{B}_k$:

Notation:

- $\vecb{x}_k,\vecb{g}_k,\vecb{p}_k,\vecb{s}_k,\vecb{y}_k \in \R^n$
- $\vecb{B}_k \in \R^{n\times n}$

$$
\vecb{p}_k = -\vecb{B}_k \vecb{g}_k,
\qquad
\vecb{x}_{k+1} = \vecb{x}_k + \alpha_k \vecb{p}_k.
$$

With

$$
\vecb{s}_k = \vecb{x}_{k+1} - \vecb{x}_k,\qquad \vecb{y}_k = \vecb{g}_{k+1} - \vecb{g}_k,
$$

the DFP inverse update is

$$
\vecb{B}_{k+1}
= \vecb{B}_k + \frac{\vecb{s}_k \vecb{s}_k^\top}{\vecb{y}_k^\top \vecb{s}_k}
- \frac{\vecb{B}_k \vecb{y}_k \vecb{y}_k^\top \vecb{B}_k}{\vecb{y}_k^\top \vecb{B}_k \vecb{y}_k}.
$$

## Practical notes

- DFP conditions are slightly less stable than BFGS/L-BFGS conditions.

