# L-BFGS (Limited-Memory BFGS)

## Problem Setup

L-BFGS targets the same problem class as BFGS,

$$
\min_{\vecb{x}\in\R^n} f(\vecb{x}),
$$

but avoids storing a dense inverse Hessian approximation.

Notation:

- $\vecb{x}_k,\vecb{g}_k,\vecb{p}_k \in \R^n$
- $\vecb{s}_i,\vecb{y}_i,\vecb{q},\vecb{r} \in \R^n$
- implicit inverse-Hessian operator $\vecb{H}_k:\R^n\to\R^n$

L-BFGS approximates the action of a BFGS inverse Hessian using only the last
$m$ secant pairs, giving quasi-Newton behavior with memory $O(mn)$ instead of
$O(n^2)$.

## Update Rule

$$
\begin{aligned}
\vecb{g}_k &= \nabla f(\vecb{x}_k), \\
\vecb{p}_k &= -\vecb{H}_k \vecb{g}_k\quad(\text{implicit}), \\
\vecb{x}_{k+1} &= \vecb{x}_k + \alpha_k \vecb{p}_k.
\end{aligned}
$$

Store recent secant pairs (most recent at back):

$$
\begin{aligned}
\vecb{s}_i &= \vecb{x}_{i+1}-\vecb{x}_i, \\
\vecb{y}_i &= \vecb{g}_{i+1}-\vecb{g}_i, \\
\rho_i &= \frac{1}{\vecb{y}_i^\top \vecb{s}_i}.
\end{aligned}
$$

## Two-loop recursion

Backward loop:

$$
\begin{aligned}
\vecb{q} &\leftarrow \vecb{g}_k, \\
\alpha_i &\leftarrow \rho_i \vecb{s}_i^\top \vecb{q}, \\
\vecb{q} &\leftarrow \vecb{q}-\alpha_i \vecb{y}_i,
\quad i=L-1,\dots,0.
\end{aligned}
$$

Initial scaling:

$$
\gamma = \frac{\vecb{s}_{\text{last}}^\top \vecb{y}_{\text{last}}}{\vecb{y}_{\text{last}}^\top \vecb{y}_{\text{last}}},
\qquad
\gamma\leftarrow\operatorname{clamp}(\gamma, h0_{\min}, h0_{\max}),
$$

then $\vecb{r}\leftarrow\gamma \vecb{q}$ (or $\vecb{r}\leftarrow \vecb{q}$ when scaling disabled/no history).

Forward loop:

$$
\begin{aligned}
\beta_i &\leftarrow \rho_i \vecb{y}_i^\top \vecb{r}, \\
\vecb{r} &\leftarrow \vecb{r} + \vecb{s}_i(\alpha_i-\beta_i),
\quad i=0,\dots,L-1.
\end{aligned}
$$

Direction:

$$
\vecb{p}_k\leftarrow-\vecb{r}.
$$

## Practical notes

- Default memory `m=20` for general-purpose starting point.
- Increase `m` for smoother, harder curvature; decrease `m` for very large $n$.