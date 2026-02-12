# Finite-Difference Hessian

Given $f : \R^n \to \R$, approximate
$\vecb{H}(\vecb{x}) = \nabla^2 f(\vecb{x})$ by finite differences of
gradients, with $\vecb{x}\in\R^n$ and
$\vecb{H}(\vecb{x})\in\R^{n\times n}$.

## Step Size Per Coordinate

For column $j$, the implementation uses:

$$
h_j = \varepsilon\,(1 + |x_j|),
$$

where $\varepsilon = \mathtt{opt.fd.eps}$.

## Column formulas

Central gradient differencing:

$$
H_{:,j} \approx
\frac{\nabla f(\vecb{x} + h_j \unitv{e}_j) - \nabla f(\vecb{x} - h_j \unitv{e}_j)}{2 h_j}.
$$

Forward gradient differencing:

$$
H_{:,j} \approx
\frac{\nabla f(\vecb{x} + h_j \unitv{e}_j) - \nabla f(\vecb{x})}{h_j}.
$$

Backward gradient differencing:

$$
H_{:,j} \approx
\frac{\nabla f(\vecb{x}) - \nabla f(\vecb{x} - h_j \unitv{e}_j)}{h_j}.
$$

Gradient evaluations:

- forward/backward: $n+1$
- central: $2n$

Symmetry enforcement in implementation:

$$
\vecb{H} \leftarrow \tfrac12(\vecb{H}+\vecb{H}^\top).
$$

or by copying the lower triangle to the top (or vice-versa).

## Typical epsilon values

In practice, Hessian FD is often more noise-sensitive than gradient FD because it
subtracts gradients. If gradients come from FD fallback, noise compounds, and a
larger $\varepsilon$ is often needed for stability.

Typical practical starts (float64):

- central Hessian FD: $\varepsilon \in [10^{-6},10^{-4}]$
- forward/backward Hessian FD: $\varepsilon \in [10^{-8},10^{-6}]$

