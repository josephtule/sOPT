# Finite-Difference Gradient

Given $f : \R^n \to \R$, approximate
$\vecb{g}(\vecb{x}) = \nabla f(\vecb{x})$ componentwise, where
$\vecb{x},\vecb{g}\in\R^n$.

## Step Size Per Coordinate

For coordinate $i$, the implementation uses:

$$
h_i = \varepsilon\,(1 + |x_i|),
$$

where $\varepsilon = \mathtt{opt.fd.eps}$.

## Stencils

Forward difference:

$$
g_i(\vecb{x}) \approx \frac{f(\vecb{x} + h_i \unitv{e}_i) - f(\vecb{x})}{h_i}.
$$

Backward difference:

$$
g_i(\vecb{x}) \approx \frac{f(\vecb{x}) - f(\vecb{x} - h_i \unitv{e}_i)}{h_i}.
$$

Central difference:

$$
g_i(\vecb{x}) \approx \frac{f(\vecb{x} + h_i \unitv{e}_i) - f(\vecb{x} - h_i \unitv{e}_i)}{2 h_i}.
$$

Function evaluations:

- forward/backward: $n+1$
- central: $2n$

## Typical epsilon values

- Forward/backward: $\varepsilon \approx 10^{-8}$ to $10^{-7}$
- Central: $\varepsilon \approx 10^{-6}$ to $10^{-5}$

