# Finite-Difference Hessian-Vector Product ($\vecb{H}\vecb{v}$)

Approximate $\vecb{H}(\vecb{x})\vecb{v}$ using finite differences of the
gradient along direction $\vecb{v}$, where
$\vecb{x},\vecb{v}\in\R^n$.

## Directional scaling used in code

The implementation uses:

$$
h = \varepsilon\,\frac{1 + \norm{\vecb{x}}}{\norm{\vecb{v}}},
$$

where $\varepsilon = \mathtt{opt.fd.hv\_eps}$.

Then $\norm{h\vecb{v}} \approx \varepsilon(1+\norm{\vecb{x}})$, so perturbation
magnitude is stable across different $\norm{\vecb{v}}$.

If $\norm{\vecb{v}}=0$, implementation returns $\vecb{H}\vecb{v}=0$.

## Stencils

Forward:

$$
\vecb{H}\vecb{v} \approx \frac{\nabla f(\vecb{x} + h\vecb{v}) - \nabla f(\vecb{x})}{h}.
$$

Backward (current implementation):

$$
\vecb{H}\vecb{v} \approx \frac{\nabla f(\vecb{x} - h\vecb{v}) - \nabla f(\vecb{x})}{h}.
$$

Central:

$$
\vecb{H}\vecb{v} \approx \frac{\nabla f(\vecb{x} + h\vecb{v}) - \nabla f(\vecb{x} - h\vecb{v})}{2h}.
$$

All variants require two gradient evaluations in the current code paths.

## Typical `hv_eps` values

- central $\vecb{H}\vecb{v}$ FD: `hv_eps` around $10^{-6}$ to $10^{-5}$
- forward/backward $\vecb{H}\vecb{v}$ FD: often $10^{-8}$ to $10^{-6}$

Current default is `opt.fd.hv_eps = 1e-6`, which is a practical central-FD
starting point.
