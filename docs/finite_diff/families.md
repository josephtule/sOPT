# Finite-Difference Families

Finite-difference methods in this repository are grouped by derivative object.

Dimensions:

- $\vecb{x},\vecb{g},\vecb{v}\in\R^n$
- $\vecb{H}\in\R^{n\times n}$

All finite difference families have the following variants:
- first order forward difference
- first order backward difference
- second order central difference
- second order forward difference
- second order backward difference
- fourth order backward difference
## 1) Gradient FD family

Goal: approximate $\nabla f(\vecb{x})$ componentwise.

Details: [gradient_fd.md](gradient_fd.md)

## 2) Hessian FD family

Goal: approximate $\nabla^2 f(\vecb{x})$ via finite differences of gradients.

Details: [hessian_fd.md](hessian_fd.md)

## 3) Hessian-vector FD family

Goal: approximate $\vecb{H}(\vecb{x})\vecb{v}$ without forming full $\vecb{H}$.

Details: [hv_fd.md](hv_fd.md)

## 4) Dispatch family (Oracle fallback order)

`Oracle<Obj>` chooses analytic derivatives given in an objective struct when available and falls back to FD methods above when not. The implementation returns failure if any required function call fails or if the
result contains non-finite entries.
