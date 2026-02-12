# Notation and Terms Glossary

This page defines the main symbols and abbreviations used across the repo (unless overrided within each page).

## Dimension conventions

Unless a page states otherwise:

- $\vecb{x}, \vecb{g}, \vecb{p}, \vecb{s}, \vecb{y}, \vecb{v} \in \R^n$
- $\vecb{H}, \vecb{B} \in \R^{n\times n}$

## Core iteration symbols

| Symbol | Meaning |
| --- | --- |
| $\vecb{x}_k$ | iterate at iteration $k$ |
| $f_k$ | objective value at $\vecb{x}_k$, i.e. $f_k = f(\vecb{x}_k)$ |
| $\vecb{g}_k$ | gradient at $\vecb{x}_k$, i.e. $\vecb{g}_k = \nabla f(\vecb{x}_k)$ |
| $\vecb{H}_k$ | Hessian at $\vecb{x}_k$, i.e. $\vecb{H}_k = \nabla^2 f(\vecb{x}_k)$ |
| $\vecb{B}_k$ | inverse-Hessian approximation (usually quasi-Newton) |
| $\vecb{p}_k$ | search direction or trial step from $\vecb{x}_k$ |
| $\alpha_k$ | step length applied to direction $\vecb{p}_k$ |
| $\vecb{s}_k$ | iterate difference, $\vecb{s}_k = \vecb{x}_{k+1} - \vecb{x}_k$ |
| $\vecb{y}_k$ | gradient difference, $\vecb{y}_k = \vecb{g}_{k+1} - \vecb{g}_k$ |
| $\rho_k$ | secant scaling, often $\rho_k = 1/(\vecb{y}_k^\top \vecb{s}_k)$ |
| $\lambda_k$ | damping parameter (modified Newton) |

## Common abbreviations

| Term | Meaning |
| --- | --- |
| QN | quasi-Newton |
| GD | gradient descent |
| NCG | nonlinear conjugate gradient |
| BFGS | Broyden-Fletcher-Goldfarb-Shanno |
| L-BFGS | limited-memory BFGS |
| DFP | Davidon-Fletcher-Powell |
| $\vecb{H}\vecb{v}$ | Hessian-vector product, $\vecb{H}(\vecb{x})\vecb{v}$ |
| FD | finite difference |
| SPD | symmetric positive definite |
| PD / PSD | positive definite / positive semidefinite |

## Line-search constants

| Symbol | Meaning |
| --- | --- |
| $c_1$ | sufficient-decrease constant (Armijo component) |
| $c_2$ | curvature constant (Wolfe component) |

## Tolerance symbols

| Symbol | Meaning | Options field |
| --- | --- | --- |
| $\tau_g$ | absolute gradient tolerance | `opt.term.grad_tol` |
| $\tau_{g,\mathrm{rel}}$ | relative gradient tolerance multiplier | `opt.term.grad_tol_rel` |
| $\tau_s$ | absolute step tolerance | `opt.term.step_tol` |
| $\tau_{s,\mathrm{rel}}$ | relative step tolerance multiplier | `opt.term.step_tol_rel` |
| $\tau_f$ | relative objective-change tolerance multiplier | `opt.term.f_tol` |
