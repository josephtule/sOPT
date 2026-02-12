# Wolfe Line Search (Weak and Strong)

Wolfe searches enforce both decrease and curvature conditions. 
## Setup

Let $\vecb{x},\vecb{p},\vecb{g}_0\in\R^n$ with descent
$\vecb{g}_0^\top\vecb{p}<0$. Define along direction $\vecb{p}$:

$$
\phi(\alpha)=f(\vecb{x}+\alpha \vecb{p}),
\\
\phi'(\alpha)=\nabla f(\vecb{x}+\alpha \vecb{p})^\top \vecb{p}.
$$

Require:

$$
\phi'(0)=\vecb{g}_0^\top \vecb{p}<0,
\\
0<c_1<c_2<1,
\\
\alpha_0>0.
$$

## Conditions

Armijo decrease:

$$
\phi(\alpha)\le\phi(0)+c_1\alpha\phi'(0).
$$

Weak Wolfe curvature:

$$
\phi'(\alpha)\ge c_2\phi'(0).
$$

Strong Wolfe curvature:

$$
|\phi'(\alpha)|\le -c_2\phi'(0).
$$

Strong Wolfe controls gradient magnitude at new point, often helping BFGS/L-BFGS
curvature quality.

## Search structure in this implementation

Outer stage:

1. test Armijo at current $\alpha$
2. if Armijo fails (or non-monotone vs previous), call `zoom`
3. evaluate $\phi'(\alpha)$
4. accept if Wolfe holds
5. if $\phi'(\alpha)\ge 0$, call `zoom` with reversed bracket
6. else expand $\alpha\leftarrow\min(2\alpha,\alpha_{\max})$

Zoom stage:

- repeatedly bisect interval
- evaluate function/gradient at midpoint
- shrink bracket based on Armijo and slope sign rules
- stop when Wolfe holds or max iterations reached

## Practical notes

- Usually best default for BFGS/L-BFGS.
- More gradient evaluations than Armijo-only searches.
- `c2=0.9` is common for strong Wolfe in quasi-Newton practice.
