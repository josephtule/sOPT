# Fixed Step

## Rule

Use constant step size:

with $\vecb{x}_k,\vecb{p}_k\in\R^n$.

$$
\alpha = \alpha_{\text{fixed}}=\mathtt{opt.ls.alpha\_fixed},
\\
\vecb{x}_{k+1}=\vecb{x}_k+\alpha \vecb{p}_k.
$$

Then evaluate $f(\vecb{x}_{k+1})$.

## Acceptance in this implementation

Accepted iff $f(\vecb{x}_{k+1})$ is finite.

No condition is enforced.

## Practical guidance

- Good for controlled experiments and baselines.
- If divergence/oscillation occurs, decrease $\alpha$ (recommended in orders of magnitude)
