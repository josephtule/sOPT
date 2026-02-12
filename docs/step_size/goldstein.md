# Goldstein Line Search

Goldstein enforces a two-sided acceptance window around predicted linear
reduction.

## Setup

Let $\vecb{x},\vecb{p},\vecb{g}_0\in\R^n$.

Require:

$$
\vecb{g}_0^\top \vecb{p} < 0,
\qquad
0<c<0.5,
\qquad
\alpha_0>0,
$$

with $c=\mathtt{opt.ls.c1}$.

Define:

$$
\phi(\alpha)=f(\vecb{x}+\alpha \vecb{p}),
\qquad
\ell(\alpha)=f_0 + c\alpha \vecb{g}_0^\top \vecb{p},
\qquad
\nu(\alpha)=f_0 + (1-c)\alpha \vecb{g}_0^\top \vecb{p}.
$$

Because $\vecb{g}_0^\top \vecb{p}<0$, we have $\nu(\alpha)\le \ell(\alpha)$.

## Acceptance window

Accept if:

$$
\nu(\alpha) \le \phi(\alpha) \le \ell(\alpha).
$$

Interpretation:

- upper bound $\phi\le\ell$: enough decrease
- lower bound $\phi\ge\nu$: avoid "too short" model-inconsistent steps

## Bracket logic in this implementation

Maintain bracket $[a_{\text{lo}},a_{\text{hi}}]$.

- if $\phi(\alpha) > \ell(\alpha)$: too large step, set $a_{\text{hi}}=\alpha$
- if $\phi(\alpha) < \nu(\alpha)$: too small/eager, set $a_{\text{lo}}=\alpha$

Then:

- if $a_{\text{hi}}$ finite: bisect
- else: expand $\alpha\leftarrow 2\alpha$

## Practical notes

- More structured than plain Armijo, often reasonable step lengths.
- Needs smooth objective behavior; can be less forgiving under noise.
