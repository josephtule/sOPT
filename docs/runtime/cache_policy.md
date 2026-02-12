# Oracle Cache Policy and Sizing Guide

See implementation behavior in [oracle_cache.md](oracle_cache.md).

## What is cached

- `f_slots`: scalar objective values keyed by exact `x`.
- `g_slots`: gradient vectors keyed by exact `x`.
- `h_slots`: dense Hessians keyed by exact `x`.

Each cache is independent and uses LRU replacement.

## Practical defaults

General-use defaults in code:

- `f_slots = 4`
- `g_slots = 2`
- `h_slots = 0`

Reasoning:

- Line search and wrapper flows often revisit a few nearby/repeated points for
  `f`.
- Accepted-iterate refresh logic can benefit from a small gradient cache.
- Dense Hessian caching is memory-expensive and should be opt-in.

## Size by problem scale

Let $n$ be the system size (number of parameters in x).

Small ($n \lesssim 100$):

- `f_slots`: 4 to 8
- `g_slots`: 2 to 4
- `h_slots`: 1 to 2 (if Hessian reuse is likely)

Medium ($100 \lesssim n \lesssim 2000$):

- `f_slots`: 4 to 8
- `g_slots`: 2 to 4
- `h_slots`: usually 0

Large ($n \gtrsim 2000$):

- `f_slots`: 2 to 4
- `g_slots`: 1 to 2
- `h_slots`: 0

## When to disable cache

Set `opt.cache.enabled = false` when:

- objective/derivative calls are already cheap and deterministic
- memory is tight
