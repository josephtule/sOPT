# Oracle Cache Behavior

The oracle maintains separate LRU caches keyed by exact $x$ equality:

- function cache (`f`)
- gradient cache (`g`)
- Hessian cache (`H`)

Cache slots are configured in `opt.cache`:

- `f_slots`
- `g_slots`
- `h_slots`
- global `enabled`

## Key Matching

Two keys match only if all coordinates are exactly equal:

$$
(a.\mathtt{array()} == b.\mathtt{array()}).\mathtt{all()}.
$$

No tolerance-based keying is used.

## Replacement Policy

Per cache:

1. Use empty slot if available.
2. Otherwise evict least-recently-used entry (smallest `lru_tick`).

Each hit updates `lru_tick`.

## Counters

Each cache tracks:

- hits
- misses

Oracle separately tracks:

- `f_evals`, `g_evals`, `h_evals`, `hv_evals`

## Hessian Cache Memory Guard

If enabled, oracle estimates Hessian-cache bytes:

$$
\mathtt{per\_entry} \approx n^2\,\mathtt{sizeof(f64)} + n\,\mathtt{sizeof(f64)},
\\
\mathtt{needed} = \mathtt{per\_entry}\cdot \mathtt{h\_slots}.
$$

If $\mathtt{needed} > \mathtt{opt.cache.max\_bytes}$ and
`enforce_max_bytes=true`, Hessian cache is disabled at runtime.

## See also: [cache_policy.md](cache_policy.md).
