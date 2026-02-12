# Core Math Utilities

Defined in `include/sOPT/core/math.hpp` containing miscellaneous helper math utilities including:

- `pow_Ti(T x, I n)`: integer-power exponentiation by squaring
- `sign(T x, T eps=0)`: sign with dead-zone threshold
- `wrap_pi(f64 a)`: angle wrapping via `atan2(sin, cos)`
- `deg(T val)`, `rad(T val)`: radians/degrees conversion
- `vieta(poles)`: polynomial coefficients from roots
- `conv(a, b)`: discrete linear convolution, or polynomial coefficients multiplication
- `eps(T x)`: machine spacing at value `x`
- `finite_nonneg(f64 v)`: checks if v is finite and non-negative
- `finite_pos(f64 v)`: checks if v is finite and positive
- `sym_transpose_avg(eref<matXd> M)`: symmetrizes M via the transpose average (append `_ip` to operate in place)
- `sym_copy_lotohi(eref<matXd>)`: symmetrizes M via copying the lower triangle to the upper (append `_ip` to operate in place)
- `sym_copy_hitolo(eref<matXd>)`: symmetrizes M via copying the upper triangle to the lower (append `_ip` to operate in place)

