#pragma once

#include "sOPT/core/typedefs.hpp"

namespace sOPT {

template <typename T>
inline T sat_mul(T a, T b) {
    constexpr T maxv = std::numeric_limits<T>::max();
    if (a <= 0 || b <= 0) return 0;
    if (a > maxv / b) return maxv;
    return a * b;
}

inline i64 cache_bytes(i32 n, i32 slots) {
    constexpr i64 maxv = std::numeric_limits<i64>::max();
    if (n <= 0 || slots <= 0) return 0;

    const i64 n64 = static_cast<i64>(n);
    const i64 matrix_bytes = sat_mul(sat_mul(n64, n64), static_cast<i64>(sizeof(f64)));
    const i64 x_bytes = sat_mul(n64, static_cast<i64>(sizeof(f64)));
    const i64 per_entry
        = (matrix_bytes >= maxv - x_bytes) ? maxv : (matrix_bytes + x_bytes);
    return sat_mul(per_entry, static_cast<i64>(slots));
}

inline bool limit_enabled(i32 v) {
    return v >= 0;
}

} // namespace sOPT