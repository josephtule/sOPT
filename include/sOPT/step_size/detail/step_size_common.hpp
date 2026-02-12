#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/typedefs.hpp"

namespace sOPT::detail {

inline f64 quad_min_val_slope(
    f64 a_lo,
    f64 f_lo,
    f64 df_lo,
    f64 a_hi,
    f64 f_hi
) {
    const f64 t = a_hi - a_lo;
    if (!finite_nonzero(t)) return qNaN<f64>;

    const f64 denom = 2.0 * (f_hi - f_lo - df_lo * t);
    if (!finite_pos(denom)) return qNaN<f64>;

    const f64 dt = -(df_lo * t * t) / denom;
    const f64 a = a_lo + dt;
    return isfinite(a) ? a : qNaN<f64>;
}

inline f64 cubic_min_val_slope(
    f64 a_base,
    f64 f_base,
    f64 df_base,
    f64 a1,
    f64 f1,
    f64 a2,
    f64 f2
) {
    const f64 t1 = a1 - a_base;
    const f64 t2 = a2 - a_base;
    if (!finite_nonzero(t1) || !finite_nonzero(t2)) return qNaN<f64>;

    const f64 t1s = t1 * t1;
    const f64 t2s = t2 * t2;
    if (!finite_nonzero(t1s) || !finite_nonzero(t2s)) return qNaN<f64>;

    const f64 d1 = f1 - f_base - df_base * t1;
    const f64 d2 = f2 - f_base - df_base * t2;
    const f64 denom = t1 - t2;
    if (!finite_nonzero(denom)) return qNaN<f64>;

    const f64 A = (d1 / t1s - d2 / t2s) / denom;
    const f64 B = (-t2 * d1 / t1s + t1 * d2 / t2s) / denom;
    if (!isfinite(A) || !isfinite(B)) return qNaN<f64>;

    auto model = [&](f64 t) -> f64 {
        return f_base + df_base * t + B * t * t + A * t * t * t;
    };
    auto try_candidate = [&](f64 t, f64& t_best, f64& f_best, bool& has_best) {
        if (!isfinite(t)) return;
        const f64 curvature = 2.0 * B + 6.0 * A * t;
        if (!finite_pos(curvature)) return;
        const f64 fm = model(t);
        if (!isfinite(fm)) return;
        if (!has_best || fm < f_best) {
            t_best = t;
            f_best = fm;
            has_best = true;
        }
    };

    f64 t_best = qNaN<f64>;
    f64 f_best = qNaN<f64>;
    bool has_best = false;

    if (std::abs(A) < 1e-16) {
        if (B > 0.0) try_candidate(-df_base / (2.0 * B), t_best, f_best, has_best);
    } else {
        const f64 disc = B * B - 3.0 * A * df_base;
        if (!finite_nonneg(disc)) return qNaN<f64>;
        const f64 root = std::sqrt(disc);
        try_candidate((-B + root) / (3.0 * A), t_best, f_best, has_best);
        try_candidate((-B - root) / (3.0 * A), t_best, f_best, has_best);
    }

    if (!has_best) return qNaN<f64>;
    const f64 a = a_base + t_best;
    return isfinite(a) ? a : qNaN<f64>;
}

inline f64 secant_root(f64 x0, f64 y0, f64 x1, f64 y1) {
    const f64 denom = y1 - y0;
    if (!finite_nonzero(denom)) return qNaN<f64>;
    const f64 x = x0 - y0 * (x1 - x0) / denom;
    return isfinite(x) ? x : qNaN<f64>;
}

inline f64 clamp_pad(f64 a, f64 alo, f64 ahi, f64 frac = 0.1) {
    if (alo > ahi) std::swap(alo, ahi);
    const f64 width = ahi - alo;
    if (!finite_pos(width)) return alo;
    const f64 pad = std::clamp(frac, 0.0, 0.49) * width;
    return std::clamp(a, alo + pad, ahi - pad);
}

inline f64 interpolated_zoom_trial(
    f64 a_lo,
    f64 f_lo,
    f64 df_lo,
    f64 a_hi,
    f64 f_hi,
    f64 frac = 0.1
) {
    f64 a = quad_min_val_slope(a_lo, f_lo, df_lo, a_hi, f_hi);
    if (!isfinite(a)) a = 0.5 * (a_lo + a_hi);
    return clamp_pad(a, a_lo, a_hi, frac);
}

} // namespace sOPT::detail
