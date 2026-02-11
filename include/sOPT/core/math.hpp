#pragma once

#include "sOPT/core/constants.hpp"
#include "sOPT/core/vecdefs.hpp"

namespace sOPT {

template <typename T, typename I>
inline T pow_Ti(T x, I n) {
    if (n == 0) return T(1.0);
    if (n < 0) {
        x = T(1.0) / x;
        n = -n;
    }

    T result = T(1.0);
    while (n) {
        if (n & 1) result *= x;
        x *= x;
        n >>= 1;
    }
    return result;
}

template <typename T>
inline T sign(T x, T eps = T(0)) {
    if (std::abs(x) <= eps) return T(0);
    return (x > T(0)) ? T(1) : T(-1);
}

template <typename T>
inline T wrap_pi(f64 a) {
    return std::atan2(std::sin(a), std::cos(a));
};

template <typename T>
inline T deg(T val) {
    return rad_to_deg * val;
}
template <typename T>
inline T rad(T val) {
    return deg_to_rad * val;
}

template <typename T>
svec<T> vieta(const eref<const vecX<T>>& poles) {
    // Vieta's formula for real or complex poles (works for any scalar T)
    svec<T> poly = {T(1)}; // start with 1

    for (int i = 0; i < poles.size(); ++i) {
        svec<T> new_poly(poly.size() + 1, T(0));
        for (size_t j = 0; j < poly.size(); ++j) {
            new_poly[j] += poly[j];                 // coefficient without this root
            new_poly[j + 1] += -poles(i) * poly[j]; // include this root
        }
        poly = new_poly;
    }

    return poly;
}

template <typename T>
vecX<T> conv(const eref<const vecX<T>>& a, const eref<const vecX<T>>& b) {
    static_assert(std::is_arithmetic_v<T>, "conv<T>: T must be an arithmetic type");

    if (a.empty() || b.empty()) return vecX<T>{};

    std::vector<T> y(a.size() + b.size() - 1, T{});

    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            y[i + j] += a[i] * b[j];
        }
    }
    return y;
}

template <typename T>
constexpr T eps(T x = 1.) {
    static_assert(
        std::is_floating_point<T>::value,
        "eps(x) requires a floating-point type"
    );

    return std::nextafter(x, std::numeric_limits<T>::infinity()) - x;
}

inline bool finite_nonneg(f64 v) {
    return isfinite(v) && v >= 0.0;
}

inline bool finite_pos(f64 v) {
    return isfinite(v) && v > 0.0;
}

// symmetrize in place
inline void sym_transpose_avg_ip(eref<matXd> M) {
    M = 0.5 * (M + M.transpose());
}
inline void sym_copy_lotohi_ip(eref<matXd> M) {
    M.template triangularView<esUp>() = M.transpose().template triangularView<esUp>();
}
inline void sym_copy_hitolo_ip(eref<matXd> M) {
    M.template triangularView<esLo>() = M.transpose().template triangularView<esLo>();
}

// return symmetrized matrix
inline matXd sym_transpose_avg(ecref<matXd> M) {
    return 0.5 * (M + M.transpose());
}
inline matXd sym_copy_lotohi(ecref<matXd> M) {
    matXd out = M;
    out.template triangularView<esUp>() = out.transpose().template triangularView<esUp>();
    return out;
}
inline matXd sym_copy_hitolo(ecref<matXd> M) {
    matXd out = M;
    out.template triangularView<esLo>() = out.transpose().template triangularView<esLo>();
    return out;
}

} // namespace sOPT