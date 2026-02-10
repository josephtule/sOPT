#pragma once

#include "sOPT/core/vecdefs.hpp"

namespace sOPT {
// ref: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

// First-order methods ---------------------------------------------------------

template <typename OracleT>
inline bool
fd_gradient_forward(OracleT oracle, ecref<vecXd> x, eref<vecXd> g, f64 eps = 1e-8) {
    const i32 n = static_cast<i32>(x.size());

    const f64 fx = 0.0;
    if (!oracle.try_func(x, fx)) return false;
    vecXd xph = x;
    for (i32 i = 0; i < n; i++) {
        const f64 h = eps * (1.0 + std::abs(x(i))); // perturb
        xph(i) = x(i) + h;
        f64 fxph = 0.0;
        if (!oracle.try_func(xph, fxph)) return false;
        g(i) = (fxph - fx) / h;
        xph(i) = x(i); // reset
    }
    return g.allFinite();
}

template <typename OracleT>
inline bool
fd_gradient_backward(OracleT oracle, ecref<vecXd> x, eref<vecXd> g, f64 eps = 1e-8) {
    const i32 n = static_cast<i32>(x.size());

    const f64 fx = 0.0;
    if (!oracle.try_func(x, fx)) return false;
    vecXd xmh = x;
    for (i32 i = 0; i < n; i++) {
        const f64 h = eps * (1.0 + std::abs(x(i)));
        xmh(i) = x(i) - h;
        f64 fxmh = 0.0;
        if (!oracle.try_func(xmh, fxmh)) return false;
        g(i) = (fx - fxmh) / h;
        xmh(i) = x(i); // reset
    }
}

template <typename OracleT>
inline bool
fd_gradient_central(OracleT oracle, ecref<vecXd> x, eref<vecXd> g, f64 eps = 1e-6) {
    const i32 n = static_cast<i32>(x.size());

    vecXd xph = x;
    vecXd xmh = x;
    for (i32 i = 0; i < n; i++) {
        const f64 h = eps * (1.0 + std::abs(x(i)));
        xmh(i) = x(i) - h;
        xph(i) = x(i) + h;
        f64 fxmh = 0.0;
        f64 fxph = 0.0;
        if (!oracle.try_func(xmh, fxmh)) return false;
        if (!oracle.try_func(xph, fxph)) return false;
        g(i) = (fxph - fxmh) / (2.0 * h);
        xmh(i) = x(i);
        xph(i) = x(i);
    }
    return g.allFinite();
}

// Second-order methods --------------------------------------------------------

template <typename OracleT>
inline bool
fd_gradient_forward_2(OracleT oracle, ecref<vecXd> x, eref<vecXd> g, f64 eps = 1e-6) {
    const i32 n = static_cast<i32>(x.size());

    f64 fx = 0.0;
    if (!oracle.try_func(x, fx)) return false;
    vecXd xph = x;
    vecXd xp2h = x;
    for (i32 i = 0; i < n; i++) {
        const f64 h = eps * (1.0 + std::abs(x(i)));
        xph(i) = x(i) + h;
        xp2h(i) = x(i) + 2.0 * h;
        f64 fxph = 0.0;
        f64 fxp2h = 0.0;
        if (!oracle.try_func(xph, fxph)) return false;
        if (!oracle.try_func(xp2h, fxp2h)) return false;
        g(i) = (-3.0 * fx + 4.0 * fxph - fxp2h) / (2.0 * h);
        xph(i) = x(i);
        xp2h(i) = x(i);
    }
    return g.allFinite();
}

template <typename OracleT>
inline bool
fd_gradient_backward_2(OracleT oracle, ecref<vecXd> x, eref<vecXd> g, f64 eps = 1e-6) {
    const i32 n = static_cast<i32>(x.size());

    f64 fx = 0.0;
    if (!oracle.try_func(x, fx)) return false;
    vecXd xmh = x;
    vecXd xm2h = x;
    for (i32 i = 0; i < n; i++) {
        const f64 h = eps * (1.0 + std::abs(x(i)));
        xmh(i) = x(i) - h;
        xm2h(i) = x(i) - 2.0 * h;
        f64 fxmh = 0.0;
        f64 fxm2h = 0.0;
        if (!oracle.try_func(xmh, fxmh)) return false;
        if (!oracle.try_func(xm2h, fxm2h)) return false;
        g(i) = (3.0 * fx - 4.0 * fxmh + fxm2h) / (2.0 * h);
        xmh(i) = x(i);
        xm2h(i) = x(i);
    }
    return g.allFinite();
}

template <typename OracleT>
inline bool
fd_gradient_central_2(OracleT oracle, ecref<vecXd> x, eref<vecXd> g, f64 eps = 1e-6) {
    const i32 n = static_cast<i32>(x.size());

    vecXd xph = x;
    vecXd xp2h = x;
    vecXd xmh = x;
    vecXd xm2h = x;
    for (i32 i = 0; i < n; i++) {
        const f64 h = eps * (1.0 + std::abs(x(i)));
        xph(i) = x(i) + h;
        xp2h(i) = x(i) + 2.0 * h;
        xmh(i) = x(i) - h;
        xm2h(i) = x(i) - 2.0 * h;
        f64 fxph = 0.0;
        f64 fxp2h = 0.0;
        f64 fxmh = 0.0;
        f64 fxm2h = 0.0;
        if (!oracle.try_func(xph, fxph)) return false;
        if (!oracle.try_func(xp2h, fxp2h)) return false;
        if (!oracle.try_func(xmh, fxmh)) return false;
        if (!oracle.try_func(xm2h, fxm2h)) return false;
        g(i) = (-fxp2h + fxm2h + 8.0 * (fxph - fxmh)) / (12.0 * h);
        xph(i) = x(i);
        xp2h(i) = x(i);
        xmh(i) = x(i);
        xm2h(i) = x(i);
    }
    return g.allFinite();
}

} // namespace sOPT