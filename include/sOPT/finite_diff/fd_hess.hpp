#pragma once

#include "Eigen/Core"
#include "sOPT/core/math.hpp"
#include "sOPT/core/vecdefs.hpp"

#include <cmath>

// First-order methods ---------------------------------------------------------

namespace sOPT {
template <typename OracleT>
inline bool
fd_hessian_forward(OracleT& oracle, ecref<vecXd> x, eref<matXd> H, f64 eps = 1e-8) {
    const i32 n = static_cast<i32>(x.size());

    vecXd gx(n);
    if (!oracle.try_gradient(x, gx)) return false;
    vecXd xph = x;
    vecXd gxph(n);
    for (i32 j = 0; j < n; j++) {
        const f64 h = eps * (1.0 + std::abs(x(j)));
        xph(j) = x(j) + h;
        if (!oracle.try_gradient(xph, gxph)) return false;
        H.col(j).noalias() = (gxph - gx) / h;
        xph(j) = x(j);
    }
    H.template triangularView<eSUp>() = H.template triangularView<eSUp>().transpose();
    return H.allFinite();
}

template <typename OracleT>
inline bool
fd_hessian_backward(OracleT& oracle, ecref<vecXd> x, eref<matXd> H, f64 eps = 1e-8) {
    const i32 n = static_cast<i32>(x.size());

    vecXd gx(n);
    if (!oracle.try_gradient(x, gx)) return false;
    vecXd xmh = x;
    vecXd gxmh(n);
    for (i32 j = 0; j < n; j++) {
        const f64 h = eps * (1.0 * std::abs(x(j)));
        xmh(j) = x(j) - h;
        if (!oracle.try_gradient(xmh, gxmh)) return false;
        H.col(j).noalias() = (gx - gxmh) / h;
        xmh(j) = x(j);
    }
    H.template triangularView<eSUp>() = H.template triangularView<eSUp>().transpose();
    return H.allFinite();
}

template <typename OracleT>
inline bool
fd_hessian_central(OracleT& oracle, ecref<vecXd> x, eref<matXd> H, f64 eps = 1e-6) {
    const i32 n = static_cast<i32>(x.size());

    vecXd xph = x;
    vecXd xmh = x;
    vecXd gxph(n);
    vecXd gxmh(n);
    for (i32 j = 0; j < n; j++) { // loop over columns
        const f64 h = eps * (1.0 + std::abs(x(j)));
        xmh(j) = x(j) - h;
        xph(j) = x(j) + h;
        if (!oracle.try_gradient(xmh, gxmh)) return false;
        if (!oracle.try_gradient(xph, gxph)) return false;
        H.col(j).noalias() = (gxph - gxmh) / (2.0 * h);
        xmh(j) = x(j);
        xph(j) = x(j);
    }
    // make H symmetric
    H.template triangularView<eSUp>() = H.template triangularView<eSUp>().transpose();
    return H.allFinite();
}

// Second-order methods --------------------------------------------------------

template <typename OracleT>
inline bool
fd_hessian_forward_2(OracleT& oracle, ecref<vecXd> x, eref<matXd> H, f64 eps = 1e-6) {
    const i32 n = static_cast<i32>(x.size());

    vecXd gx(n);
    if(!oracle.try_gradient(x,gx)) return false;
    vecXd xph = x;
    vecXd xp2h = x;
    vecXd gxph(n);
    vecXd gxp2h(n);
    for (i32 j = 0; j < n; j++) { // loop over columns
        const f64 h = eps * (1.0 + std::abs(x(j)));
        xph(j) = x(j) + h;
        xp2h(j) = x(j) + 2.0 * h;
        if (!oracle.try_gradient(xph, gxph)) return false;
        if (!oracle.try_gradient(xp2h, gxp2h)) return false;
        H.col(j).noalias() = (-gx + 4.0*gxph - gxp2h ) / (2.0 * h);
        xph(j) = x(j);
        xp2h(j) = x(j);
    }
    // make H symmetric 
    H.template triangularView<eSUp>() = H.template triangularView<eSUp>().transpose();
    return H.allFinite();
}

template <typename OracleT>
inline bool
fd_hessian_backward_2(OracleT& oracle, ecref<vecXd> x, eref<matXd> H, f64 eps = 1e-6) {
    const i32 n = static_cast<i32>(x.size());

    vecXd gx(n);
    if(!oracle.try_gradient(x,gx)) return false;
    vecXd xmh = x;
    vecXd xm2h = x;
    vecXd gxmh(n);
    vecXd gxm2h(n);
    for (i32 j = 0; j < n; j++) { // loop over columns
        const f64 h = eps * (1.0 + std::abs(x(j)));
        xmh(j) = x(j) - h;
        xm2h(j) = x(j) - 2.0 * h;
        if (!oracle.try_gradient(xmh, gxmh)) return false;
        if (!oracle.try_gradient(xm2h, gxm2h)) return false;
        H.col(j).noalias() = (gx - 4.0*gxmh + gxm2h ) / (2.0 * h);
        xmh(j) = x(j);
        xm2h(j) = x(j);
    }
    // make H symmetric 
    H.template triangularView<eSUp>() = H.template triangularView<eSUp>().transpose();
    return H.allFinite();
}

template <typename OracleT>
inline bool
fd_hessian_central_2(OracleT& oracle, ecref<vecXd> x, eref<matXd> H, f64 eps = 1e-6) {
    const i32 n = static_cast<i32>(x.size());

    vecXd xph = x;
    vecXd xp2h = x;
    vecXd xmh = x;
    vecXd xm2h = x;
    vecXd gxph(n);
    vecXd gxp2h(n);
    vecXd gxmh(n);
    vecXd gxm2h(n);
    for (i32 j = 0; j < n; j++) { // loop over columns
        const f64 h = eps * (1.0 + std::abs(x(j)));
        xmh(j) = x(j) - h;
        xm2h(j) = x(j) - 2.0 * h;
        xph(j) = x(j) + h;
        xp2h(j) = x(j) + 2.0 * h;
        if (!oracle.try_gradient(xmh, gxmh)) return false;
        if (!oracle.try_gradient(xm2h, gxm2h)) return false;
        if (!oracle.try_gradient(xph, gxph)) return false;
        if (!oracle.try_gradient(xp2h, gxp2h)) return false;
        H.col(j).noalias() = (-gxp2h + gxm2h + 8.0 * (gxph - gxmh)) / (12.0 * h);
        xmh(j) = x(j);
        xm2h(j) = x(j);
        xph(j) = x(j);
        xp2h(j) = x(j);
    }
    // make H symmetric 
    // H.template triangularView<esUp>() = H.template triangularView<esUp>().transpose();
    sym_copy_lotohi_ip(H);
    return H.allFinite();
}

} // namespace sOPT
