#pragma once

#include "sOPT/core/vecdefs.hpp"

namespace sOPT {

// First-order methods -----------------------------------------------------
template <typename OracleT>
inline bool fd_hv_forward(
    OracleT& oracle,
    ecref<vecXd> x,
    ecref<vecXd> v,
    eref<vecXd> Hv,
    f64 eps = 1e-6
) {
    const i32 n = static_cast<i32>(x.size());

    const f64 vnorm = v.norm(); // scale step by v to avoid tiny steps
    if (vnorm == 0.0) {
        Hv.setZero();
        return true;
    }
    const f64 h = eps * (1.0 + x.norm()) / vnorm;
    vecXd gx(n);
    vecXd gxph(n);
    vecXd xph = x;
    xph.noalias() += h * v;
    if (!oracle.try_gradient(x, gx)) return false;
    if (!oracle.try_gradient(xph, gxph)) return false;
    Hv.noalias() = (gxph - gx) / h;
    return Hv.allFinite();
}

template <typename OracleT>
inline bool fd_hv_backward(
    OracleT& oracle,
    ecref<vecXd> x,
    ecref<vecXd> v,
    eref<vecXd> Hv,
    f64 eps = 1e-6
) {
    const i32 n = static_cast<i32>(x.size());

    const f64 vnorm = v.norm(); // scale step by v to avoid tiny steps
    if (vnorm == 0.0) {
        Hv.setZero();
        return true;
    }
    const f64 h = eps * (1.0 + x.norm()) / vnorm;
    vecXd gx(n);
    vecXd gxmh(n);
    vecXd xmh = x;
    xmh.noalias() -= h * v;
    if (!oracle.try_gradient(x, gx)) return false;
    if (!oracle.try_gradient(xmh, gxmh)) return false;
    Hv.noalias() = (gx - gxmh) / h;
    return Hv.allFinite();
}

template <typename OracleT>
inline bool fd_hv_central(
    OracleT& oracle,
    ecref<vecXd> x,
    ecref<vecXd> v,
    eref<vecXd> Hv,
    f64 eps = 1e-6
) {
    const i32 n = static_cast<i32>(x.size());

    const f64 vnorm = v.norm(); // scale step by v to avoid tiny steps
    if (vnorm == 0.0) {
        Hv.setZero();
        return true;
    }
    const f64 h = eps * (1.0 + x.norm()) / vnorm;
    vecXd xmh = x;
    vecXd xph = x;
    vecXd gxmh(n);
    vecXd gxph(n);
    xmh.noalias() -= h * v;
    xph.noalias() += h * v;
    if (!oracle.try_gradient(xmh, gxmh)) return false;
    if (!oracle.try_gradient(xph, gxph)) return false;
    Hv.noalias() = (gxph - gxmh) / (2.0 * h);
    return Hv.allFinite();
}

// Second-order methods --------------------------------------------------------

template <typename OracleT>
inline bool fd_hv_forward_2(
    OracleT& oracle,
    ecref<vecXd> x,
    ecref<vecXd> v,
    eref<vecXd> Hv,
    f64 eps = 1e-6
) {
    const i32 n = static_cast<i32>(x.size());

    const f64 vnorm = v.norm(); // scale step by v to avoid tiny steps
    if (vnorm == 0.0) {
        Hv.setZero();
        return true;
    }
    const f64 h = eps * (1.0 + x.norm()) / vnorm;
    vecXd gx(n);
    vecXd gxph(n);
    vecXd gxp2h(n);
    vecXd xph = x;
    vecXd xp2h = x;
    xph.noalias() += h * v;
    xp2h.noalias() += 2.0 * h * v;
    if (!oracle.try_gradient(x, gx)) return false;
    if (!oracle.try_gradient(xph, gxph)) return false;
    if (!oracle.try_gradient(xp2h, gxp2h)) return false;
    Hv.noalias() = (-3.0 * gx + 4.0 * gxph - gxp2h) / (2.0 * h);
    return Hv.allFinite();
}

template <typename OracleT>
inline bool fd_hv_backward_2(
    OracleT& oracle,
    ecref<vecXd> x,
    ecref<vecXd> v,
    eref<vecXd> Hv,
    f64 eps = 1e-6
) {
    const i32 n = static_cast<i32>(x.size());

    const f64 vnorm = v.norm(); // scale step by v to avoid tiny steps
    if (vnorm == 0.0) {
        Hv.setZero();
        return true;
    }
    const f64 h = eps * (1.0 + x.norm()) / vnorm;
    vecXd gx(n);
    vecXd gxmh(n);
    vecXd gxm2h(n);
    vecXd xmh = x;
    vecXd xm2h = x;
    xmh.noalias() -= h * v;
    xm2h.noalias() -= 2.0 * h * v;
    if (!oracle.try_gradient(x, gx)) return false;
    if (!oracle.try_gradient(xmh, gxmh)) return false;
    if (!oracle.try_gradient(xm2h, gxm2h)) return false;
    Hv.noalias() = (3.0 * gx - 4.0 * gxmh + gxm2h) / (2.0 * h);
    return Hv.allFinite();
}

template <typename OracleT>
inline bool fd_hv_central_2(
    OracleT& oracle,
    ecref<vecXd> x,
    ecref<vecXd> v,
    eref<vecXd> Hv,
    f64 eps = 1e-6
) {
    const i32 n = static_cast<i32>(x.size());

    const f64 vnorm = v.norm(); // scale step by v to avoid tiny steps
    if (vnorm == 0.0) {
        Hv.setZero();
        return true;
    }
    const f64 h = eps * (1.0 + x.norm()) / vnorm;
    vecXd xmh = x;
    vecXd xm2h = x;
    vecXd xph = x;
    vecXd xp2h = x;
    vecXd gxmh(n);
    vecXd gxm2h(n);
    vecXd gxph(n);
    vecXd gxp2h(n);
    xmh.noalias() -= h * v;
    xm2h.noalias() -= 2.0 * h * v;
    xph.noalias() += h * v;
    xp2h.noalias() += 2.0 * h * v;
    if (!oracle.try_gradient(xmh, gxmh)) return false;
    if (!oracle.try_gradient(xm2h, gxm2h)) return false;
    if (!oracle.try_gradient(xph, gxph)) return false;
    if (!oracle.try_gradient(xp2h, gxp2h)) return false;
    Hv.noalias() = (-gxp2h + gxm2h + 8.0 * (gxph - gxmh)) / (12.0 * h);
    return Hv.allFinite();
}

} // namespace sOPT