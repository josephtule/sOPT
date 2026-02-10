#pragma once

#include "sOPT/core/vecdefs.hpp"

namespace sOPT {
// ref: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

// First-order methods ---------------------------------------------------------

template <typename OracleT>
inline bool
fd_gradient_forward(OracleT oracle, ecref<vecXd> x, eref<vecXd> g, f64 eps = 1e-8) {
    const i32 n = static_cast<i32>(x.size());

    f64 fx = 0.0;
    if (!oracle.try_func(x, fx)) return false;
    vecXd xph = x;
    for (i32 i = 0; i < n; i++) {
        const f64 h = eps * (f64(1) + std::abs(x[i])); // perturb
        xph[i] = x[i] + h;
        f64 fxph = 0.0;
        if (!oracle.try_func(x, fxph)) return false;
        g[i] = (fxph - fx) / h;
        xph[i] = x[i];
    }
    return g.allFinite();
}

} // namespace sOPT