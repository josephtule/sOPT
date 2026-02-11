#pragma once

#include "sOPT/core/vecdefs.hpp"
#include <cassert>

namespace sOPT {

// Rosenbrock function in R^n (chain form)
struct RosenbrockChained {
    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());
        assert(n >= 2);

        f64 f = 0.;

        for (i32 ii = 2; ii <= n; ii++) {
            i32 i = ii - 1;

            const f64 xim1 = x(i - 1);
            const f64 xi = x(i);
            const f64 t1 = xim1 * xim1 - xi;
            const f64 t2 = xim1 - 1.;
            f += 100. * t1 * t1 + t2 * t2;
        }

        return f;
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());
        assert(n >= 2);

        g.setZero();

        for (i32 ii = 2; ii <= n; ii++) {
            i32 i = ii - 1;

            const f64 xim1 = x(i - 1);
            const f64 xi = x(i);
            const f64 t1 = xim1 * xim1 - xi;
            const f64 t2 = xim1 - 1.;
            g(i - 1) += 400. * xim1 * t1 + 2. * t2;
            g(i + 0) += -200. * t1;
        }
    }

    vecXd x0(i32 n) {
        vecXd x(n);
        for (i32 ii = 1; ii <= n; ii++) {
            i32 i = ii - 1;
            switch (ii % 2) {
            case 1: x(i) = -1.2; break;
            case 0: x(i) = 1.; break;
            }
        }
        return x;
    }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 2);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

} // namespace sOPT
