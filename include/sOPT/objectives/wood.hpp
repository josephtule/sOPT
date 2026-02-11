#pragma once

#include "sOPT/core/vecdefs.hpp"
#include <cassert>

namespace sOPT {

struct WoodNDChained {

    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());
        assert(n >= 4 && n % 2 == 0);

        f64 f = 0.;
        i32 k = (n - 2) / 2;
        for (i32 jj = 1; jj <= k; jj++) {
            const i32 ii = 2 * jj;
            const i32 i = ii - 1;

            const f64 xim1 = x(i - 1);
            const f64 xi = x(i);
            const f64 xip1 = x(i + 1);
            const f64 xip2 = x(i + 2);

            const f64 t1 = xim1 * xim1 - xi;
            const f64 t2 = xim1 - 1.;
            const f64 t3 = xip1 * xip1 - xip2;
            const f64 t4 = xip1 - 1.;
            const f64 t5 = xi + xip2 - 2.;
            const f64 t6 = xi - xip2;

            f += 100. * t1 * t1        //
                 + t2 * t2             //
                 + 90. * t3 * t3       //
                 + t4 * t4             //
                 + 10. * t5 * t5       //
                 + 1. / 10. * t6 * t6; //
        }

        return f;
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());
        assert(n % 2 == 0 && n >= 4);

        g.setZero();
        const i32 k = (n - 2) / 2;
        for (i32 jj = 1; jj <= k; ++jj) {
            const i32 ii = 2 * jj;
            const i32 i = ii - 1;

            const f64 xim1 = x(i - 1);
            const f64 xi = x(i);
            const f64 xip1 = x(i + 1);
            const f64 xip2 = x(i + 2);

            const f64 t1 = xim1 * xim1 - xi;
            const f64 t2 = xim1 - 1.;
            const f64 t3 = xi + xip2 - 2.;
            const f64 t4 = xi - xip2;
            const f64 t5 = xip1 * xip1 - xip2;
            const f64 t6 = xip1 - 1.;

            g(i - 1) += 400. * xim1 * t1 + 2. * t2;
            g(i + 0) += -200. * t1 + 20. * t3 + 0.2 * t4;
            g(i + 1) += 360. * xip1 * t5 + 2. * t6;
            g(i + 2) += -180. * t5 + 20. * t3 - 0.2 * t4;
        }
    }

    vecXd x0(i32 n) {
        assert(n >= 4 && n % 2 == 0);
        vecXd x(n);

        // double index (ii, jj) is 1-based indexing
        // single index (i, j) is 0-based indexing
        for (i32 ii = 1; ii <= n; ii++) {
            bool is_odd = (ii % 2 == 1);
            i32 i = ii - 1;

            if (ii <= 4) {
                if (is_odd)
                    x(i) = -3.;
                else
                    x(i) = -1.;
            } else {
                if (is_odd)
                    x(i) = -2.;
                else
                    x(i) = 0.;
            }
        }

        return x;
    }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 4 && n % 2 == 0);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

} // namespace sOPT