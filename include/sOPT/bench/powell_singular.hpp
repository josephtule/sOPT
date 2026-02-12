#pragma once

#include "sOPT/core/vecdefs.hpp"
#include <cassert>

namespace sOPT {

// Powell singular function, n must be multiple of 4.
struct PowellSingularChained {

    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());
        assert(n % 4 == 0);
        assert(n >= 4);

        f64 f = 0.;

        i32 k = (n - 2) / 2;
        for (i32 jj = 1; jj <= k; jj++) {
            const i32 ii = 2 * jj;
            const i32 i = ii - 1;

            const f64 xim1 = x(i - 1);
            const f64 xi = x(i);
            const f64 xip1 = x(i + 1);
            const f64 xip2 = x(i + 2);

            const f64 t1 = xim1 + 10. * xi;
            const f64 t2 = xip1 - xip2;
            const f64 t3 = xi - 2. * xip1;
            const f64 t4 = xim1 - xip2;

            f += t1 * t1                    //
                 + 5. * t2 * t2             //
                 + t3 * t3 * t3 * t3        //
                 + 10. * t4 * t4 * t4 * t4; //
        }

        return f;
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());
        assert(n >= 4 && n % 4 == 0);

        g.setZero();
        i32 k = (n - 2) / 2;
        for (i32 jj = 1; jj <= k; jj++) {
            const i32 ii = 2 * jj;
            const i32 i = ii - 1;

            const f64 xim1 = x(i - 1);
            const f64 xi = x(i);
            const f64 xip1 = x(i + 1);
            const f64 xip2 = x(i + 2);

            const f64 t1 = xim1 + 10. * xi;
            const f64 t2 = xip1 - xip2;
            const f64 t3 = xi - 2. * xip1;
            const f64 t4 = xim1 - xip2;
            const f64 t3_3 = t3 * t3 * t3;
            const f64 t4_3 = t4 * t4 * t4;

            g(i - 1) += 2. * t1 + 40. * t4_3;
            g(i + 0) += 20. * t1 + 4. * t3_3;
            g(i + 1) += 10. * t2 - 8. * t3_3;
            g(i + 2) += -10. * t2 - 40. * t4_3;
        }
    }

    vecXd x0(i32 n) {
        assert(n >= 4 && n % 4 == 0);
        vecXd x(n);

        for (i32 ii = 1; ii <= n; ii++) {
            i32 i = ii - 1;
            switch (ii % 4) {
            case 1: x(i) = 3.; break;
            case 2: x(i) = -1.; break;
            case 3: x(i) = 3.; break;
            case 0: x(i) = 1.; break;
            }
        }

        return x;
    }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 4 && n % 4 == 0);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

static f64 func(ecref<vecXd> x) {
    const i32 n = static_cast<i32>(x.size());
    assert(n % 4 == 0);

    f64 f = 0.;
    for (i32 i = 0; i < n; i += 4) {
        const f64 x1 = x(i + 0);
        const f64 x2 = x(i + 1);
        const f64 x3 = x(i + 2);
        const f64 x4 = x(i + 3);

        const f64 t1 = x1 + 10. * x2;
        const f64 t2 = x3 - x4;
        const f64 t3 = x2 - 2. * x3;
        const f64 t4 = x1 - x4;

        f += t1 * t1             //
             + 5. * t2 * t2      //
             + t3 * t3 * t3 * t3 //
             + 10. * t4 * t4 * t4 * t4;
    }
    return f;
}
static vecXd x0_powellnd_chained(i32 n) {
    vecXd x(n);
    for (i32 i = 0; i < n; i += 4) {
        x(i + 0) = 3.;
        x(i + 1) = -1.;
        x(i + 2) = 0.;
        x(i + 3) = 1.;
    }
    return x;
}

static void gradient_old(ecref<vecXd> x, eref<vecXd> g) {
    const i32 n = static_cast<i32>(x.size());
    assert(n % 4 == 0);

    g.setZero();

    for (i32 i = 0; i < n; i += 4) {
        const f64 x1 = x(i + 0);
        const f64 x2 = x(i + 1);
        const f64 x3 = x(i + 2);
        const f64 x4 = x(i + 3);

        const f64 t1 = x1 + 10. * x2;
        const f64 t2 = x3 - x4;
        const f64 t3 = x2 - 2. * x3;
        const f64 t4 = x1 - x4;

        g(i + 0) += 2. * t1 + 40. * t4 * t4 * t4;
        g(i + 1) += 20. * t1 + 4. * t3 * t3 * t3;
        g(i + 2) += 10. * t2 - 8. * t3 * t3 * t3;
        g(i + 3) += -10. * t2 - 40. * t4 * t4 * t4;
    }
}
} // namespace sOPT
