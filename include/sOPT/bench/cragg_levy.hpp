#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/vecdefs.hpp"

#include <cassert>

namespace sOPT {

struct CraggLevyChained {

    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());

        f64 f = 0.;
        i32 k = (n - 2) / 2;
        for (i32 jj = 1; jj <= k; jj++) {
            const i32 ii = 2 * jj;
            const i32 i = ii - 1;

            const f64 xim1 = x(i - 1);
            const f64 xi = x(i);
            const f64 xip1 = x(i + 1);
            const f64 xip2 = x(i + 2);

            const f64 t1 = std::exp(xim1) - xi;
            const f64 t2 = xi - xip1;
            const f64 t3 = xip1 - xip2;
            const f64 t4 = xip2 - 1.;
            const f64 t1_4 = pow_Ti(t1, 4);
            const f64 t2_6 = pow_Ti(t2, 6);
            const f64 tant3 = std::tan(t3);
            const f64 tant3_4 = pow_Ti(tant3, 4);
            const f64 xim1_8 = pow_Ti(xim1, 8);

            f += t1_4          //
                 + 100. * t2_6 //
                 + tant3_4     //
                 + xim1_8      //
                 + t4 * t4;    //
        }

        return f;
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());

        g.setZero();
        const i32 k = (n - 2) / 2;
        for (i32 jj = 1; jj <= k; ++jj) {
            const i32 ii = 2 * jj;
            const i32 i = ii - 1;

            const f64 xim1 = x(i - 1);
            const f64 xi = x(i);
            const f64 xip1 = x(i + 1);
            const f64 xip2 = x(i + 2);

            const f64 t1 = std::exp(xim1) - xi;
            const f64 t2 = xi - xip1;
            const f64 t3 = xip1 - xip2;
            const f64 t4 = xip2 - 1.;
            const f64 t1_3 = pow_Ti(t1, 3);
            const f64 t2_5 = pow_Ti(t2, 5);
            const f64 tant3 = std::tan(t3);
            const f64 tant3_3_sect3_2 = pow_Ti(tant3, 3) * (1. + tant3 * tant3);

            g(i - 1) += 4. * std::exp(xim1) * t1_3 + 8. * pow_Ti(xim1, 7);
            g(i + 0) += -4. * t1_3 + 600. * t2_5;
            g(i + 1) += -600. * t2_5 + 4. * tant3_3_sect3_2;
            g(i + 2) += -4. * tant3_3_sect3_2 + 2. * t4;
        }
    }

    vecXd x0(i32 n) {
        vecXd x(n);

        for (i32 ii = 1; ii <= n; ii++) {
            i32 i = ii - 1;
            if (ii == 1)
                x(i) = 1;
            else
                x(i) = 2;
        }

        return x;
    }

    bool check_x0(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 4 && n % 2 == 0);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x0(x)); }
};

} // namespace sOPT