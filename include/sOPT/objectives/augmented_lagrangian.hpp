#pragma once

#include "sOPT/core/vecdefs.hpp"

namespace sOPT {
struct AugmentedLagrangian {

    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());

        f64 f = 0.;

        const f64 l1 = -0.002008;
        const f64 l2 = -0.001900;
        const f64 l3 = -0.000261;

        for (i32 ii = 5; ii <= n; ii += 5) {
            const i32 i = ii - 1;

            f64 t1 = 1.;
            f64 t2 = 0.;
            for (i32 jj = 1; jj <= 5; jj++) {
                const i32 j = jj - 1;
                const f64 xip1mj = x(i + 1 - j);
                t1 *= xip1mj;
                t2 += xip1mj * xip1mj;
            }
            t2 += -10. - l1;

            const f64 xi = x(i);
            const f64 xim1 = x(i - 1);
            const f64 xim2 = x(i - 2);
            const f64 xim3 = x(i - 3);
            const f64 xim4 = x(i - 4);

            f64 t3 = xim3 * xim2 - 5. * xim1 * xi - l2;
            f64 t4 = xim4 * xim4 * xim4 + xim3 * xim3 * xim3 + 1. - l3;

            f += std::exp(t1) + 10. * (t2 * t2 + t3 * t3 + t4 * t4);
        }

        return f;
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());

        g.setZero();
        const f64 l1 = -0.002008;
        const f64 l2 = -0.001900;
        const f64 l3 = -0.000261;

        for (i32 ii = 5; ii <= n; ii += 5) {
            const i32 i = ii - 1;

            const f64 xim4 = x(i - 4);
            const f64 xim3 = x(i - 3);
            const f64 xim2 = x(i - 2);
            const f64 xim1 = x(i - 1);
            const f64 xi = x(i - 0);

            const f64 xim4_2 = xim4 * xim4;
            const f64 xim3_2 = xim3 * xim3;
            const f64 xim2_2 = xim2 * xim2;
            const f64 xim1_2 = xim1 * xim1;
            const f64 xi_2 = xi * xi;

            const f64 t1 = xim4 * xim3 * xim2 * xim1 * xi;
            const f64 t2 = xim4_2 + xim3_2 + xim2_2 + xim1_2 + xi_2 - 10. - l1;
            const f64 t3 = xim3 * xim2 - 5. * xim1 * xi - l2;
            const f64 t4 = xim4_2 * xim4 + xim3_2 * xim3 + 1. - l3;

            // \frac{\partial}{\partial x_k} s_1
            // \frac{\partial}{\partial x_k} s_2
            // \frac{\partial}{\partial x_k} s_3
            // \frac{\partial}{\partial x_k} s_4
            g(i - 4) += xim3 * xim2 * xim1 * xi * std::exp(t1)   //
                        + 40. * xim4 * t2                        //
                                                                 // + 0
                        + 60 * xim4 * xim4 * t4;                 // wrt_x{i-4}
            g(i - 3) += xim4 * xim2 * xim1 * xi * std::exp(t1)   //
                        + 40. * xim3 * t2                        //
                        + 20. * xim2 * t3                        //
                        + 60. * xim3_2 * t4;                     // wrt x_{i-3}
            g(i - 2) += xim4 * xim3 * xim1 * xi * std::exp(t1)   //
                        + 40. * xim2 * t2                        //
                        + 20. * xim3 * t3;                       // wrt x_{i-2}
            g(i - 1) += xim4 * xim3 * xim2 * xi * std::exp(t1)   //
                        + 40. * xim1 * t2                        //
                        - 100. * xi * t3;                        //
            g(i - 0) += xim4 * xim3 * xim2 * xim1 * std::exp(t1) //
                        + 40. * xi * t2                          //
                        - 100. * xim1 * t3;                      //
        }
    }

    vecXd x0(i32 n) {
        vecXd x(n);

        for (i32 ii = 1; ii <= n; ii++) {
            const i32 i = ii - 1;
            switch (ii % 5) {
            case 1:
                if (ii <= 2)
                    x(i) = -2;
                else
                    x(i) = -1;
                break;
            case 2:
                if (ii <= 2)
                    x(i) = 2;
                else
                    x(i) = -1;
                break;
            case 3: x(i) = 2; break;
            case 4: x(i) = -1; break;
            case 0: x(i) = -1; break;
            }
        }

        return x;
    }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 5);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

} // namespace sOPT