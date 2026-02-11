#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/vecdefs.hpp"

#include <cassert>

namespace sOPT {

struct BroydenGenTridiag {

    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());

        f64 f = 0.;
        const f64 p = 7. / 3.;
        for (i32 ii = 1; ii <= n; ii++) {
            const i32 i = ii - 1;

            const f64 xim1 = (i > 0) ? x(i - 1) : 0.0;     // x_{i} in 1-based
            const f64 xi = x(i);                           // x_{i+1}
            const f64 xip1 = (i + 1 < n) ? x(i + 1) : 0.0; // x_{i+2}

            const f64 q = (3. - 2 * xi) * xi - xim1 - xip1 + 1.;

            f += std::pow(std::abs(q), p);
        }

        return f;
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());

        g.setZero();
        const f64 p = 7. / 3.;
        for (i32 ii = 1; ii <= n; ++ii) {
            const i32 i = ii - 1;

            const f64 xim1 = (ii == 1) ? 0. : x(i - 1);
            const f64 xi = x(i);
            const f64 xip1 = (ii == n) ? 0. : x(i + 1);

            const f64 q = (3. - 2 * xi) * xi - xim1 - xip1 + 1.;
            const f64 aq = std::abs(q);
            // const f64 w = p * q * std::pow(aq, p - 2.);
            // const f64 w = (aq == 0.0) ? 0.0 : p * q * std::pow(aq, p - 2.0);
            // equivalent to the above, more numerically stable
            const f64 w = p * sign(q) * std::pow(aq, p - 1.);

            if (ii > 1) {
                g(i - 1) += -w;
            }
            g(i) += (3.0 - 4.0 * xi) * w;
            if (ii < n) {
                g(i + 1) += -w;
            }
        }
    }

    vecXd x0(i32 n) { return vecXd::Constant(n, -1.); }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 1);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

// -----------------------------------------------------------------------------

struct BroydenGenBanded {
    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());

        f64 f = 0.;
        const f64 p = 7. / 3.;
        for (i32 ii = 1; ii <= n; ii++) {
            const i32 i = ii - 1;

            const f64 xi = x(i);

            f64 val = 0.;
            val += (2. + 5. * xi * xi) * xi + 1.;

            const i32 lo = std::max(1, ii - 5);
            const i32 hi = std::min(n, ii + 1);
            for (i32 jj = lo; jj <= hi; jj++) {
                if (jj == ii) continue;
                const i32 j = jj - 1;
                const f64 xj = x(j);
                val += xj * (1. + xj);
            }

            f += std::pow(std::abs(val), p);
        }

        return f;
    }

    // no gradient

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());

        g.setZero();
        vecXd s(n);

        const f64 p = 7. / 3.;
        for (i32 ii = 1; ii <= n; ii++) {
            const i32 i = ii - 1;

            const f64 xi = x(i);

            f64 si = (2. + 5. * xi * xi) * xi + 1.;

            const i32 lo = std::max(1, ii - 5);
            const i32 hi = std::min(n, ii + 1);
            for (i32 jj = lo; jj <= hi; jj++) {
                if (jj == ii) continue;
                const i32 j = jj - 1;
                const f64 xj = x(j);
                si += (1. + xj) * xj;
            }
            s(i) = si;
        }

        for (i32 ii = 1; ii <= n; ii++) {
            const i32 i = ii - 1;

            const f64 si = s(i);
            if (si == 0.) continue;

            // const f64 wi = p * si * std::pow(std::abs(si), p - 2);
            // equivalent to the above, numerically safer
            const f64 wi = p * std::pow(std::abs(si), p - 1.) * sign(si);

            const f64 xi = x(i);
            g(i) += wi * (2. + 15. * xi * xi);

            const i32 lo = std::max(1, ii - 5);
            const i32 hi = std::min(n, ii + 1);
            for (i32 jj = lo; jj <= hi; jj++) {
                if (jj == ii) continue;
                const i32 j = jj - 1;
                const f64 xj = x(j);
                g(j) += wi * (1. + 2. * xj);
            }
        }
    }

    vecXd x0(i32 n) {
        // return -vecXd::Ones(n);
        return vecXd::Constant(n, -1.);
    }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 1);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

// -----------------------------------------------------------------------------

struct BroydenGen7Diag {
    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());

        f64 f = 0.;
        const f64 p = 7. / 3.;
        for (i32 ii = 1; ii <= n; ii++) {
            const i32 i = ii - 1;

            const f64 xim1 = (ii == 1) ? 0. : x(i - 1);
            const f64 xi = x(i);
            const f64 xip1 = (ii == n) ? 0. : x(i + 1);

            f += std::pow(std::abs((3. - 2. * xi) * xi - xim1 - xip1 + 1.), p);
            if (ii <= n / 2) {
                const i32 jj = ii + n / 2;
                const i32 j = jj - 1;
                f += std::pow(std::abs(xi + x(j)), p);
            }
        }

        return f;
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());

        g.setZero();
        const f64 p = 7. / 3.;
        for (i32 ii = 1; ii <= n; ++ii) {
            const i32 i = ii - 1;

            const f64 xim1 = (ii == 1) ? 0. : x(i - 1);
            const f64 xi = x(i);
            const f64 xip1 = (ii == n) ? 0. : x(i + 1);

            const f64 q = (3. - 2 * xi) * xi - xim1 - xip1 + 1.;
            const f64 aq = std::abs(q);

            const f64 w = p * sign(q) * std::pow(aq, p - 1.);

            if (ii > 1) {
                g(i - 1) += -w;
            }
            g(i) += (3.0 - 4.0 * xi) * w;
            if (ii < n) {
                g(i + 1) += -w;
            }

            if (ii <= n / 2) {
                const i32 j = i + n / 2;
                const f64 xipnd2 = x(j);
                const f64 r = xi + xipnd2;
                const f64 wr = p * sign(r) * std::pow(std::abs(r), p - 1.);

                g(i) += wr;
                g(j) += wr;
            }
        }
    }

    vecXd x0(i32 n) { return vecXd::Constant(n, -1.); }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 2 && n % 2 == 0);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

} // namespace sOPT