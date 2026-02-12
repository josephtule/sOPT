#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/vecdefs.hpp"

#include <cassert>

namespace sOPT {

struct NazarethMod {
    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());

        f64 f = 0.;
        const i32 d = n / 2;

        for (i32 ii = 1; ii <= n; ii++) {
            f64 val = f64(n) + f64(ii);

            const i32 lo = std::max(1, ii - 2);
            const i32 hi = std::min(n, ii + 2);
            for (i32 jj = lo; jj <= hi; jj++) {
                const i32 j = jj - 1;
                const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                const f64 b = f64(ii + jj) * 0.1;
                const f64 xj = x(j);
                val -= a * std::sin(xj) + b * std::cos(xj);
            }

            if (d > 2) {
                if (ii + d <= n) {
                    const i32 jj = ii + d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    val -= a * std::sin(xj) + b * std::cos(xj);
                }
                if (ii - d >= 1) {
                    const i32 jj = ii - d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    val -= a * std::sin(xj) + b * std::cos(xj);
                }
            }

            f += pow_Ti(val, 2);
        }

        return f / f64(n);
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());

        g.setZero();
        vecXd s(n);

        const i32 d = n / 2;

        for (i32 ii = 1; ii <= n; ii++) {
            const i32 i = ii - 1;

            f64 si = f64(n) + f64(ii);

            const i32 lo = std::max(1, ii - 2);
            const i32 hi = std::min(n, ii + 2);
            for (i32 jj = lo; jj <= hi; jj++) {
                const i32 j = jj - 1;
                const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                const f64 b = f64(ii + jj) * 0.1;
                const f64 xj = x(j);
                si -= a * std::sin(xj) + b * std::cos(xj);
            }

            if (d > 2) {
                if (ii + d <= n) {
                    const i32 jj = ii + d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    si -= a * std::sin(xj) + b * std::cos(xj);
                }
                if (ii - d >= 1) {
                    const i32 jj = ii - d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    si -= a * std::sin(xj) + b * std::cos(xj);
                }
            }

            s(i) = si;
        }

        for (i32 ii = 1; ii <= n; ii++) {
            const i32 i = ii - 1;
            const f64 si = s(i);

            const i32 lo = std::max(1, ii - 2);
            const i32 hi = std::min(n, ii + 2);
            for (i32 jj = lo; jj <= hi; jj++) {
                const i32 j = jj - 1;
                const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                const f64 b = f64(ii + jj) * 0.1;
                const f64 xj = x(j);
                g(j) += -si * (a * std::cos(xj) - b * std::sin(xj));
            }

            if (d > 2) {
                if (ii + d <= n) {
                    const i32 jj = ii + d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    g(j) += -si * (a * std::cos(xj) - b * std::sin(xj));
                }
                if (ii - d >= 1) {
                    const i32 jj = ii - d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    g(j) += -si * (a * std::cos(xj) - b * std::sin(xj));
                }
            }
        }

        g *= 2. / f64(n);
    }

    vecXd x0(i32 n) { return vecXd::Constant(n, 1. / f64(n)); }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 2 && n % 2 == 0);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

// -----------------------------------------------------------------------------

struct NazarethModAlt {
    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());

        f64 f = 0.;
        const i32 d = n / 2;

        for (i32 ii = 1; ii <= n; ii++) {
            // const int i = ii - 1;
            // const f64 xi = x(i)
            f += f64(ii) * (1. - std::cos(x(ii - 1)));

            const i32 lo = std::max(1, ii - 2);
            const i32 hi = std::min(n, ii + 2);

            for (i32 jj = lo; jj <= hi; jj++) {
                const i32 j = jj - 1;
                const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                const f64 b = f64(ii + jj) * 0.1;
                const f64 xj = x(j);
                f += a * std::sin(xj) + b * std::cos(xj);
            }

            if (d > 2) {
                if (ii + d <= n) {
                    const i32 jj = ii + d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    f += a * std::sin(xj) + b * std::cos(xj);
                }
                if (ii - d >= 1) {
                    const i32 jj = ii - d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    f += a * std::sin(xj) + b * std::cos(xj);
                }
            }
        }

        return f / f64(n);
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());

        g.setZero();

        const i32 d = n / 2;
        for (i32 ii = 1; ii <= n; ii++) {
            const i32 i = ii - 1;

            g(i) += f64(ii) * std::sin(x(i));

            const i32 lo = std::max(1, ii - 2);
            const i32 hi = std::min(n, ii + 2);
            for (i32 jj = lo; jj <= hi; jj++) {
                const i32 j = jj - 1;
                const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                const f64 b = f64(ii + jj) * 0.1;
                const f64 xj = x(j);
                g(j) += (a * std::cos(xj) - b * std::sin(xj));
            }

            if (d > 2) {
                if (ii + d <= n) {
                    const i32 jj = ii + d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    g(j) += (a * std::cos(xj) - b * std::sin(xj));
                }
                if (ii - d >= 1) {
                    const i32 jj = ii - d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 xj = x(j);
                    g(j) += (a * std::cos(xj) - b * std::sin(xj));
                }
            }
        }

        g *= 1. / f64(n);
    }

    vecXd x0(i32 n) { return vecXd::Constant(n, 1. / f64(n)); }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 2 && n % 2 == 0);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

// -----------------------------------------------------------------------------

struct TointTrig {
    f64 func(ecref<vecXd> x) const {
        const i32 n = static_cast<i32>(x.size());

        f64 f = 0.;
        const i32 d = n / 2;

        for (i32 ii = 1; ii <= n; ii++) {
            const f64 xi = x(ii - 1);
            const f64 ci = 1. + f64(ii) * 0.1;

            const i32 lo = std::max(1, ii - 2);
            const i32 hi = std::min(n, ii + 2);
            for (i32 jj = lo; jj <= hi; jj++) {
                const i32 j = jj - 1;
                const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                const f64 b = f64(ii + jj) * 0.1;
                const f64 cj = 1. + f64(jj) * 0.1;
                const f64 xj = x(j);
                f += a * std::sin(b + ci * xi + cj * xj);
            }

            if (d > 2) {
                if (ii + d <= n) {
                    const i32 jj = ii + d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 cj = 1. + f64(jj) * 0.1;
                    const f64 xj = x(j);
                    f += a * std::sin(b + ci * xi + cj * xj);
                }
                if (ii - d >= 1) {
                    const i32 jj = ii - d;
                    const i32 j = jj - 1;
                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;
                    const f64 cj = 1. + f64(jj) * 0.1;
                    const f64 xj = x(j);
                    f += a * std::sin(b + ci * xi + cj * xj);
                }
            }
        }

        return f / f64(n);
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const i32 n = static_cast<i32>(x.size());
        g.setZero();

        const i32 d = n / 2;

        for (i32 ii = 1; ii <= n; ++ii) {
            const i32 i = ii - 1;
            const f64 xi = x(i);
            const f64 ci = 1. + f64(ii) * 0.1;

            const i32 lo = std::max(1, ii - 2);
            const i32 hi = std::min(n, ii + 2);

            for (i32 jj = lo; jj <= hi; ++jj) {
                const i32 j = jj - 1;
                const f64 xj = x(j);
                const f64 cj = 1. + f64(jj) * 0.1;

                const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                const f64 b = f64(ii + jj) * 0.1;

                const f64 arg = b + ci * xi + cj * xj;
                const f64 c = std::cos(arg);

                g(i) += a * ci * c;
                g(j) += a * cj * c;
            }

            if (d > 2) {
                if (ii + d <= n) {
                    const i32 jj = ii + d;
                    const i32 j = jj - 1;
                    const f64 xj = x(j);
                    const f64 cj = 1. + f64(jj) * 0.1;

                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;

                    const f64 arg = b + ci * xi + cj * xj;
                    const f64 c = std::cos(arg);

                    g(i) += a * ci * c;
                    g(j) += a * cj * c;
                }
                if (ii - d >= 1) {
                    const i32 jj = ii - d;
                    const i32 j = jj - 1;
                    const f64 xj = x(j);
                    const f64 cj = 1. + f64(jj) * 0.1;

                    const f64 a = 5. * (1. + f64(ii % 5) + f64(jj % 5));
                    const f64 b = f64(ii + jj) * 0.1;

                    const f64 arg = b + ci * xi + cj * xj;
                    const f64 c = std::cos(arg);

                    g(i) += a * ci * c;
                    g(j) += a * cj * c;
                }
            }
        }

        g *= 1. / f64(n);
    }

    vecXd x0(i32 n) { return vecXd::Constant(n, 1.); }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 2 && n % 2 == 0);
    }
    void check_assert(ecref<vecXd> x) { assert(check_x(x)); }
};

} // namespace sOPT
