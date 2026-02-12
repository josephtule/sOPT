#pragma once

#include "sOPT/core/vecdefs.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cassert>

namespace sOPT {

// SPD quadratic:
//   f(x) = 0.5 * x^T A x - b^T x
//   grad = A x - b
//
// This generator creates A with controlled condition number by using a random
// orthonormal basis Q and a diagonal spectrum D with log-spaced eigenvalues.
struct QuadraticSPD {
    matXd A;
    vecXd b;

    QuadraticSPD() = default;

    QuadraticSPD(i32 n, f64 lambda_min, f64 lambda_max, u32 seed = 1) {
        assert(n >= 1);
        assert(lambda_min > 0.0);
        assert(lambda_max >= lambda_min);

        // Random matrix -> QR -> Q orthonormal
        eig::MatrixXd M(n, n);
        M.setRandom();
        eig::HouseholderQR<eig::MatrixXd> qr(M);
        eig::MatrixXd Q = qr.householderQ() * eig::MatrixXd::Identity(n, n);

        // log-spaced eigenvalues
        vecXd d(n);
        if (n == 1) {
            d(0) = lambda_max;
        } else {
            const f64 a = std::log(lambda_min);
            const f64 b = std::log(lambda_max);
            for (i32 i = 0; i < n; ++i) {
                const f64 t = f64(i) / f64(n - 1);
                d(i) = std::exp(a + t * (b - a));
            }
        }

        eig::MatrixXd D = d.asDiagonal();
        A = (Q * D * Q.transpose()).eval();

        // choose b so solution isn't trivial; random-ish but deterministic via
        // Eigen RNG (Eigen's setRandom uses internal RNG; seed isn't wired
        // here)
        this->b.resize(n);
        this->b.setRandom();
    }

    f64 func(ecref<vecXd> x) const { return 0.5 * x.dot(A * x) - b.dot(x); }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        g.resize(x.size());
        g.noalias() = A * x - b;
    }

    void hessian(ecref<vecXd>, eref<matXd> H) const { H = A; }

    bool check_x(ecref<vecXd> x) {
        const i32 n = static_cast<i32>(x.size());
        return (n >= 1);
    }
    void check_assert(ecref<vecXd> x) {
        assert(check_x(x));
    }
};

static vecXd x0_quadratic(i32 n) {
    vecXd x(n);
    x.setZero();
    return x;
}

} // namespace sOPT
