#include "sOPT/algorithms/gradient_descent.hpp"
#include "sOPT/core/result.hpp"
#include "sOPT/sOPT.hpp"

#include <print>
using namespace sOPT;
void print_sOPT_results(Result& res) {
    std::println(
        "Solver exited in {} iterations with exit type: '{}'",
        res.iterations,
        to_string(res.status)
    );
    std::println(
        "Evaluations:\nFunction evals: {}\nGradient evals: {}\nHessian evals: "
        "{}",
        res.f_evals,
        res.g_evals,
        res.h_evals
    );
    std::println("gradient norm: {}", res.grad_norm);
    std::println("Optimal value x* = {}", res.x);
    std::println("With the optimal objective: J(x*) = {}", res.f);
}

struct RosenbrockAnalytic {
    f64 func(ecref<vecXd> x) const {
        const f64 X = x(0), Y = x(1);
        const f64 t1 = 1.0 - X;
        const f64 t2 = Y - X * X;
        return t1 * t1 + 100.0 * t2 * t2;
    }

    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        const f64 X = x(0), Y = x(1);
        g.resize(2);
        g(0) = -2.0 * (1.0 - X) - 400.0 * X * (Y - X * X);
        g(1) = 200.0 * (Y - X * X);
    }

    void hessian(ecref<vecXd> x, eref<matXd> H) const {
        const f64 X = x(0), Y = x(1);
        H.resize(2, 2);
        H(0, 0) = 2.0 - 400.0 * (Y - X * X) + 800.0 * X * X;
        H(0, 1) = -400.0 * X;
        H(1, 0) = -400.0 * X;
        H(1, 1) = 200.0;
    }
};

struct RosenbrockFDHess {
    f64 func(ecref<vecXd> x) const { return RosenbrockAnalytic{}.func(x); }
    void gradient(ecref<vecXd> x, eref<vecXd> g) const {
        RosenbrockAnalytic{}.gradient(x, g);
    }
    // no hessian()
};

struct RosenbrockFDGrad {
    f64 func(ecref<vecXd> x) const { return RosenbrockAnalytic{}.func(x); }
    // no gradient()
    // no hessian()
};

int main() {
    vecXd x0(2);
    x0 << -1.2, 1.0;

    Options opt;
    opt.term.max_iters = 20000;
    opt.term.grad_tol = 1e-8;
    opt.term.step_tol = 1e-12;

    // line search params
    opt.ls.alpha0 = 1.0;
    opt.ls.alpha_max = 64.0;
    opt.ls.rho = 0.5;
    opt.ls.c1 = 1e-4;
    opt.ls.c2 = 0.9;

    // fixed-step baseline
    opt.ls.alpha_fixed = 1e-3;

    // auto obj = RosenbrockAnalytic{};
    // Result res = gradient_descent(obj, x0, opt, Armijo{});
    // std::println("Analytic:");
    // print_sOPT_results(res);
    // std::println();

    auto obj2 = RosenbrockFDGrad{};
    Result res2 = gradient_descent(obj2, x0, opt, Armijo{});
    std::println("FD Gradient:");
    print_sOPT_results(res2);
    std::println();

    return 0;
}