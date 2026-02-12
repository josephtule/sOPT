#include "sOPT/algorithms/gradient_descent.hpp"
#include "sOPT/bench/rosenbrock.hpp" 
#include "sOPT/sOPT.hpp"

#include <print>

using namespace sOPT;
int main() {
    Options opt;
    opt.term.max_iters = 200000;
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

    auto obj = RosenbrockChained{};
    auto step = Armijo{};
    const i32 n = 2;
    vecXd x0 = obj.x0(n);

    std::println("Gradient Descent: Rosenbrock Chained Objective, n = {}", n);
    auto res = gradient_descent(obj, x0, opt, step);
    print_sOPT_results(res);
    std::println();

    return 0;
}
