#include "sOPT/objectives/rosenbrock.hpp"
#include "sOPT/sOPT.hpp"

#include <print>

using namespace sOPT;
int main() {
    Options opt;
    opt.term.max_iters = 200;
    opt.term.grad_tol = 1e-10;
    opt.term.step_tol = 1e-12;

    // FD eps used by fd_hessian_central
    opt.fd.eps = 1e-6;

    // line search options (reused)
    opt.ls.alpha0 = 1.0;
    opt.ls.alpha_max = 64.0;
    opt.ls.rho = 0.5;
    opt.ls.c1 = 1e-4;
    opt.ls.c2 = 0.9;
    opt.ls.try_full_step = false;

    auto obj = RosenbrockChained{};
    auto step = Armijo{};
    const i32 n = 2;
    vecXd x0 = obj.x0(n);

    std::println("Newton: Rosenbrock Chained Objective, n = {}", n);
    auto res = newton(obj, x0, opt, step);
    print_sOPT_results(res);
    std::println();

    return 0;
}
