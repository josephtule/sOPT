#include "sOPT/objectives/rosenbrock.hpp"
#include "sOPT/sOPT.hpp"

#include <print>
using namespace sOPT;

int main() {

    Options opt;
    opt.term.max_iters = 20000;
    opt.term.grad_tol = 1e-10;
    opt.term.step_tol = 1e-12;

    // line search params
    opt.ls.alpha0 = 1.0;
    opt.ls.alpha_max = 64.0;
    opt.ls.rho = 0.5;
    opt.ls.c1 = 1e-4;
    opt.ls.c2 = 0.9;
    opt.ls.try_full_step = true;

    // fixed-step baseline
    opt.ls.alpha_fixed = 1e-3;

    opt.lbfgs.memory = 20;

    // expect to converge via gradient norm
    auto obj = RosenbrockChained{};

    vecXd x0 = obj.x0(2);

    Result res = dfp(obj, x0, opt, WolfeStrong{});
    print_sOPT_results(res);
    std::println();

    // // expect to exceed f_evals limit
    // auto obj2 = RosenbrockFDGrad{};
    // Result res2 = gradient_descent(obj2, x0, opt, Armijo{});
    // std::println("FD Gradient:");
    // print_sOPT_results(res2);
    // std::println();

    return 0;
}