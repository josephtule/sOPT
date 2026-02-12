#include "sOPT/bench/rosenbrock.hpp"
#include "sOPT/sOPT.hpp"
#include "sOPT/step_size/fixed_step.hpp"
#include "sOPT/step_size/goldstein.hpp"
#include "sOPT/step_size/wolfe.hpp"

#include <print>

using namespace sOPT;

int main() {
    Options opt;
    opt.term.max_iters = 10000;
    opt.term.grad_tol = 1e-10;
    opt.term.step_tol = 1e-12;
    opt.fd.eps = 1e-8;

    // line search options (reused)
    opt.ls.alpha_fixed = 1e-2;
    opt.ls.alpha0 = 1.0;
    opt.ls.alpha_max = 64.0;
    opt.ls.rho = 0.5;
    opt.ls.c1 = 1e-4;
    opt.ls.c2 = 0.9;
    opt.ls.try_full_step = false;
    opt.lbfgs.memory = 20;

    auto obj = RosenbrockChained{};
    const i32 n = 2;
    vecXd x0 = obj.x0(n);
    std::string solver_name = "Newton";
    auto solver = [&]<typename Step>(const Step& step) {
        return newton(obj, x0, opt, step);
    };

    {
        std::println(
            "{} (Fixed Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(FixedStep{});
        print_sOPT_results(res);
        std::println();
    }

    {
        std::println(
            "{} (Armijo Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(Armijo{});
        print_sOPT_results(res);
        std::println();
    }

    {
        std::println(
            "{} (Goldstein Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(Goldstein{});
        print_sOPT_results(res);
        std::println();
    }

    {
        std::println(
            "{} (Weak Wolfe Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(WolfeWeak{});
        print_sOPT_results(res);
        std::println();
    }

    {
        std::println(
            "{} (Strong Wolfe Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(WolfeStrong{});
        print_sOPT_results(res);
        std::println();
    }

    std::println("------------------------------------------------------------");
    opt.ls.try_full_step = true;

    {
        std::println(
            "{} (Fixed Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(FixedStep{});
        print_sOPT_results(res);
        std::println();
    }

    {
        std::println(
            "{} (Armijo Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(Armijo{});
        print_sOPT_results(res);
        std::println();
    }

    {
        std::println(
            "{} (Goldstein Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(Goldstein{});
        print_sOPT_results(res);
        std::println();
    }

    {
        std::println(
            "{} (Weak Wolfe Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(WolfeWeak{});
        print_sOPT_results(res);
        std::println();
    }

    {
        std::println(
            "{} (Strong Wolfe Step): Rosenbrock Chained Objective, n = {}",
            solver_name,
            n
        );
        auto res = solver(WolfeStrong{});
        print_sOPT_results(res);
        std::println();
    }

    return 0;
}
