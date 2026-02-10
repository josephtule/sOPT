#pragma once

#include "sOPT/algorithms/detail/solver_common.hpp"
#include "sOPT/core/callback.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/result.hpp"
#include "sOPT/core/typedefs.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/problem/oracle.hpp"
#include "sOPT/step_size/armijo.hpp"

namespace sOPT {
// Gradient Descent
//
// \begin{aligned}
// g_k &= \nabla f(x_k) \\
// p_k &= -g_k \\
// x_{k+1} &= x_k + \alpha_k p_k
// \end{aligned}
//
// where $\alpha_k > 0$ is chosen by a step-size rule
template <typename Obj, typename StepStrategy>
Result gradient_descent(
    const Obj& obj,
    ecref<vecXd> x0,
    const Options& opt,
    const StepStrategy step_strategy,
    const IterCallback& on_iter = {},
    const StopCallback& should_stop = {}
) {
    Oracle<Obj> oracle(obj, opt);
    Result res;

    const i32 n = static_cast<i32>(x0.size());
    vecXd x_next = x0;
    f64 f = 0.0;
    f64 f_next = 0.0;
    f64 alpha = 0.0;
    vecXd g(n); // gradient
    vecXd p(n); // descent direction
    detail::TerminationScales term_scales;

    // check if early stop
    if (auto st = detail::init_common(
            oracle,
            opt,
            x0,
            res,
            g,
            f,
            on_iter,
            should_stop,
            &term_scales
        )) {
        res.status = *st;
        detail::finalize_common(res, oracle, f, g.norm());
        return res;
    }

    // iteration loop
    for (i32 k = 0; k < opt.term.max_iters; k++) {
        const f64 gnorm = g.norm();

        // check for invalid solutions
        if (auto st = detail::pre_step_checks(f, gnorm, opt, &term_scales)) {
            res.status = *st;
            break;
        }

        // init diagnostics
        p.noalias() = -g; // descent direction
        IterDiagnostics diag;
        if (opt.diag.enabled && opt.diag.record_directional_derivative) {
            diag.gTp = g.dot(p);
        }

        const detail::StepStatus step_status = detail::run_step(
            oracle,
            opt,
            step_strategy,
            res.x,
            f,
            g,
            p,
            alpha,
            x_next,
            f_next
        );

        if (step_status != detail::StepStatus::accepted) {
            res.status = detail::to_status(step_status);
            break;
        }

        const f64 f_prev = f;
        const f64 step_norm = (x_next - res.x).norm();
        if (auto st = detail::post_accept_with_step_status(
                oracle,
                opt,
                res,
                res.x,
                f,
                g,
                x_next,
                f_next,
                f_prev,
                alpha,
                step_norm,
                diag,
                on_iter,
                should_stop,
                &term_scales
            )) {
            res.status = *st;
            break;
        }
    }

    // final bookkeeping
    detail::finalize_common(res, oracle, f, g.norm());

    return res;
}

// overload (default to Armijo)
template <typename Obj>
Result gradient_descent(
    const Obj& obj,
    ecref<vecXd> x0,
    const Options& opt,
    const IterCallback& on_iter = {},
    const StopCallback& should_stop = {}
) {
    return gradient_descent(obj, x0, opt, Armijo{}, on_iter, should_stop);
}

} // namespace sOPT