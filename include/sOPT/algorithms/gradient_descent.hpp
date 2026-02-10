#pragma once

#include "sOPT/core/callback.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/result.hpp"
#include "sOPT/core/typedefs.hpp"
#include "sOPT/core/vecdefs.hpp"
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
    ecref<vecXd> x,
    const Options& opt,
    const StepStrategy step_strategy,
    const IterCallback& on_iter = {},
    const StopCallback& should_stop = {}
) {
    Oracle<obj> oracle(obj, opt);
    Result res;

    const i32 n = static_cast<i32>(x.size());
    vecXd x_next = x;
    f64 f = 0.0;
    f64 f_next = 0.0;
    vecXd g(n); // gradient
    vecXd p(n); // descent direction

    detail::TerminationScales term_scales;

    // iteration loop
    for (i32 k = 0; k < opt.term.max_iters; k++) {
        // init diagnostics
        IterDiagnostics diag;
        if (opt.diag.enabled && opt.diag.record_directional_derivative) {
            diag.gTp = g.dot(p);
        }

        const f64 g_norm = g.norm();

        // check for invalid solutions
        if (auto st = detail::pre_step_checks(f, gnorm, opt, &term_scales)) {
            res.status = *st;
            break;
        }

        // descent direction
        p.noalias() = -g;
    }
}

} // namespace sOPT