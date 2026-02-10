#pragma once

#include "sOPT/core/options.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/step_attempt.hpp"

#include <cmath>

namespace sOPT {

// Armijo/Wolfe-Sufficient Decrease Condition:
// f(x_k + \alpha p_k) \leq f(x_k) + c_1 \alpha \nabla f_k^T p_k
// where p_k is the descent direction,\nabla f_k^T is the gradient at x_k
struct Armijo {
    template <typename OracleT>
    StepAttempt operator()(
        OracleT& oracle,
        ecref<vecXd> x,
        f64 fx,
        ecref<vecXd> g,
        ecref<vecXd> p,
        f64& alpha,
        vecXd& x_next,
        f64& f_next,
        const Options& opt
    ) const {
        const f64 c1 = opt.ls.c1;
        const f64 rho = opt.ls.rho;
        alpha = opt.ls.alpha0;
        // TODO: remove these checks, checked in options validation already
        if (!(alpha > 0.0) || !std::isfinite(alpha))
            return StepAttempt::line_search_failed;
        if (!(rho > 0.0 && rho < 1.0)) return StepAttempt::line_search_failed;
        if (!(c1 > 0.0 && c1 < 1.0)) return StepAttempt::line_search_failed;

        const f64 gTp = g.dot(p);
        if (!(gTp < 0.0)) return StepAttempt::line_search_failed; // require descent

        x_next.resize(x.size());
        for (i32 k = 0; k < opt.ls.max_iters; k++) {
            x_next.noalias() = x + alpha * p; // candidate
            if (!oracle.try_func(x_next, f_next)) return StepAttempt::eval_failed;
            if (std::isfinite(f_next) && (f_next <= fx + c1 * alpha * gTp)) {
                return StepAttempt::accepted;
            }
            alpha *= rho;
            if (!(alpha > 0.0) || !std::isfinite(alpha))
                return StepAttempt::line_search_failed;
        }

        return StepAttempt::line_search_failed;
    }
};

} // namespace sOPT
