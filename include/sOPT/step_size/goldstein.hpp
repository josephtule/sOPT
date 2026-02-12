#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/step_attempt.hpp"

#include <print>

namespace sOPT {
struct Goldstein {
    template <typename OracleT>
    StepAttempt operator()(
        OracleT& oracle,
        ecref<vecXd> x,
        f64 f0,
        ecref<vecXd> g0,
        ecref<vecXd> p,
        f64& alpha,
        vecXd& x_next,
        f64& f_next,
        const Options& opt
    ) const {
        const f64 c = opt.ls.c1;
        if (!in_op(c, 0.0, 0.5)) return StepAttempt::line_search_failed;

        const f64 g0p = g0.dot(p);
        if (!finite_neg(g0p)) return StepAttempt::line_search_failed;

        f64 alo = 0.0;
        f64 ahi = inf<f64>;
        alpha = opt.ls.alpha0;
        if (!finite_pos(alpha)) return StepAttempt::line_search_failed;

        x_next.resize(x.size());
        for (i32 k = 0; k < opt.ls.max_iters; k++) {
            x_next.noalias() = x + alpha * p;
            if (!oracle.try_func(x_next, f_next)) return StepAttempt::eval_failed;
            if (!isfinite(f_next)) return StepAttempt::eval_failed;

            const f64 upper = f0 + c * alpha * g0p;
            const f64 lower = f0 + (1.0 - c) * alpha * g0p;

            if (f_next > upper) {
                ahi = alpha;
                alpha = 0.5 * (alo + ahi); // bisection
                continue;
            }
            if (f_next < lower) {
                alo = alpha;
                if (isfinite(ahi))
                    alpha = 0.5 * (alo + ahi); // bisection
                else
                    alpha *= 2.0;
                continue;
            }

            return StepAttempt::accepted;
        }
        return StepAttempt::line_search_failed;
    }
};

} // namespace sOPT