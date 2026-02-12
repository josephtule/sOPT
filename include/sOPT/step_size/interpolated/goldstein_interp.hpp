#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/detail/step_size_common.hpp"
#include "sOPT/step_size/step_attempt.hpp"

#include <algorithm>
#include <cmath>

namespace sOPT {

struct GoldsteinInterp {
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
        const f64 g0p = g0.dot(p);
        alpha = opt.ls.alpha0;
        if (!in_op(c, 0.0, 0.5)) return StepAttempt::line_search_failed;
        if (!finite_neg(g0p)) return StepAttempt::line_search_failed;
        if (!finite_pos(alpha)) return StepAttempt::line_search_failed;
        const f64 alpha_max = opt.ls.alpha_max;

        x_next.resize(x.size());
        f64 alo = 0.0;
        f64 ahi = inf<f64>;
        f64 flo = f0;
        f64 fhi = qNaN<f64>;
        bool has_lo = false;
        bool has_hi = false;
        i32 bracket_steps = 0;
        for (i32 k = 0; k < opt.ls.max_iters; k++) {
            x_next.noalias() = x + alpha * p;
            if (!oracle.try_func(x_next, f_next)) return StepAttempt::eval_failed;

            const f64 upper = f0 + c * alpha * g0p;
            const f64 lower = f0 + (1.0 - c) * alpha * g0p;

            if (in_cl(f_next, lower, upper)) return StepAttempt::accepted;

            if (f_next > upper) {
                ahi = alpha;
                fhi = f_next;
                has_hi = true;
            } else {
                alo = alpha;
                flo = f_next;
                has_lo = true;
            }

            if (has_hi && has_lo) {
                f64 cand = qNaN<f64>;
                if (bracket_steps == 0) {
                    // First step: quadratic from phi(0), phi'(0), phi(ahi).
                    cand = detail::quad_min_val_slope(0.0, f0, g0p, ahi, fhi);
                } else {
                    // Next steps: cubic from phi(0), phi'(0), phi(alo), phi(ahi).
                    cand = detail::cubic_min_val_slope(0.0, f0, g0p, alo, flo, ahi, fhi);
                }
                if (!std::isfinite(cand)) cand = 0.5 * (alo + ahi);
                alpha = detail::clamp_pad(cand, alo, ahi);
                ++bracket_steps;
                continue;
            }

            if (has_hi) {
                alpha = 0.5 * (alo + ahi);
                continue;
            }

            if (isfinite(alpha_max))
                alpha = std::min(alpha * 2.0, alpha_max);
            else
                alpha *= 2.0;
        }
        return StepAttempt::line_search_failed;
    }
};

} // namespace sOPT
