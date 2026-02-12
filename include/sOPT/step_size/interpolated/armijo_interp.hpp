#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/detail/step_size_common.hpp"
#include "sOPT/step_size/step_attempt.hpp"

#include <algorithm>

namespace sOPT {

// Armijo backtracking with quadratic/cubic interpolation.
// Uses quadratic model on first rejection, then cubic model.
struct ArmijoInterp {
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
        if (!finite_pos(alpha)) return StepAttempt::line_search_failed;
        if (!in_op(c1, 0.0, 1.0)) return StepAttempt::line_search_failed;
        if (!in_op(rho, 0.0, 1.0)) return StepAttempt::line_search_failed;

        const f64 gTp = g.dot(p);
        if (!finite_neg(gTp)) return StepAttempt::line_search_failed;

        x_next.resize(x.size());
        bool has_prev = false;
        f64 alpha_prev = 0.0;
        f64 f_prev = fx;
        for (i32 k = 0; k < opt.ls.max_iters; ++k) {
            x_next.noalias() = x + alpha * p;
            if (!oracle.try_func(x_next, f_next)) return StepAttempt::eval_failed;
            if (f_next <= fx + c1 * alpha * gTp) return StepAttempt::accepted;

            f64 alpha_next = rho * alpha; // bisection/geometric fallback

            if (!has_prev) {
                // Quadratic interpolation using phi(0), phi'(0), phi(alpha)
                const f64 cand = detail::quad_min_val_slope(0.0, fx, gTp, alpha, f_next);
                if (finite_pos(cand)) alpha_next = cand;
            } else {
                // Cubic interpolation using phi(0), phi'(0), and two rejected points.
                const f64 cand = detail::cubic_min_val_slope(
                    0.0,
                    fx,
                    gTp,
                    alpha_prev,
                    f_prev,
                    alpha,
                    f_next
                );
                if (finite_pos(cand)) alpha_next = cand;
            }

            if (!finite_pos(alpha_next)) alpha_next = rho * alpha;

            // keep strict backtracking and avoid tiny interpolation steps
            const f64 lo = std::min(0.1 * alpha, rho * alpha);
            const f64 hi = std::max(0.1 * alpha, rho * alpha);
            alpha_next = detail::clamp_pad(alpha_next, lo, hi, 0.0);
            alpha_prev = alpha;
            f_prev = f_next;
            has_prev = true;
            alpha = alpha_next;
        }

        return StepAttempt::line_search_failed;
    }
};

} // namespace sOPT
