#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/step_attempt.hpp"

#include <algorithm>
#include <cmath>

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
        if (!(alpha > 0.0) || !isfinite(alpha)) return StepAttempt::line_search_failed;
        if (!(c1 > 0.0 && c1 < 1.0)) return StepAttempt::line_search_failed;
        if (!(rho > 0.0 && rho < 1.0)) return StepAttempt::line_search_failed;

        const f64 gTp = g.dot(p);
        if (!(gTp < 0.0)) return StepAttempt::line_search_failed;

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
                // Quadratic model minimizing interpolation through phi(0), phi'(0),
                // phi(alpha)
                const f64 denom = 2.0 * (f_next - fx - gTp * alpha);
                if (finite_pos(denom)) {
                    const f64 cand = -(gTp * alpha * alpha) / denom;
                    if (finite_pos(cand)) alpha_next = cand;
                }
            } else {
                // Cubic model using previous rejected point + current rejected point.
                const f64 a0 = alpha_prev;
                const f64 a1 = alpha;
                const f64 d0 = f_prev - fx - gTp * a0;
                const f64 d1 = f_next - fx - gTp * a1;
                const f64 denom = a0 - a1;

                if (finite_nonzero(denom)) {
                    const f64 A = (d0 / (a0 * a0) - d1 / (a1 * a1)) / denom;
                    const f64 B = (-a1 * d0 / (a0 * a0) + a0 * d1 / (a1 * a1)) / denom;

                    if (isfinite(A) && isfinite(B)) {
                        if (std::abs(A) < 1e-16) {
                            if (B > 0.0) {
                                const f64 cand = -gTp / (2.0 * B);
                                if (finite_pos(cand)) alpha_next = cand;
                            }
                        } else {
                            const f64 disc = B * B - 3.0 * A * gTp;
                            if (finite_nonneg(disc)) {
                                const f64 cand = (-B + std::sqrt(disc)) / (3.0 * A);
                                if (finite_pos(cand)) alpha_next = cand;
                            }
                        }
                    }
                }
            }

            if (!(alpha_next > 0.0) || !isfinite(alpha_next)) {
                alpha_next = rho * alpha;
            }

            // keep strict backtracking and avoid tiny interpolation steps
            const f64 lo = std::min(0.1 * alpha, rho * alpha);
            const f64 hi = std::max(0.1 * alpha, rho * alpha);
            alpha_next = std::clamp(alpha_next, lo, hi);

            alpha_prev = alpha;
            f_prev = f_next;
            has_prev = true;
            alpha = alpha_next;
        }

        return StepAttempt::line_search_failed;
    }
};

// Backward-compatible alias; prefer ArmijoInterp.
using BacktrackingInterp = ArmijoInterp;

} // namespace sOPT
