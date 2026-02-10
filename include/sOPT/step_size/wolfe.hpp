#pragma once

#include "sOPT/core/options.hpp"
#include "sOPT/core/status.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/step_attempt.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace sOPT {

// ref: nocedal2006numerical pp.34 & pp.60-61
template <bool Strong, typename OracleT>
inline StepAttempt wolfe_impl(
    OracleT& oracle,
    ecref<vecXd> x,
    f64 f0,
    ecref<vecXd> g0,
    ecref<vecXd> p,
    f64& alpha,
    vecXd& x_next,
    f64& f_next,
    const Options& opt
) {
    const i32 n = static_cast<i32>(x.size());
    const f64 c1 = opt.ls.c1;
    const f64 c2 = opt.ls.c2;

    if (!(c1 > 0.0 && c1 < 1.0)) return StepAttempt::line_search_failed;
    if (!(c2 > c1 && c2 < 1.0)) return StepAttempt::line_search_failed;

    const f64 g0p = g0.dot(p);
    if (!(g0p < 0.0)) return StepAttempt::line_search_failed; // must be descent

    const f64 alpha_max = opt.ls.alpha_max;

    vecXd g_trial(n);
    vecXd xt_zoom(n);

    // phi and phi' in Nocedal
    auto phi = [&](f64 a, vecXd& xt, f64& ft) -> StepAttempt {
        xt.noalias() = x + a * p;
        if (!oracle.try_func(xt, ft)) return StepAttempt::eval_failed;
        return std::isfinite(ft) ? StepAttempt::accepted : StepAttempt::eval_failed;
    };
    auto dphi = [&](ecref<vecXd> xt, f64& dphi_val) -> StepAttempt {
        if (!oracle.try_gradient(xt, g_trial)) return StepAttempt::eval_failed;
        if (!g_trial.allFinite()) return StepAttempt::eval_failed;
        dphi_val = g_trial.dot(p);
        return std::isfinite(dphi_val) ? StepAttempt::accepted : StepAttempt::eval_failed;
    };
    auto wolfe_ok = [&](f64 a, f64 ft, f64 dft) -> bool {
        if (!(ft <= f0 + c1 * a * g0p)) return false;
        if constexpr (Strong) {
            return std::abs(dft) <= (-c2 * g0p);
        } else {
            return dft >= c2 * g0p;
        }
    };
    auto zoom = [&](f64 alo, f64 ahi, f64 flo) -> StepAttempt {
        for (i32 k = 0; k < opt.ls.max_iters; k++) {
            const f64 aj = 0.5 * (alo + ahi); // bisection

            const StepAttempt phi_status = phi(aj, xt_zoom, f_next);
            if (phi_status != StepAttempt::accepted) return phi_status;

            if ((f_next > f0 + c1 * aj * g0p) || (f_next >= flo)) {
                ahi = aj;
            } else {
                f64 dphi_j = 0.0;
                const StepAttempt dphi_status = dphi(xt_zoom, dphi_j);
                if (dphi_status != StepAttempt::accepted) return dphi_status;

                if (wolfe_ok(aj, f_next, dphi_j)) {
                    alpha = aj;
                    x_next.swap(xt_zoom);
                    return StepAttempt::accepted;
                }

                if (dphi_j * (ahi - alo) >= 0) {
                    ahi = alo;
                }

                alo = aj;
                flo = f_next;
            }
        }
        return StepAttempt::line_search_failed;
    };

    f64 a_prev = 0.0;
    f64 f_prev = f0;
    alpha = opt.ls.alpha0;
    if (!(alpha > 0.0) || !std::isfinite(alpha)) // TODO: maybe remove this
        return StepAttempt::line_search_failed;

    x_next.resize(x.size()); // TODO: maybe remove this
    for (i32 i = 0; i < opt.ls.max_iters; i++) {
        const StepAttempt phi_status = phi(alpha, x_next, f_next);
        if (phi_status != StepAttempt::accepted) return phi_status;

        if ((f_next > f0 + c1 * alpha * g0p) || (i > 0 && f_next >= f_prev)) {
            return zoom(a_prev, alpha, f_prev);
        }

        f64 dphi_a = 0.0;
        const StepAttempt dphi_status = dphi(x_next, dphi_a);
        if (dphi_status != StepAttempt::accepted) return dphi_status;

        if (wolfe_ok(alpha, f_next, dphi_a)) {
            return StepAttempt::accepted;
        }
        if (dphi_a >= 0.0) {
            return zoom(alpha, a_prev, f_next);
        }

        a_prev = alpha;
        f_prev = f_next;
        alpha = std::min(2.0 * alpha, alpha_max);
    }
    return StepAttempt::line_search_failed;
}

struct WolfeWeak {
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
        return wolfe_impl<false>(oracle, x, f0, g0, p, alpha, x_next, f_next, opt);
    }
};

struct WolfeStrong {
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
        return wolfe_impl<true>(oracle, x, f0, g0, p, alpha, x_next, f_next, opt);
    }
};

} // namespace sOPT