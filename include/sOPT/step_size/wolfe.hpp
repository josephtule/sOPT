#pragma once

#include "sOPT/core/options.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/step_attempt.hpp"

#include <algorithm>
#include <cmath>

namespace sOPT {

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
    auto phi = [&](f64 a, eref<vecXd>& xt, f64& ft) {
        xt.noalias() = x + a * p;
        if (!oracle.try_func(xt, ft)) return StepAttempt::eval_failed;
        return std::isfinite(ft) ? StepAttempt::accepted : StepAttempt::eval_failed;
    };
    auto dphi = [&](eref<vecXd>& xt, f64& dphi_val) {
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


        }
    };
}

} // namespace sOPT