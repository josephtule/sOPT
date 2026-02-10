#pragma once

#include "sOPT/core/options.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/step_attempt.hpp"

#include <cmath>
#include <type_traits>
#include <utility>

namespace sOPT {

// never pass TryFull<step_strategy>{} explicitly, control with
// opt.ls.try_full_step = true/false
template <typename InnerStep>
struct TryFull {
    InnerStep inner{};

    TryFull() = default;
    explicit TryFull(InnerStep s) : inner(std::move(s)) {}

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
        const f64 g0p = g0.dot(p);

        // only try alpha=1 if p is a descent direction
        if (g0p < 0.0) {
            alpha = 1.0;
            x_next.resize(x.size());
            x_next.noalias() = x + alpha * p;
            if (!oracle.try_func(x_next, f_next)) return StepAttempt::eval_failed;

            if (std::isfinite(f_next) && (f_next <= f0 + opt.ls.c1 * alpha * g0p)) {
                return StepAttempt::accepted;
            }
        }

        const auto inner_result = inner(oracle, x, f0, g0, p, alpha, x_next, f_next, opt);

        if constexpr (std::is_same_v<std::decay_t<decltype(inner_result)>, StepAttempt>) {
            return inner_result;
        } else {
            return inner_result ? StepAttempt::accepted : StepAttempt::line_search_failed;
        }
    }
};

} // namespace sOPT
