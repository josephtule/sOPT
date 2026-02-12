#pragma once

#include "sOPT/core/options.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/step_attempt.hpp"

namespace sOPT {

struct FixedStep {
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
        (void)g;
        (void)fx;

        alpha = opt.ls.alpha_fixed;
        x_next.resize(x.size());
        x_next.noalias() = x + alpha * p;

        if (!oracle.try_func(x_next, f_next)) return StepAttempt::eval_failed;

        return isfinite(f_next) ? StepAttempt::accepted : StepAttempt::eval_failed;
    }
};

} // namespace sOPT
