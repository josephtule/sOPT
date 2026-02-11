#pragma once

#include "sOPT/algorithms/detail/solver_common.hpp"
#include "sOPT/core/callback.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/result.hpp"
#include "sOPT/problem/oracle.hpp"
#include "sOPT/step_size/wolfe.hpp"

#include <cmath>
#include <limits>

namespace sOPT {
// BFGS (inverse-Hessian form)
//
// \begin{aligned}
// g_k &= \nabla f(x_k) \\
// p_k &= -B_k g_k \\
// x_{k+1} &= x_k + \alpha_k p_k \\
// s_k &= x_{k+1} - x_k \\
// y_k &= g_{k+1} - g_k \\
// \rho_k &= \frac{1}{y_k^T s_k} \\
// B_{k+1} &= (I - \rho_k s_k y_k^T)\, B_k\, (I - \rho_k y_k s_k^T) + \rho_k s_k
// s_k^T
// \end{aligned}
//
// Strong Wolfe is typically used so that $y_k^T s_k > 0$, helping keep $B_k$
// SPD.
template <typename Obj, typename StepStrategy>
Result bfgs(
    const Obj& obj,
    ecref<vecXd> x0,
    const Options& opt,
    const StepStrategy& step_strategy,
    const IterCallback& on_iter = {},
    const StopCallback& should_stop = {}
) {
    Oracle<Obj> oracle(obj, opt);
    Result res;
    const i32 n = static_cast<i32>(x0.size());

    vecXd g(n);
    vecXd g_prev(n);
}

} // namespace sOPT