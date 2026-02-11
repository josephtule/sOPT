#pragma once

#include "sOPT/algorithms/detail/solver_common.hpp"
#include "sOPT/core/callback.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/result.hpp"
#include "sOPT/core/typedefs.hpp"
#include "sOPT/core/vecdefs.hpp"
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
// s_k^T (inverse hessian approximation)
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
    vecXd p(n);
    vecXd s(n);  // displacement vector
    vecXd y(n);  // change of gradients vector
    vecXd By(n); //
    vecXd x_next(n);
    f64 f = 0.0;
    f64 f_next = 0.0;
    f64 alpha = 0.0;
    f64 last_ys = qNaN<f64>;
    f64 last_ys_cos = qNaN<f64>;
    matXd B(n, n);
    detail::TerminationScales term_scales;
    B.setIdentity();

    if (auto st = detail::init_common( // sets x, f, and g in a first pass
            oracle,
            opt,
            x0,
            res,
            g,
            f,
            on_iter,
            should_stop,
            &term_scales
        )) {
        res.status = *st;
        detail::finalize_common(res, oracle, f, g.norm());
        return res;
    }

    for (i32 k = 0; k < opt.term.max_iters; k++) {
        const f64 gnorm = g.norm();

        if (auto st = detail::pre_step_checks(f, gnorm, opt, &term_scales)) {
            res.status = *st;
            break;
        }

        p.noalias() = -(B * g);
        if (g.dot(p) >= 0.0) { // ensure p is descending
            B.setIdentity();
            p.noalias() = -g;
        }

        IterDiagnostics diag;
        if (opt.diag.enabled) {
            if (opt.diag.record_directional_derivative) diag.gTp = g.dot(p);
            if (opt.diag.record_qn_curvature) {
                diag.ys = last_ys;
                diag.ys_cos = last_ys_cos;
            }
        }

        const detail::StepStatus step_status = detail::run_step(
            oracle,
            opt,
            step_strategy,
            res.x,
            f,
            g,
            p,
            alpha,
            x_next,
            f_next
        );
        if (step_status != detail::StepStatus::accepted) {
            res.status = detail::to_status(step_status);
            break;
        }

        const f64 step_norm = (x_next - res.x).norm();
        const bool step_converged
            = detail::is_step_converged(step_norm, opt, &term_scales);
        const f64 f_prev = f;

        if (!step_converged) {
            // save secant pair data before x/g are updated
            s.noalias() = x_next - res.x;
            g_prev = g;
        }

        if (auto st = detail::post_accept_with_step_status(
                oracle,
                opt,
                res,
                res.x,
                f,
                g,
                x_next,
                f_next,
                f_prev,
                alpha,
                step_norm,
                diag,
                on_iter,
                should_stop,
                &term_scales
            )) {
            res.status = *st;
            break;
        }

        y.noalias() = g - g_prev;

        // BFGS curvature condition y^T * s = s^T * y > 0
        const f64 ys = y.dot(s);
        if (opt.diag.enabled && opt.diag.record_qn_curvature) {
            const f64 denom = y.norm() * s.norm();
            const f64 ys_cos
                = (isfinite(denom) && denom > 0.0) ? (ys / denom) : qNaN<f64>;
            last_ys = ys;
            last_ys_cos = ys_cos;
            if (res.trace && !res.trace->ys.empty()) {
                res.trace->ys.back() = ys;
                res.trace->ys_cos.back() = ys_cos;
            }
        }

        if (isfinite(ys) && ys > tol_max * s.squaredNorm()) {
            const f64 rho = 1.0 / ys;
            By.noalias() = B.selfadjointView<eLo>() * y;
            const f64 yBy = y.dot(By);
            const f64 coeff = (1.0 + rho * yBy) * rho;

            // inverse-BFGS update in symmetric form
            B.selfadjointView<eLo>().rankUpdate(s, coeff); // B += coeff * s s^T
            B.noalias() -= rho * (By * s.transpose());
            B.noalias() -= rho * (s * By.transpose());
            B = B.selfadjointView<eLo>(); // re-symmetrize
        } else {
            // skip update if curvature is bad (keeps B SPD-ish)
            // strong wolfe should usually prevent this
        }

    } // end iterations

    detail::finalize_common(res, oracle, f, g.norm());
    return res;
}

// overload default Strong Wolfe
template <typename Obj>
Result bfgs(
    const Obj& obj,
    ecref<vecXd> x0,
    const Options& opt,
    const IterCallback& on_iter = {},
    const StopCallback& should_stop = {}
) {
    return bfgs(obj, x0, opt, WolfeStrong{}, on_iter, should_stop);
}

} // namespace sOPT