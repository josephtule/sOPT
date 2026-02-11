#pragma once

#include "sOPT/algorithms/detail/solver_common.hpp"
#include "sOPT/core/callback.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/result.hpp"
#include "sOPT/core/status.hpp"
#include "sOPT/problem/oracle.hpp"
#include "sOPT/step_size/wolfe.hpp"

#include <algorithm>
#include <deque>
#include <limits>

namespace sOPT {
// Limited-memory BFGS (L-BFGS)
//
// \begin{aligned}
// g_k &= \nabla f(x_k) \\
// p_k &= -H_k g_k \\
// x_{k+1} &= x_k + \alpha_k p_k \\
// s_k &= x_{k+1} - x_k \\
// y_k &= g_{k+1} - g_k \\
// \rho_k &= \frac{1}{y_k^T s_k}
// \end{aligned}
//
// The inverse-Hessian $H_k$ is not formed implicitly by storing the last $m$
// pairs $\{(s_i, y_i)\}$ and compute $p_k$ via the two-loop recursion:
//
// \begin{aligned}
// q &\leftarrow g_k \\
// \alpha_i &= \rho_i s_i^T q,\quad q \leftarrow q - \alpha_i y_i \qquad (i =
// k-1,\dots,k-m) \\
// r &\leftarrow \gamma_k q \quad \text{(often } \gamma_k = \frac{s_{k-1}^T
// y_{k-1}}{y_{k-1}^T y_{k-1}} \text{)} \\
// \beta_i &= \rho_i y_i^T r,\quad r \leftarrow r + s_i(\alpha_i - \beta_i)
// \qquad (i = k-m,\dots,k-1) \\ p_k &\leftarrow -r
// \end{aligned}
//
// Strong Wolfe line search is commonly used to help ensure $y_k^T s_k > 0$.
template <typename Obj, typename StepStrategy>
Result lbfgs(
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
    const i32 m = opt.lbfgs.memory;
    vecXd g(n), g_prev(n);
    vecXd p(n), q(n), r(n); // temp gradient-like vectors for first and second loops
    vecXd s(n), y(n);       // see bfgs.hpp

    vecXd x_next(n);
    f64 f = 0.0;
    f64 f_next = 0.0;
    f64 alpha = 0.0;
    f64 last_ys = qNaN<f64>;
    f64 last_ys_cos = qNaN<f64>;
    detail::TerminationScales term_scales;

    // history (most recent at back)
    std::deque<vecXd> S;
    std::deque<vecXd> Y;
    std::deque<f64> RHO;
    svec<f64> alpha_i(((m > 0) ? m : 0), 0.0);

    if (auto st = detail::init_common(
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

        // Two-loop recursion
        q = g;
        const i32 L = static_cast<i32>(S.size());

        if (L > 0) {
            for (i32 i = L - 1; i >= 0; i--) {
                alpha_i[i] = RHO[i] * S[i].dot(q);
                q.noalias() -= alpha_i[i] * Y[i];
            } // end loop 1
        }

        // Initial H_0 scaling
        if (L > 0 && opt.lbfgs.h0_auto_scale) {
            const vecXd& s_last = S.back();
            const vecXd& y_last = Y.back();
            const f64 sy = s_last.dot(y_last);
            const f64 yy = y_last.dot(y_last);

            f64 gamma = 1.0; // fallback if gamma calculations fail
            if (yy > 0.0) gamma = sy / yy;
            if (!isfinite(gamma)) gamma = 1.0;
            gamma = std::clamp(gamma, opt.lbfgs.h0_scale_min, opt.lbfgs.h0_scale_max);

            r.noalias() = gamma * q;
        } else {
            r = q;
        }

        // loop 2
        for (i32 i = 0; i < L; i++) {
            const f64 beta = RHO[i] * Y[i].dot(r);
            r.noalias() += S[i] * (alpha_i[i] - beta);
        }

        p.noalias() = -r;

        if (g.dot(p) >= 0.0) { // ensure descent direction or reset vectors
            p.noalias() = -g;
            S.clear();
            Y.clear();
            RHO.clear();
        }

        IterDiagnostics diag;
        if (opt.diag.enabled) { // quasi-newton diagnostics
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

        // curvature condition (skip or restart if bad)
        const f64 ys = y.dot(s);
        const f64 ss = s.squaredNorm();
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

        if (!std::isfinite(ys) || ys <= 0.0) {
            // restart on invalid/negative curvature
            S.clear();
            Y.clear();
            RHO.clear();
        } else if (ys > tol_strict * ss) { // curvature acceptance threshold
            // push pair
            if (m > 0) {
                if ((i32)S.size() == m) {
                    S.pop_front();
                    Y.pop_front();
                    RHO.pop_front();
                }
                S.push_back(s);
                Y.push_back(y);
                RHO.push_back(1.0 / ys);
            } else {
                // m == 0 => no memory (degenerates to steepest descent)
                S.clear();
                Y.clear();
                RHO.clear();
            }
        } else {
            // curvature too small (likely noisy): skip update
        }
    } // end iterations

    detail::finalize_common(res, oracle, f, g.norm());
    return res;
}

// overload default Strong Wolfe
template <typename Obj>
Result lbfgs(
    const Obj& obj,
    ecref<vecXd> x0,
    const Options& opt,
    const IterCallback& on_iter = {},
    const StopCallback& should_stop = {}
) {
    return lbfgs(obj, x0, opt, WolfeStrong{}, on_iter, should_stop);
}

} // namespace sOPT