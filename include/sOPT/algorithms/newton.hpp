#pragma once

#include "sOPT/algorithms/detail/solver_common.hpp"
#include "sOPT/core/callback.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/result.hpp"
#include "sOPT/problem/oracle.hpp"
#include "sOPT/step_size/armijo.hpp"

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <cmath>

namespace sOPT {
// Damped Newton (Globalized Newton)
//
// \begin{aligned}
// g_k &= \nabla f(x_k) \\
// H_k &= \nabla^2 f(x_k) \\
// (H_k + \lambda_k I)\, p_k &= -g_k \\
// x_{k+1} &= x_k + \alpha_k p_k
// \end{aligned}
//
// with $\lambda_k \ge 0$ and $\alpha_k \in (0,1]$.
// $\lambda_k$ is increased until $H_k + \lambda_k I$ is SPD, and $\alpha_k$ is chosen by
// a line search

template <typename Obj, typename StepStrategy>
Result newton(
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

    vecXd g(n);       // gradient
    matXd H(n, n);    // hessian
    matXd Hmod(n, n); // modified hessian: Hmod = H + lambda * I
    vecXd p(n);       // descent direction
    vecXd x_next(n);
    f64 f = 0.0;
    f64 f_next = 0.0;
    f64 alpha = 0.0;
    detail::TerminationScales term_scales;

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
        } // end prechecks

        IterDiagnostics diag;
        {
            const detail::EvalStatus hess_status = detail::eval_hess(oracle, res.x, H);
            if (hess_status != detail::EvalStatus::ok) {
                res.status = detail::to_status(hess_status);
                break;
            }
            detail::maybe_fill_hessian_diagnostics(opt, H, diag);
        } // end diagnostics

        p.noalias() = -g;
        bool solved = false;
        f64 lambda = 0.0;

        for (i32 t = 0; t <= opt.newton.damping_max_tries; t++) {
            if (t == 0) {
                Hmod.noalias() = H;
            } else {
                if (t == 1)
                    lambda = opt.newton.damping0;
                else
                    lambda *= opt.newton.damping_scale;
                Hmod.noalias() = H;
                Hmod.diagonal().array() += lambda; // Hmod = H + I \lambda
            }

            eig::LLT<matXd> llt(Hmod); // factor Hmod = L * L^T
            if (llt.info() == eig::Success) {
                p = llt.solve(p);
                if (p.allFinite()) {
                    solved = true;
                    break;
                }
            }
        } // end newton damping

        if (!solved) {                // fallback solve for p
            eig::LDLT<matXd> ldlt(H); // factor Hmod = L * D * L^T
            if (ldlt.info() == eig::Success) {
                p = ldlt.solve(p);
                if (p.allFinite()) solved = true;
            }
        }
        if (!solved) {                      // addition fallback
            eig::PartialPivLU<matXd> lu(H); // Hmod = L * U with partial pivoting
            if (lu.info() != eig::Success) {
                res.status = Status::linear_solve_failed;
                break;
            }
            p = lu.solve(p);
            if (!p.allFinite()) {
                res.status = Status::linear_solve_failed;
                break;
            }
        }

        if (opt.diag.enabled && opt.diag.record_directional_derivative) {
            diag.gTp = g.dot(p);
        }

        // run step strategy with descent direction
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

        const f64 f_prev = f;
        const f64 step_norm = (x_next - res.x).norm();
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
    } // end iteration
    detail::finalize_common(res, oracle, f, g.norm());
    return res;
}

// overload default Armijo
template <typename Obj>
Result newton(
    const Obj& obj,
    ecref<vecXd> x0,
    const Options& opt,
    const IterCallback& on_iter = {},
    const StopCallback& should_stop = {}
) {
    return newton(obj, x0, opt, Armijo{}, on_iter, should_stop);
}

} // namespace sOPT
