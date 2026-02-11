#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/options_validation.hpp"
#include "sOPT/core/result.hpp"
#include "sOPT/core/status.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/step_size/step_attempt.hpp"
#include "sOPT/step_size/try_full.hpp"

#include <Eigen/Cholesky>
#include <cmath>
#include <optional>
#include <type_traits>

// helpers to clean up algorithms
namespace sOPT::detail {

// helper wrappers for status --------------------------------------------------
enum class EvalStatus : u8 { ok = 0, max_evals, eval_failed };
enum class StepStatus : u8 { accepted = 0, line_search_failed, max_evals, eval_failed };
inline Status to_status(EvalStatus s) {
    switch (s) {
        // not used directly as terminal
    case EvalStatus::ok: return Status::success;
    case EvalStatus::max_evals: return Status::max_evals;
    case EvalStatus::eval_failed: return Status::eval_failed;
    }
    return Status::eval_failed;
}
inline Status to_status(StepStatus s) {
    switch (s) {
        // not terminal
    case StepStatus::accepted: return Status::success;
    case StepStatus::line_search_failed: return Status::line_search_failed;
    case StepStatus::max_evals: return Status::max_evals;
    case StepStatus::eval_failed: return Status::eval_failed;
    }
    return Status::line_search_failed;
}

// wrappers so solvers don’t touch oracle-specific failure logic
template <typename OracleT>
inline EvalStatus eval_func(OracleT& oracle, ecref<vecXd> x, f64& f) {
    // try_func failure is max_evals iff the function-eval budget is exhausted.
    if (!oracle.try_func(x, f)) {
        return oracle.f_limit_reached() ? EvalStatus::max_evals : EvalStatus::eval_failed;
    }
    return isfinite(f) ? EvalStatus::ok : EvalStatus::eval_failed;
}

template <typename OracleT>
inline EvalStatus eval_grad(OracleT& oracle, ecref<vecXd> x, eref<vecXd> g) {
    // try_gradient failure is max_evals iff the gradient-eval budget is exhausted.
    if (!oracle.try_gradient(x, g)) {
        return oracle.g_limit_reached() ? EvalStatus::max_evals : EvalStatus::eval_failed;
    }
    return g.allFinite() ? EvalStatus::ok : EvalStatus::eval_failed;
}

template <typename OracleT>
inline EvalStatus eval_hess(OracleT& oracle, ecref<vecXd> x, eref<matXd> H) {
    // try_hessian failure is max_evals iff the hessian-eval budget is exhausted.
    if (!oracle.try_hessian(x, H)) {
        return oracle.h_limit_reached() ? EvalStatus::max_evals : EvalStatus::eval_failed;
    }
    return H.allFinite() ? EvalStatus::ok : EvalStatus::eval_failed;
}

// Tolerance and termination ---------------------------------------------------
struct TerminationScales {
    // used for relative tolerance termination
    f64 grad_ref = 1.0;
    f64 step_ref = 1.0;
};

inline f64 sanitize_tol_nonneg(f64 v) {
    return (isfinite(v) && v > 0.0) ? v : 0.0;
}
inline f64 sanitize_ref_pos(f64 v) {
    return (isfinite(v) && v > 0.0) ? v : 1.0;
}

inline f64
grad_tol_effective(const Options& opt, const TerminationScales* scales = nullptr) {
    const f64 abs_tol = sanitize_tol_nonneg(opt.term.grad_tol);
    const f64 rel_tol = sanitize_tol_nonneg(opt.term.grad_tol_rel);
    const f64 grad_ref = scales ? sanitize_ref_pos(scales->grad_ref) : 1.0;
    return std::max(abs_tol, rel_tol * grad_ref);
}

inline f64
step_tol_effective(const Options& opt, const TerminationScales* scales = nullptr) {
    const f64 abs_tol = sanitize_tol_nonneg(opt.term.step_tol);
    const f64 rel_tol = sanitize_tol_nonneg(opt.term.step_tol_rel);
    const f64 step_ref = scales ? sanitize_ref_pos(scales->step_ref) : 1.0;
    return std::max(abs_tol, rel_tol * step_ref);
}

inline bool is_step_converged(
    f64 step_norm,
    const Options& opt,
    const TerminationScales* scales = nullptr
) {
    if (!isfinite(step_norm)) return false;
    return step_norm <= step_tol_effective(opt, scales);
}

inline bool is_f_change_converged(f64 f_prev, f64 f_curr, const Options& opt) {
    const f64 f_tol = sanitize_tol_nonneg(opt.term.f_tol);
    if (f_tol <= 0.0) return false;
    if (!isfinite(f_prev) || !isfinite(f_curr)) return false;
    const f64 scale = std::max(1.0, std::abs(f_prev));
    return std::abs(f_curr - f_prev) <= f_tol * scale;
}

// init, checks, bookkeeping ---------------------------------------------------
// returns terminal status if solver should exit immediately, else std::nullopt
template <typename OracleT>
inline std::optional<Status> init_common(
    OracleT& oracle,
    const Options& opt,
    ecref<vecXd> x0,
    Result& res,
    eref<vecXd> g,
    f64& f,
    const IterCallback& on_iter,
    const StopCallback& should_stop,
    TerminationScales* scales = nullptr
) {
    res.trace_init(opt);
    res.x = x0;
    res.iterations = 0;
    res.status = Status::invalid_input;

    if (g.size() != res.x.size()) {
        res.status = Status::invalid_input;
        return res.status;
    }

    if (opt.validate_options) {
        const OptionsValidationResult v = validate_options(opt);
        if (!v.ok) {
            res.status = Status::invalid_input;
            return res.status;
        }
    }

    { // try first function eval
        const EvalStatus stf = eval_func(oracle, res.x, f);
        if (stf != EvalStatus::ok) {
            res.status = to_status(stf);
            return res.status;
        }
    }

    { // try first gradient eval
        const EvalStatus stg = eval_grad(oracle, res.x, g);
        if (stg != EvalStatus::ok) {
            res.status = to_status(stg);
            return res.status;
        }
    }

    const f64 gnorm = g.norm();
    if (scales) {
        scales->grad_ref = std::max(1.0, std::abs(gnorm));
        scales->step_ref = std::max(1.0, res.x.norm());
    }
    IterDiagnostics diag;
    res.trace_push(opt, oracle, f, gnorm, 0.0, 0.0, diag);

    if (!isfinite(f) || !isfinite(gnorm)) {
        res.status = Status::nan_detected;
        return res.status;
    }

    IterInfo it;
    it.iter = 0;
    it.f = f;
    it.grad_norm = gnorm;
    it.step_norm = 0.0;
    it.alpha = 0.0;
    it.diag = diag;

    if (on_iter) on_iter(it);
    if (should_stop && should_stop(it)) {
        res.status = Status::user_terminated;
        return res.status;
    }

    return std::nullopt;
}

inline std::optional<Status> pre_step_checks(
    f64 f,
    f64 gnorm,
    const Options& opt,
    const TerminationScales* scales = nullptr
) {
    if (!isfinite(f) || !isfinite(gnorm)) {
        return Status::nan_detected;
    }
    if (gnorm <= grad_tol_effective(opt, scales)) {
        return Status::converged_grad;
    }
    return std::nullopt;
}

inline std::optional<Status> check_step_convergence(
    f64 step_norm,
    f64 f_prev,
    f64 f_curr,
    const Options& opt,
    const TerminationScales* scales = nullptr
) {
    if (!isfinite(step_norm)) {
        return Status::nan_detected;
    }
    if (is_step_converged(step_norm, opt, scales)) {
        return Status::converged_step;
    }
    if (!isfinite(f_prev) || !isfinite(f_curr)) {
        return Status::nan_detected;
    }
    if (is_f_change_converged(f_prev, f_curr, opt)) {
        return Status::success;
    }
    return std::nullopt;
}

// compatibility overload
inline std::optional<Status> check_step_convergence(f64 step_norm, const Options& opt) {
    return check_step_convergence(step_norm, qNaN<f64>, qNaN<f64>, opt, nullptr);
}

// handles accepted step bookkeeping
// returns termination status if should exit, else std::nullopt
template <typename OracleT>
inline std::optional<Status> post_accept_common(
    OracleT& oracle,
    const Options& opt,
    Result& res,
    vecXd& x,
    f64& f,
    eref<vecXd> g,
    vecXd& x_next,
    f64 f_next,
    f64 alpha,
    f64 step_norm,
    const IterDiagnostics& diag,
    const IterCallback& on_iter,
    const StopCallback& should_stop
) {
    // accept step
    x.swap(x_next);
    f = f_next;
    ++res.iterations;

    // refresh gradient at accepted iterate
    {
        const EvalStatus stg = eval_grad(oracle, x, g);
        if (stg != EvalStatus::ok) {
            res.status = to_status(stg);
            return res.status;
        }
    }

    const f64 gnorm = g.norm();

    res.trace_push(opt, oracle, f, gnorm, step_norm, alpha, diag);

    IterInfo it;
    it.iter = res.iterations;
    it.f = f;
    it.grad_norm = gnorm;
    it.step_norm = step_norm;
    it.alpha = alpha;
    it.diag = diag;

    if (on_iter) on_iter(it);
    if (should_stop && should_stop(it)) {
        res.status = Status::user_terminated;
        return res.status;
    }

    return std::nullopt;
}

template <typename OracleT>
inline std::optional<Status> post_accept_with_step_status(
    OracleT& oracle,
    const Options& opt,
    Result& res,
    vecXd& x,
    f64& f,
    eref<vecXd> g,
    vecXd& x_next,
    f64 f_next,
    f64 f_prev,
    f64 alpha,
    f64 step_norm,
    const IterDiagnostics& diag,
    const IterCallback& on_iter,
    const StopCallback& should_stop,
    const TerminationScales* scales = nullptr
) {
    if (auto st = post_accept_common(
            oracle,
            opt,
            res,
            x,
            f,
            g,
            x_next,
            f_next,
            alpha,
            step_norm,
            diag,
            on_iter,
            should_stop
        )) {
        return st;
    }
    return check_step_convergence(step_norm, f_prev, f, opt, scales);
}

template <typename OracleT>
void finalize_common(Result& res, const OracleT& oracle, f64 f, f64 grad_norm) {
    res.f = f;
    res.grad_norm = grad_norm;
    res.sync_eval_counts(oracle);
    if (res.status == Status::invalid_input) {
        const bool no_evals
            = (res.f_evals == 0) && (res.g_evals == 0) && (res.h_evals == 0);
        if (!no_evals) {
            res.status = Status::max_iters;
        }
    }
}

// run step size strategy algo or try full step
// step wrapper returns StepStatus, not bool
//
// Status precedence (during step attempt):
// 1) accepted step => StepStatus::accepted
// 2) eval failure => StepStatus::eval_failed (or max_evals if step-relevant
//    budgets are exhausted)
// 3) line-search rule failure => StepStatus::line_search_failed (or max_evals if
//    step-relevant budgets are exhausted)
//
// Limit semantics note:
// Step-size strategies use function/gradient evaluations, not Hessian
// evaluations. Mapping here intentionally checks only f/g budgets to avoid
// unrelated h-limit exhaustion masking true step outcomes in first-order solvers.
template <typename OracleT, typename StepStrategy>
inline StepStatus run_step(
    OracleT& oracle,
    const Options& opt,
    const StepStrategy& step,
    ecref<vecXd> x,
    f64 f,
    ecref<vecXd> g,
    ecref<vecXd> p,
    f64& alpha,
    vecXd& x_next,
    f64& f_next
) {
    auto map_raw = [&](const auto& raw_result) -> StepStatus {
        // Query budgets after the step attempt has run, so limits crossed
        // during line-search evaluations are reflected in terminal mapping.
        const bool step_limits_reached
            = oracle.f_limit_reached() || oracle.g_limit_reached();

        if constexpr (std::is_same_v<std::decay_t<decltype(raw_result)>, StepAttempt>) {
            if (raw_result == StepAttempt::accepted) return StepStatus::accepted;
            if (raw_result == StepAttempt::eval_failed) {
                return step_limits_reached ? StepStatus::max_evals
                                           : StepStatus::eval_failed;
            }
            return step_limits_reached ? StepStatus::max_evals
                                       : StepStatus::line_search_failed;
        } else {
            // TODO: remove this later
            if (raw_result) return StepStatus::accepted;
            return step_limits_reached ? StepStatus::max_evals
                                       : StepStatus::line_search_failed;
        }
    };

    if (opt.ls.try_full_step) {
        const auto raw_result
            = TryFull<StepStrategy>{step}(oracle, x, f, g, p, alpha, x_next, f_next, opt);
        return map_raw(raw_result);
    }

    const auto raw_result = step(oracle, x, f, g, p, alpha, x_next, f_next, opt);
    return map_raw(raw_result);
}

inline f64 condition_estimate_power_iteration(ecref<matXd> H, i32 iters, f64 eps) {
    // checks condition estimate via power iteration
    // the matrix condition number indicates how stable the inversion is
    const i32 n = static_cast<i32>(H.rows());
    if (n <= 0 || H.cols() != n) {
        return qNaN<f64>;
    }

    // Use a symmetrized copy for robust Rayleigh/inverse-iteration estimates.
    matXd Hsym = sym_transpose_avg(H);
    if (!Hsym.allFinite()) return qNaN<f64>;

    const i32 kmax = std::max(1, iters);
    const f64 eps_safe = std::max(eps, 1e-16);

    vecXd v = vecXd::Ones(n);
    f64 vnorm = v.norm();
    if (!(vnorm > eps_safe) || !isfinite(vnorm)) return qNaN<f64>;

    v /= vnorm;

    // Power iteration to estimate largest eigenvalue of H
    // ref: https://en.wikipedia.org/wiki/Power_iteration
    f64 lambda_max = qNaN<f64>;
    for (i32 k = 0; k < kmax; k++) {
        vecXd w = Hsym * v;
        const f64 wn = w.norm();
        if (!(wn > eps_safe) || !isfinite(wn)) return qNaN<f64>;
        v = w / wn;
        lambda_max = std::abs(v.dot(Hsym * v));
    }
    if (!(lambda_max > eps_safe) || !isfinite(lambda_max)) return qNaN<f64>;

    // Check Hsym nonsingular
    eig::LDLT<matXd> ldlt(Hsym);
    if (ldlt.info() != eig::Success) return qNaN<f64>;

    // check smallest on factor Hsym diag
    const auto D = ldlt.vectorD();
    if (D.size() <= 0) return qNaN<f64>;
    const f64 min_abs_diag = D.cwiseAbs().minCoeff();
    if (!(min_abs_diag > eps_safe) || !isfinite(min_abs_diag)) return qNaN<f64>;

    vecXd u = vecXd::Ones(n);
    f64 unorm = u.norm();
    if (!(unorm > eps_safe) || !isfinite(unorm)) return qNaN<f64>;
    u /= unorm;
    // Inverse power iteration to find smallest eigenvalue
    f64 lambda_min = qNaN<f64>;
    for (i32 k = 0; k < kmax; ++k) {
        vecXd z = ldlt.solve(u);
        const f64 zn = z.norm();
        if (!(zn > eps_safe) || !isfinite(zn)) return qNaN<f64>;
        u = z / zn;
        lambda_min = std::abs(u.dot(Hsym * u));
    }
    if (!(lambda_min > eps_safe) || !isfinite(lambda_min)) return qNaN<f64>;
    const f64 cond = lambda_max / lambda_min;
    return (isfinite(cond) && cond >= 1.0) ? cond : qNaN<f64>;
}

inline void maybe_fill_hessian_diagnostics(
    const Options& opt,
    ecref<matXd> H,
    IterDiagnostics& diag
) {
    if (!opt.diag.enabled) return;
    if (H.rows() <= 0 || H.cols() != H.rows()) return;

    if (opt.diag.record_hessian_diag_bounds) {
        const auto d = H.diagonal();
        diag.hdiag_min = d.minCoeff();
        diag.hdiag_max = d.maxCoeff();
    }

    const f64 eps = std::max(opt.diag.cond_eps, 1e-16);
    switch (opt.diag.cond_mode) {
    case ConditionEstimateMode::off: return;
    case ConditionEstimateMode::diagonal_proxy: {
        // cheaper than power iteration
        const auto d = H.diagonal().cwiseAbs();
        const f64 dmax = d.maxCoeff();
        const f64 dmin = d.minCoeff();
        if (isfinite(dmax) && isfinite(dmin)) {
            diag.cond_est = dmax / std::max(dmin, eps);
        }
        return;
    }
    case ConditionEstimateMode::power_iteration:
        // expensive
        diag.cond_est = condition_estimate_power_iteration(
            H,
            opt.diag.cond_power_iters,
            opt.diag.cond_eps
        );
        return;
    }
}
} // namespace sOPT::detail