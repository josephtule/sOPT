#pragma once

#include "sOPT/core/constants.hpp"
#include "sOPT/core/trace.hpp"
#include "sOPT/core/typedefs.hpp"

namespace sOPT {

// finite difference options
enum struct FallbackGrad {
    fd_forward,
    fd_backward,
    fd_central,
    fd_forward_2,
    fd_backward_2,
    fd_central_2
};
enum struct FallbackHess {
    fd_forward,
    fd_backward,
    fd_central,
    fd_forward_2,
    fd_backward_2,
    fd_central_2
};
enum struct FallbackHv {
    fd_forward,
    fd_backward,
    fd_central,
    fd_forward_2,
    fd_backward_2,
    fd_central_2
};

struct FDOptions {
    FallbackGrad fallback_grad = FallbackGrad::fd_central;
    FallbackHess fallback_hess = FallbackHess::fd_central;
    FallbackHv fallback_hv = FallbackHv::fd_central;
    f64 eps = 1e-8;
    f64 hv_eps = 1e-6;
};

struct LineSearchOptions {
    bool try_full_step = true;

    f64 alpha_fixed = 1e-2; // fixed-step GD
    f64 alpha0 = 1.0;       // initial step
    f64 alpha_max = 64.0;   // max expansion

    f64 rho = 0.5; // backtracking factor
    f64 c1 = 1e-4; // Armijo / Wolfe c1
    f64 c2 = 0.9;  // Wolfe c2

    i32 max_iters = 40;
};

struct TerminationOptions {
    i32 max_iters = 2000;

    f64 grad_tol = tol_med;   // absolute: stop if ||g|| <= grad_tol
    f64 grad_tol_rel = 0.0;   // relative: stop if ||g|| <= grad_tol_rel * grad_ref
    f64 step_tol = tol_tight; // absolute: stop if ||dx|| <= step_tol
    f64 step_tol_rel = 0.0;   // relative: stop if ||dx|| <= step_tol_rel * step_ref
    f64 f_tol = 0.0;          // relative: stop if |df| <= f_tol * max(1, |f_prev|)
};

struct CacheOptions {
    bool enabled = true;
    i32 f_slots = 4;
    i32 g_slots = 2;
    i32 h_slots = 0; // set 0 for large n
    bool enforce_max_bytes = true;
    i64 max_bytes = 256ll * 1024ll * 1024ll; // 256 MiB
};

struct EvalLimitOptions {
    i32 max_f_evals = 200000;
    i32 max_g_evals = 100000;
    i32 max_h_evals = 25000;
};

enum class ConditionEstimateMode : u8 { off = 0, diagonal_proxy, power_iteration };

struct DiagnosticsOptions {
    bool enabled = false;

    bool record_directional_derivative = true; // g^T p
    bool record_qn_curvature = true;           // (QN) y^T s and normalized alignment
    bool record_hessian_diag_bounds = true;    // min/max diag(H)

    ConditionEstimateMode cond_mode = ConditionEstimateMode::diagonal_proxy;
    i32 cond_power_iters = 6;
    f64 cond_eps = 1e-12;
};

struct NewtonOptions {
    f64 damping0 = 1e-6;
    f64 damping_scale = 10.0;
    i32 damping_max_tries = 10;
};

struct LBFGSOptions {
    i32 memory = 20;
    bool h0_auto_scale = true;
    f64 h0_scale_min = 1e-8;
    f64 h0_scale_max = 1e+8;
};

struct Options {
    // Core options
    TerminationOptions term;
    FDOptions fd;
    CacheOptions cache;
    EvalLimitOptions limits;
    DiagnosticsOptions diag;

    // Situational options
    LineSearchOptions ls;
    NewtonOptions newton;
    LBFGSOptions lbfgs;

    // Trace Options
    TraceLevel trace_level = TraceLevel::off;
    i32 trace_reserve = 0; // 0 => reserver max_iters + 1

    bool validate_options = true; // set false if solver called multiple times (i.e. in
                                  // optimal control problems)
};

} // namespace sOPT