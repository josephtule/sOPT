#pragma once

#include "sOPT/core/math.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/typedefs.hpp"

namespace sOPT {

enum class OptionsValidationError : u8 {
    none = 0,
    // Core options
    term_max_iters_nonpositive,
    term_grad_tol_negative,
    term_grad_tol_rel_negative,
    term_step_tol_negative,
    term_step_tol_rel_negative,
    term_f_tol_negative,
    cache_f_slots_negative,
    cache_g_slots_negative,
    cache_h_slots_negative,
    ls_alpha_fixed_nonpositive,
    ls_alpha0_nonpositive,
    ls_alpha_max_too_small,
    ls_rho_out_of_range,
    ls_c1_out_of_range,
    ls_c2_out_of_range,
    ls_c1_c2_inconsistent,
    ls_max_iters_nonpositive,
    // Situational options
    newton_damping0_nonpositive,
    newton_damping_scale_nonpositive,
    newton_damping_max_tries_negative,
    lbfgs_memory_negative,
    lbfgs_h0_scale_min_nonpositive,
    lbfgs_h0_scale_max_nonpositive,
    lbfgs_h0_scale_bounds_invalid,
    diag_cond_power_iters_negative,
    diag_cond_eps_nonpositive,
};

struct OptionsValidationResult {
    bool ok = true;
    OptionsValidationError code = OptionsValidationError::none;
    const char* message = "ok";

    explicit operator bool() const { return ok; }
};

inline OptionsValidationResult
options_invalid(OptionsValidationError code, const char* message) {
    return OptionsValidationResult{.ok = false, .code = code, .message = message};
}

inline OptionsValidationResult validate_options(const Options& opt) {
    if (opt.term.max_iters <= 0) {
        return options_invalid(
            OptionsValidationError::term_max_iters_nonpositive,
            "term.max_iters must be > 0"
        );
    }
    if (!finite_nonneg(opt.term.grad_tol)) {
        return options_invalid(
            OptionsValidationError::term_grad_tol_negative,
            "term.grad_tol must be finite and >= 0"
        );
    }
    if (!finite_nonneg(opt.term.grad_tol_rel)) {
        return options_invalid(
            OptionsValidationError::term_grad_tol_rel_negative,
            "term.grad_tol_rel must be finite and >= 0"
        );
    }
    if (!finite_nonneg(opt.term.step_tol)) {
        return options_invalid(
            OptionsValidationError::term_step_tol_negative,
            "term.step_tol must be finite and >= 0"
        );
    }
    if (!finite_nonneg(opt.term.step_tol_rel)) {
        return options_invalid(
            OptionsValidationError::term_step_tol_rel_negative,
            "term.step_tol_rel must be finite and >= 0"
        );
    }
    if (!finite_nonneg(opt.term.f_tol)) {
        return options_invalid(
            OptionsValidationError::term_f_tol_negative,
            "term.f_tol must be finite and >= 0"
        );
    }

    if (!finite_pos(opt.ls.alpha_fixed)) {
        return options_invalid(
            OptionsValidationError::ls_alpha_fixed_nonpositive,
            "ls.alpha_fixed must be finite and > 0"
        );
    }
    if (!finite_pos(opt.ls.alpha0)) {
        return options_invalid(
            OptionsValidationError::ls_alpha0_nonpositive,
            "ls.alpha0 must be finite and > 0"
        );
    }
    if (!finite_pos(opt.ls.alpha_max) || opt.ls.alpha_max < opt.ls.alpha0) {
        return options_invalid(
            OptionsValidationError::ls_alpha_max_too_small,
            "ls.alpha_max must be finite, > 0, and >= ls.alpha0"
        );
    }
    if (!(isfinite(opt.ls.rho) && opt.ls.rho > 0.0 && opt.ls.rho < 1.0)) {
        return options_invalid(
            OptionsValidationError::ls_rho_out_of_range,
            "ls.rho must satisfy 0 < rho < 1"
        );
    }
    if (!(isfinite(opt.ls.c1) && opt.ls.c1 > 0.0 && opt.ls.c1 < 1.0)) {
        return options_invalid(
            OptionsValidationError::ls_c1_out_of_range,
            "ls.c1 must satisfy 0 < c1 < 1"
        );
    }
    if (!(isfinite(opt.ls.c2) && opt.ls.c2 > 0.0 && opt.ls.c2 < 1.0)) {
        return options_invalid(
            OptionsValidationError::ls_c2_out_of_range,
            "ls.c2 must satisfy 0 < c2 < 1"
        );
    }
    if (!(opt.ls.c1 < opt.ls.c2)) {
        return options_invalid(
            OptionsValidationError::ls_c1_c2_inconsistent,
            "ls constants must satisfy c1 < c2"
        );
    }
    if (opt.ls.max_iters <= 0) {
        return options_invalid(
            OptionsValidationError::ls_max_iters_nonpositive,
            "ls.max_iters must be > 0"
        );
    }
    if (!finite_pos(opt.newton.damping0)) {
        return options_invalid(
            OptionsValidationError::newton_damping0_nonpositive,
            "newton.damping0 must be finite and > 0"
        );
    }
    if (!finite_pos(opt.newton.damping_scale)) {
        return options_invalid(
            OptionsValidationError::newton_damping_scale_nonpositive,
            "newton.damping_scale must be finite and > 0"
        );
    }
    if (opt.newton.damping_max_tries < 0) {
        return options_invalid(
            OptionsValidationError::newton_damping_max_tries_negative,
            "newton.damping_max_tries must be >= 0"
        );
    }

    if (opt.lbfgs.memory < 0) {
        return options_invalid(
            OptionsValidationError::lbfgs_memory_negative,
            "lbfgs.memory must be >= 0"
        );
    }
    if (!finite_pos(opt.lbfgs.h0_scale_min)) {
        return options_invalid(
            OptionsValidationError::lbfgs_h0_scale_min_nonpositive,
            "lbfgs.h0_scale_min must be finite and > 0"
        );
    }
    if (!finite_pos(opt.lbfgs.h0_scale_max)) {
        return options_invalid(
            OptionsValidationError::lbfgs_h0_scale_max_nonpositive,
            "lbfgs.h0_scale_max must be finite and > 0"
        );
    }
    if (opt.lbfgs.h0_scale_min > opt.lbfgs.h0_scale_max) {
        return options_invalid(
            OptionsValidationError::lbfgs_h0_scale_bounds_invalid,
            "lbfgs.h0_scale_min must be <= lbfgs.h0_scale_max"
        );
    }
    if (opt.cache.f_slots < 0) {
        return options_invalid(
            OptionsValidationError::cache_f_slots_negative,
            "cache.f_slots must be >= 0"
        );
    }
    if (opt.cache.g_slots < 0) {
        return options_invalid(
            OptionsValidationError::cache_g_slots_negative,
            "cache.g_slots must be >= 0"
        );
    }
    if (opt.cache.h_slots < 0) {
        return options_invalid(
            OptionsValidationError::cache_h_slots_negative,
            "cache.h_slots must be >= 0"
        );
    }

    if (opt.diag.cond_power_iters < 0) {
        return options_invalid(
            OptionsValidationError::diag_cond_power_iters_negative,
            "diag.cond_power_iters must be >= 0"
        );
    }
    if (!finite_pos(opt.diag.cond_eps)) {
        return options_invalid(
            OptionsValidationError::diag_cond_eps_nonpositive,
            "diag.cond_eps must be finite and > 0"
        );
    }

    return OptionsValidationResult{};
}

} // namespace sOPT