#pragma once

namespace sOPT {

enum struct Status {
    success,
    converged_grad,
    converged_step,
    max_iters,
    max_evals,
    invalid_input,
    eval_failed,
    line_search_failed,
    nan_detected,
    user_terminated,
    linear_solve_failed,
    not_implemented,
};

constexpr const char* to_string(Status s) {
    switch (s) {
    case Status::success: return "success";
    case Status::converged_grad: return "converged_grad";
    case Status::converged_step: return "converged_step";
    case Status::max_iters: return "max_iters";
    case Status::max_evals: return "max_evals";
    case Status::invalid_input: return "invalid_input";
    case Status::eval_failed: return "eval_failed";
    case Status::line_search_failed: return "line_search_failed";
    case Status::linear_solve_failed: return "linear_solve_failed";
    case Status::not_implemented: return "not_implemented";
    case Status::nan_detected: return "nan_detected";
    case Status::user_terminated: return "user_terminated";
    }
    return "unknown";
}

} // namespace sOPT
