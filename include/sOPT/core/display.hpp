#pragma once

#include "sOPT/core/result.hpp"

#include <print>

namespace sOPT {

inline void print_sOPT_results(Result& res) {
    std::println(
        "Solver exited in {} iterations with exit type: '{}'",
        res.iterations,
        to_string(res.status)
    );
    std::println(
        "Evaluations:\nFunction evals: {}\nGradient evals: {}\nHessian evals: "
        "{}",
        res.f_evals,
        res.g_evals,
        res.h_evals
    );
    std::println("gradient norm: {}", res.grad_norm);
    std::println("Optimal value x* = {}", res.x);
    std::println("With the optimal objective: J(x*) = {}", res.f);
}

} // namespace sOPT