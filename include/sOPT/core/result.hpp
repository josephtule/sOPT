#pragma once

#include "sOPT/core/callback.hpp"
#include "sOPT/core/options.hpp"
#include "sOPT/core/trace.hpp"
#include "sOPT/core/vecdefs.hpp"
#include <optional>

namespace sOPT {

struct Result {
    vecXd x;
    f64 f = 0.0;
    f64 grad_norm = 0.0;

    i32 iterations = 0;
    i32 f_evals = 0;
    i32 g_evals = 0;
    i32 h_evals = 0;

    // Trace
    std::optional<Trace> trace;
    void trace_init(const Options& opt) {
        if (opt.trace_level != TraceLevel::off) {
            trace.emplace();
            const i32 cap
                = (opt.trace_reserve > 0) ? opt.trace_reserve : opt.term.max_iters + 1;
            trace->reserve(cap, opt.trace_level, opt.diag.enabled);
        }
    }

    template <typename OracleT>
    void trace_push(
        const Options& opt,
        const OracleT& oracle,
        f64 f,
        f64 grad_norm,
        f64 step_norm,
        f64 alpha,
        const IterDiagnostics& diag = {}
    ) {
        if (!trace || opt.trace_level == TraceLevel::off) return;

        switch (opt.trace_level) {
        case TraceLevel::full:
            trace->f_evals.push_back(oracle.f_evals());
            trace->g_evals.push_back(oracle.g_evals());
            trace->h_evals.push_back(oracle.h_evals());
            [[fallthrough]];
        case TraceLevel::basic:
            trace->f.push_back(f);
            trace->grad_norm.push_back(grad_norm);
            trace->step_norm.push_back(step_norm);
            trace->alpha.push_back(alpha);
            if (opt.diag.enabled) {
                // Push optional diagnostics
            }
            break;
        case TraceLevel::off: [[unlikely]]; break; // unreachable
        }
    }
};

} // namespace sOPT