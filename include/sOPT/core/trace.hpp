#pragma once

#include "sOPT/core/typedefs.hpp"

namespace sOPT {

enum class TraceLevel : u8 {
    off = 0,
    basic, // f, grad_norm, step_norm, alpha
    full   // basic + eval counts
};

struct Trace {
    // TraceLevel::basic
    svec<f64> f;
    svec<f64> grad_norm;
    svec<f64> step_norm;
    svec<f64> alpha;

    // TraceLevel::full
    svec<i32> f_evals;
    svec<i32> g_evals;
    svec<i32> h_evals;

    // Optionals
    svec<f64> gTp;
    svec<f64> ys;
    svec<f64> ys_cos;
    svec<f64> hdiag_min;
    svec<f64> hdiag_max;
    svec<f64> cond_est;
    // Reserve vectors

    void
    reserve(i32 n, TraceLevel trace_level = TraceLevel::off, bool with_diag = false) {
        switch (trace_level) {
        case TraceLevel::off: return;
        case TraceLevel::full:
            f_evals.reserve(n);
            g_evals.reserve(n);
            h_evals.reserve(n);
            [[fallthrough]];
        case TraceLevel::basic:
            f.reserve(n);
            grad_norm.reserve(n);
            step_norm.reserve(n);
            alpha.reserve(n);
            if (with_diag) {
                // Add optionals here
                gTp.reserve(n);
                ys.reserve(n);
                ys_cos.reserve(n);
                hdiag_min.reserve(n);
                hdiag_max.reserve(n);
                cond_est.reserve(n);
            }
            return;
        }
    }
};

} // namespace sOPT