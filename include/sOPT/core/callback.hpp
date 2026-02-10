#pragma once

#include "sOPT/core/typedefs.hpp"
#include <functional>

namespace sOPT {

struct IterDiagnostics {
    bool accepted = true;
    
    // directional derivative
    f64 gTp = std::numeric_limits<f64>::quiet_NaN();
};

struct IterInfo {
    i32 iter = 0;
    f64 f = 0.0;
    f64 grad_norm = 0.0;
    f64 step_norm = 0.0;
    f64 alpha = 0.0;
    IterDiagnostics diag{};
};

using IterCallback = std::function<void(const IterInfo&)>;
using StopCallback = std::function<bool(const IterInfo&)>; // return true to stop

} // namespace sOPT
