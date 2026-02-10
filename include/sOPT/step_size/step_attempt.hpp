#pragma once

#include "sOPT/core/typedefs.hpp"

namespace sOPT {

enum class StepAttempt : u8 {
    accepted = 0,
    line_search_failed,
    eval_failed
};

} // namespace sOPT

