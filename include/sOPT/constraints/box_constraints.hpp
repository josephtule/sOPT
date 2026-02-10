#pragma once

#include "sOPT/core/vecdefs.hpp"
namespace sOPT {

struct BoxConstraints {
    vecXd lower;
    vecXd upper;
};

inline bool has_box_lower(const BoxConstraints& box) {
    return box.lower.size() > 0;
}
inline bool has_box_upper(const BoxConstraints& box) {
    return box.upper.size() > 0;
}

} // namespace sOPT