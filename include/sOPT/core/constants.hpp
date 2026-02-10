#pragma once

#include "sOPT/core/typedefs.hpp"

namespace sOPT {

const f64 pi = 3.1415926535897932384626433832795;
const f64 pio2 = pi / 2.;
const f64 pio4 = pi / 4.;
const f64 pio8 = pi / 8.;
const f64 pio16 = pi / 16.;
const f64 pio32 = pi / 32.;
const f64 pio64 = pi / 64.;

const f64 deg_to_rad = pi / 180.;
const f64 rad_to_deg = 180. / pi;

const f64 rl_to_u = 1000;
const f64 u_to_rl = 1. / rl_to_u;

// gravitational constant
const f64 G_m = 6.6743e-11;
const f64 G_km = 6.6743e-20;

const f64 tol_loose = 1e-3;
const f64 tol_med = 1e-6;
const f64 tol_tight = 1e-9;
const f64 tol_strict = 1e-12;
const f64 tol_max = 1e-16;

} // namespace sOPT