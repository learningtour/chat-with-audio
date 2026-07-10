#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ait {

constexpr double kPi = 3.14159265358979323846;

inline float db2lin(float db) { return std::pow(10.0f, db * 0.05f); }

inline float lin2db(float lin) { return 20.0f * std::log10(std::max(lin, 1e-10f)); }

// One-pole smoothing coefficient for a time constant in milliseconds.
inline float smooth_coef(float ms, float sr) {
    if (ms <= 0.0f) return 0.0f;
    return std::exp(-1.0f / (0.001f * ms * sr));
}

}  // namespace ait
