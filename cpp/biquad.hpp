#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "dsp_common.hpp"

namespace ait {

struct BiquadCoeffs {
    float b0, b1, b2, a1, a2;  // normalized (a0 == 1)
};

// RBJ Audio EQ Cookbook designs. gain_db is only used by peaking/shelf types.
inline BiquadCoeffs design_biquad(const std::string& type, double sr, double freq,
                                  double gain_db, double q) {
    const double A = std::pow(10.0, gain_db / 40.0);
    const double w0 = 2.0 * kPi * freq / sr;
    const double cw = std::cos(w0);
    const double sw = std::sin(w0);
    const double alpha = sw / (2.0 * q);
    double b0, b1, b2, a0, a1, a2;

    if (type == "lowpass") {
        b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2;
        a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
    } else if (type == "highpass") {
        b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2;
        a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
    } else if (type == "notch") {
        b0 = 1; b1 = -2 * cw; b2 = 1;
        a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
    } else if (type == "peaking") {
        b0 = 1 + alpha * A; b1 = -2 * cw; b2 = 1 - alpha * A;
        a0 = 1 + alpha / A; a1 = -2 * cw; a2 = 1 - alpha / A;
    } else if (type == "lowshelf") {
        const double sqA = std::sqrt(A);
        b0 = A * ((A + 1) - (A - 1) * cw + 2 * sqA * alpha);
        b1 = 2 * A * ((A - 1) - (A + 1) * cw);
        b2 = A * ((A + 1) - (A - 1) * cw - 2 * sqA * alpha);
        a0 = (A + 1) + (A - 1) * cw + 2 * sqA * alpha;
        a1 = -2 * ((A - 1) + (A + 1) * cw);
        a2 = (A + 1) + (A - 1) * cw - 2 * sqA * alpha;
    } else if (type == "highshelf") {
        const double sqA = std::sqrt(A);
        b0 = A * ((A + 1) + (A - 1) * cw + 2 * sqA * alpha);
        b1 = -2 * A * ((A - 1) + (A + 1) * cw);
        b2 = A * ((A + 1) + (A - 1) * cw - 2 * sqA * alpha);
        a0 = (A + 1) - (A - 1) * cw + 2 * sqA * alpha;
        a1 = 2 * ((A - 1) - (A + 1) * cw);
        a2 = (A + 1) - (A - 1) * cw - 2 * sqA * alpha;
    } else {
        throw std::invalid_argument("unknown biquad type: " + type);
    }

    return {float(b0 / a0), float(b1 / a0), float(b2 / a0), float(a1 / a0), float(a2 / a0)};
}

class Biquad {
public:
    explicit Biquad(const BiquadCoeffs& c) : c_(c) {}

    // Transposed direct form II.
    inline float process(float x) {
        const float y = c_.b0 * x + z1_;
        z1_ = c_.b1 * x - c_.a1 * y + z2_;
        z2_ = c_.b2 * x - c_.a2 * y;
        return y;
    }

    void process_block(float* buf, size_t n) {
        for (size_t i = 0; i < n; ++i) buf[i] = process(buf[i]);
    }

private:
    BiquadCoeffs c_;
    float z1_ = 0.0f, z2_ = 0.0f;
};

}  // namespace ait
