#pragma once

#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

#include "dsp_common.hpp"

namespace ait {

// All processors operate in-place on `nch` channel buffers of `n` samples and use a
// linked detector (max |x| across channels) so stereo images stay intact.

inline void noise_gate(float** ch, int nch, size_t n, float sr, float threshold_db,
                       float attack_ms, float release_ms, float hold_ms, float range_db) {
    const float thr = db2lin(threshold_db);
    const float floor_gain = db2lin(-std::fabs(range_db));
    const float env_a = smooth_coef(1.0f, sr);  // fast envelope attack
    const float env_r = smooth_coef(std::max(release_ms * 0.5f, 10.0f), sr);
    const float open_c = smooth_coef(attack_ms, sr);
    const float close_c = smooth_coef(release_ms, sr);
    const long hold_samples = long(hold_ms * 0.001f * sr);

    float env = 0.0f, gain = 1.0f;
    long hold = 0;
    for (size_t i = 0; i < n; ++i) {
        float det = 0.0f;
        for (int c = 0; c < nch; ++c) det = std::max(det, std::fabs(ch[c][i]));
        env = det > env ? env_a * env + (1 - env_a) * det
                        : env_r * env + (1 - env_r) * det;
        float target;
        if (env >= thr) {
            hold = hold_samples;
            target = 1.0f;
        } else if (hold > 0) {
            --hold;
            target = 1.0f;
        } else {
            target = floor_gain;
        }
        const float coef = target > gain ? open_c : close_c;
        gain = coef * gain + (1 - coef) * target;
        for (int c = 0; c < nch; ++c) ch[c][i] *= gain;
    }
}

// Feed-forward soft-knee compressor (Giannoulis et al. design), gain computer in dB domain.
inline void compressor(float** ch, int nch, size_t n, float sr, float threshold_db,
                       float ratio, float attack_ms, float release_ms, float knee_db,
                       float makeup_db) {
    const float aA = smooth_coef(attack_ms, sr);
    const float aR = smooth_coef(release_ms, sr);
    const float T = threshold_db;
    const float W = std::max(knee_db, 0.01f);
    const float R = std::max(ratio, 1.0f);
    const float makeup = db2lin(makeup_db);

    float gr = 0.0f;  // smoothed gain reduction in dB (<= 0)
    for (size_t i = 0; i < n; ++i) {
        float det = 0.0f;
        for (int c = 0; c < nch; ++c) det = std::max(det, std::fabs(ch[c][i]));
        const float xg = lin2db(det);
        float yg;
        if (2 * (xg - T) < -W) {
            yg = xg;
        } else if (2 * std::fabs(xg - T) <= W) {
            const float d = xg - T + W / 2;
            yg = xg + (1.0f / R - 1.0f) * d * d / (2 * W);
        } else {
            yg = T + (xg - T) / R;
        }
        const float g = yg - xg;  // <= 0
        gr = g < gr ? aA * gr + (1 - aA) * g : aR * gr + (1 - aR) * g;
        const float lin = db2lin(gr) * makeup;
        for (int c = 0; c < nch; ++c) ch[c][i] *= lin;
    }
}

// Offline look-ahead brickwall limiter. Guaranteed no sample above the ceiling:
// required gain -> forward sliding-window minimum over the lookahead -> trailing
// moving average (smooth attack ramp that stays <= the required gain at each peak)
// -> slow-rise release.
inline void limiter(float** ch, int nch, size_t n, float sr, float ceiling_db,
                    float release_ms, float lookahead_ms) {
    if (n == 0) return;
    const float c = db2lin(ceiling_db);
    const size_t L = std::max<size_t>(1, size_t(lookahead_ms * 0.001f * sr));

    std::vector<float> r(n);
    for (size_t i = 0; i < n; ++i) {
        float det = 0.0f;
        for (int cc = 0; cc < nch; ++cc) det = std::max(det, std::fabs(ch[cc][i]));
        r[i] = det > c ? c / det : 1.0f;
    }

    // Forward sliding-window minimum over [i, i+L].
    std::vector<float> m(n);
    std::deque<size_t> dq;
    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t hi = std::min(n - 1, i + L);
        while (j <= hi) {
            while (!dq.empty() && r[dq.back()] >= r[j]) dq.pop_back();
            dq.push_back(j);
            ++j;
        }
        while (dq.front() < i) dq.pop_front();
        m[i] = r[dq.front()];
    }

    // Trailing moving average of length L: smooth attack that never exceeds the
    // required gain at the peak sample itself.
    std::vector<double> ps(n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) ps[i + 1] = ps[i] + m[i];
    std::vector<float> ma(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t lo = i + 1 >= L ? i + 1 - L : 0;
        ma[i] = float((ps[i + 1] - ps[lo]) / double(i + 1 - lo));
    }

    const float aR = smooth_coef(release_ms, sr);
    float g = 1.0f;
    for (size_t i = 0; i < n; ++i) {
        g = ma[i] < g ? ma[i] : aR * g + (1 - aR) * ma[i];
        if (g > 1.0f) g = 1.0f;
        for (int cc = 0; cc < nch; ++cc) {
            float v = ch[cc][i] * g;
            if (v > c) v = c;
            if (v < -c) v = -c;
            ch[cc][i] = v;
        }
    }
}

inline void apply_gain(float** ch, int nch, size_t n, float gain_db) {
    const float g = db2lin(gain_db);
    for (int c = 0; c < nch; ++c)
        for (size_t i = 0; i < n; ++i) ch[c][i] *= g;
}

}  // namespace ait
