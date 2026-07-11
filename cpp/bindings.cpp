#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <tuple>
#include <vector>

#include "biquad.hpp"
#include "dynamics.hpp"

namespace py = pybind11;
using Arr = py::array_t<float, py::array::c_style | py::array::forcecast>;
// (type, freq_hz, gain_db, q)
using BandSpec = std::tuple<std::string, double, double, double>;

namespace {

struct Buffer {
    Arr out;
    std::vector<float*> ch;
    size_t n = 0;
};

// Copy the (channels, n) input into a fresh output array and collect channel pointers.
Buffer make_buffer(const Arr& x) {
    if (x.ndim() != 2) throw std::invalid_argument("expected a 2D array of shape (channels, n)");
    Buffer b;
    b.out = Arr(std::vector<py::ssize_t>{x.shape(0), x.shape(1)});
    std::memcpy(b.out.mutable_data(), x.data(), size_t(x.size()) * sizeof(float));
    b.n = size_t(x.shape(1));
    for (py::ssize_t c = 0; c < x.shape(0); ++c) b.ch.push_back(b.out.mutable_data(c, 0));
    return b;
}

}  // namespace

PYBIND11_MODULE(_dsp, m) {
    m.doc() = "Chat with Audio native DSP core";

    m.def("backend_info", [] { return std::string("native"); });

    m.def("apply_gain", [](Arr x, float gain_db) {
        Buffer b = make_buffer(x);
        {
            py::gil_scoped_release rel;
            ait::apply_gain(b.ch.data(), int(b.ch.size()), b.n, gain_db);
        }
        return b.out;
    }, py::arg("x"), py::arg("gain_db"));

    m.def("biquad_chain", [](Arr x, float sr, std::vector<BandSpec> bands) {
        // Validate designs before touching audio so errors surface with the bad band.
        std::vector<ait::BiquadCoeffs> coeffs;
        for (const auto& band : bands)
            coeffs.push_back(ait::design_biquad(std::get<0>(band), sr, std::get<1>(band),
                                                std::get<2>(band), std::get<3>(band)));
        Buffer b = make_buffer(x);
        {
            py::gil_scoped_release rel;
            for (size_t c = 0; c < b.ch.size(); ++c) {
                for (const auto& co : coeffs) {
                    ait::Biquad bq(co);
                    bq.process_block(b.ch[c], b.n);
                }
            }
        }
        return b.out;
    }, py::arg("x"), py::arg("sr"), py::arg("bands"));

    m.def("noise_gate", [](Arr x, float sr, float threshold_db, float attack_ms,
                           float release_ms, float hold_ms, float range_db) {
        Buffer b = make_buffer(x);
        {
            py::gil_scoped_release rel;
            ait::noise_gate(b.ch.data(), int(b.ch.size()), b.n, sr, threshold_db, attack_ms,
                            release_ms, hold_ms, range_db);
        }
        return b.out;
    }, py::arg("x"), py::arg("sr"), py::arg("threshold_db"), py::arg("attack_ms") = 5.0f,
       py::arg("release_ms") = 120.0f, py::arg("hold_ms") = 50.0f, py::arg("range_db") = 12.0f);

    m.def("compressor", [](Arr x, float sr, float threshold_db, float ratio, float attack_ms,
                           float release_ms, float knee_db, float makeup_db) {
        Buffer b = make_buffer(x);
        {
            py::gil_scoped_release rel;
            ait::compressor(b.ch.data(), int(b.ch.size()), b.n, sr, threshold_db, ratio,
                            attack_ms, release_ms, knee_db, makeup_db);
        }
        return b.out;
    }, py::arg("x"), py::arg("sr"), py::arg("threshold_db"), py::arg("ratio") = 3.0f,
       py::arg("attack_ms") = 10.0f, py::arg("release_ms") = 150.0f, py::arg("knee_db") = 6.0f,
       py::arg("makeup_db") = 0.0f);

    m.def("limiter", [](Arr x, float sr, float ceiling_db, float release_ms, float lookahead_ms) {
        Buffer b = make_buffer(x);
        {
            py::gil_scoped_release rel;
            ait::limiter(b.ch.data(), int(b.ch.size()), b.n, sr, ceiling_db, release_ms,
                         lookahead_ms);
        }
        return b.out;
    }, py::arg("x"), py::arg("sr"), py::arg("ceiling_db") = -1.5f, py::arg("release_ms") = 60.0f,
       py::arg("lookahead_ms") = 5.0f);
}
