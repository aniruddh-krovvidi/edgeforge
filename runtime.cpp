// edgeforge runtime: executes a lowered .efm graph (fp32 or int8) on the CPU.
//
// Single translation unit, C++17, no dependencies. The int8 path is the one
// that matters for edge deployment: int8 weights and activations, int32
// accumulation, per-output-channel weight scales, requantization fused with
// ReLU, and a NEON dot-product kernel on arm64 (scalar fallback elsewhere).
//
//   edgeforge eval  model.efm mnist_test.bin        accuracy over the test set
//   edgeforge bench model.efm mnist_test.bin [iters] per-image latency
//   edgeforge dump  model.efm mnist_test.bin index  print logits for one image (parity tests)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define EF_NEON 1
#endif

namespace ef {

enum Op : uint32_t { CONV = 1, RELU = 2, POOL = 3, LINEAR = 4, FLATTEN = 5 };

struct Layer {
    Op op;
    uint32_t in_c = 0, out_c = 0, k = 0, stride = 1, pad = 0;   // conv
    uint32_t in_f = 0, out_f = 0;                               // linear
    float s_out = 0;                                            // 0 => emit float
    std::vector<float> s_w, w_f, b_f;
    std::vector<int8_t> w_q;
    std::vector<int32_t> b_q;
};

struct Model {
    uint32_t dtype = 0;           // 0 = fp32, 1 = int8
    uint32_t in_c = 0, in_h = 0, in_w = 0;
    float s_in = 1.f;
    std::vector<Layer> layers;
};

// Activations: exactly one of f / q is live, plus shape.
struct Tensor {
    std::vector<float> f;
    std::vector<int8_t> q;
    uint32_t c = 0, h = 0, w = 0;
    size_t size() const { return size_t(c) * h * w; }
};

// ---------------------------------------------------------------- loading

template <class T> static void rd(std::ifstream& in, T* dst, size_t n) {
    in.read(reinterpret_cast<char*>(dst), std::streamsize(n * sizeof(T)));
    if (!in) throw std::runtime_error("truncated .efm");
}
template <class T> static T rd1(std::ifstream& in) { T v; rd(in, &v, 1); return v; }

Model load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[4]; rd(in, magic, 4);
    if (std::memcmp(magic, "EFM1", 4) != 0) throw std::runtime_error("bad magic");
    Model m;
    m.dtype = rd1<uint32_t>(in);
    uint32_t n = rd1<uint32_t>(in);
    m.in_c = rd1<uint32_t>(in); m.in_h = rd1<uint32_t>(in); m.in_w = rd1<uint32_t>(in);
    m.s_in = rd1<float>(in);
    auto read_weighted = [&](Layer& L, size_t n_w, size_t n_out) {
        L.s_out = rd1<float>(in);
        L.s_w.resize(n_out); rd(in, L.s_w.data(), n_out);
        if (m.dtype == 0) { L.w_f.resize(n_w); rd(in, L.w_f.data(), n_w); L.b_f.resize(n_out); rd(in, L.b_f.data(), n_out); }
        else              { L.w_q.resize(n_w); rd(in, L.w_q.data(), n_w); L.b_q.resize(n_out); rd(in, L.b_q.data(), n_out); }
    };
    for (uint32_t i = 0; i < n; i++) {
        Layer L; L.op = Op(rd1<uint32_t>(in));
        switch (L.op) {
            case CONV:
                L.in_c = rd1<uint32_t>(in); L.out_c = rd1<uint32_t>(in); L.k = rd1<uint32_t>(in);
                L.stride = rd1<uint32_t>(in); L.pad = rd1<uint32_t>(in);
                read_weighted(L, size_t(L.out_c) * L.in_c * L.k * L.k, L.out_c);
                break;
            case LINEAR:
                L.in_f = rd1<uint32_t>(in); L.out_f = rd1<uint32_t>(in);
                read_weighted(L, size_t(L.out_f) * L.in_f, L.out_f);
                break;
            case POOL: L.k = rd1<uint32_t>(in); break;
            case RELU: case FLATTEN: break;
            default: throw std::runtime_error("unknown op");
        }
        m.layers.push_back(std::move(L));
    }
    return m;
}

// ---------------------------------------------------------------- kernels

// int8 dot product with int32 accumulation. NEON: widen 8 lanes at a time,
// accumulate pairwise into int32x4 -- no overflow for n < 2^23.
static inline int32_t dot_i8(const int8_t* a, const int8_t* b, uint32_t n) {
    int32_t acc = 0;
    uint32_t i = 0;
#ifdef EF_NEON
    int32x4_t vacc = vdupq_n_s32(0);
    for (; i + 16 <= n; i += 16) {
        int8x16_t va = vld1q_s8(a + i), vb = vld1q_s8(b + i);
        vacc = vpadalq_s16(vacc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
        vacc = vpadalq_s16(vacc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
    }
    acc = vaddvq_s32(vacc);
#endif
    for (; i < n; i++) acc += int32_t(a[i]) * int32_t(b[i]);
    return acc;
}

static inline float dot_f32(const float* a, const float* b, uint32_t n) {
    float acc = 0.f;
    uint32_t i = 0;
#ifdef EF_NEON
    float32x4_t v0 = vdupq_n_f32(0.f), v1 = vdupq_n_f32(0.f);   // two accumulators hide FMA latency
    for (; i + 8 <= n; i += 8) {
        v0 = vfmaq_f32(v0, vld1q_f32(a + i), vld1q_f32(b + i));
        v1 = vfmaq_f32(v1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    acc = vaddvq_f32(vaddq_f32(v0, v1));
#endif
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
}

// requantize: float -> int8 with round-to-nearest-even (matches numpy.rint)
static inline int8_t requant(float y, float s_out) {
    long v = std::lrintf(y / s_out);
    return int8_t(std::clamp(v, -128L, 127L));
}

// Gather the (in_c*k*k) receptive field for output pixel (oy, ox) into `patch`
// (zero padded). Same element order as the weight layout [ic][kh][kw].
template <class T>
static inline void gather(const T* x, uint32_t C, uint32_t H, uint32_t W, const Layer& L,
                          uint32_t oy, uint32_t ox, T* patch) {
    uint32_t n = 0;
    for (uint32_t ic = 0; ic < C; ic++)
        for (uint32_t kh = 0; kh < L.k; kh++) {
            int iy = int(oy * L.stride + kh) - int(L.pad);
            for (uint32_t kw = 0; kw < L.k; kw++) {
                int ix = int(ox * L.stride + kw) - int(L.pad);
                patch[n++] = (iy < 0 || iy >= int(H) || ix < 0 || ix >= int(W)) ? T(0) : x[(size_t(ic) * H + iy) * W + ix];
            }
        }
}

static Tensor conv(const Tensor& x, const Layer& L, uint32_t dtype) {
    Tensor y; y.c = L.out_c;
    y.h = (x.h + 2 * L.pad - L.k) / L.stride + 1;
    y.w = (x.w + 2 * L.pad - L.k) / L.stride + 1;
    const uint32_t n = x.c * L.k * L.k;
    if (dtype == 0) {
        std::vector<float> patch(n);
        y.f.resize(y.size());
        for (uint32_t oy = 0; oy < y.h; oy++)
            for (uint32_t ox = 0; ox < y.w; ox++) {
                gather(x.f.data(), x.c, x.h, x.w, L, oy, ox, patch.data());
                for (uint32_t oc = 0; oc < L.out_c; oc++)
                    y.f[(size_t(oc) * y.h + oy) * y.w + ox] = dot_f32(&L.w_f[size_t(oc) * n], patch.data(), n) + L.b_f[oc];
            }
    } else {
        std::vector<int8_t> patch(n);
        const bool emit_float = L.s_out == 0.f;
        if (emit_float) y.f.resize(y.size()); else y.q.resize(y.size());
        for (uint32_t oy = 0; oy < y.h; oy++)
            for (uint32_t ox = 0; ox < y.w; ox++) {
                gather(x.q.data(), x.c, x.h, x.w, L, oy, ox, patch.data());
                for (uint32_t oc = 0; oc < L.out_c; oc++) {
                    int32_t acc = dot_i8(&L.w_q[size_t(oc) * n], patch.data(), n) + L.b_q[oc];
                    float v = float(acc) * L.s_w[oc];                 // s_w already includes s_x (see lower.py)
                    size_t o = (size_t(oc) * y.h + oy) * y.w + ox;
                    if (emit_float) y.f[o] = v; else y.q[o] = requant(v, L.s_out);
                }
            }
    }
    return y;
}

static Tensor linear(const Tensor& x, const Layer& L, uint32_t dtype) {
    Tensor y; y.c = L.out_f; y.h = 1; y.w = 1;
    if (dtype == 0) {
        y.f.resize(L.out_f);
        for (uint32_t o = 0; o < L.out_f; o++) y.f[o] = dot_f32(&L.w_f[size_t(o) * L.in_f], x.f.data(), L.in_f) + L.b_f[o];
    } else {
        const bool emit_float = L.s_out == 0.f;
        if (emit_float) y.f.resize(L.out_f); else y.q.resize(L.out_f);
        for (uint32_t o = 0; o < L.out_f; o++) {
            int32_t acc = dot_i8(&L.w_q[size_t(o) * L.in_f], x.q.data(), L.in_f) + L.b_q[o];
            float v = float(acc) * L.s_w[o];
            if (emit_float) y.f[o] = v; else y.q[o] = requant(v, L.s_out);
        }
    }
    return y;
}

template <class T> static void relu(std::vector<T>& v) { for (auto& e : v) if (e < T(0)) e = T(0); }

template <class T> static std::vector<T> pool(const std::vector<T>& x, uint32_t c, uint32_t h, uint32_t w, uint32_t k) {
    uint32_t oh = h / k, ow = w / k;
    std::vector<T> y(size_t(c) * oh * ow);
    for (uint32_t ch = 0; ch < c; ch++)
        for (uint32_t oy = 0; oy < oh; oy++)
            for (uint32_t ox = 0; ox < ow; ox++) {
                T m = x[(size_t(ch) * h + oy * k) * w + ox * k];
                for (uint32_t dy = 0; dy < k; dy++)
                    for (uint32_t dx = 0; dx < k; dx++)
                        m = std::max(m, x[(size_t(ch) * h + oy * k + dy) * w + ox * k + dx]);
                y[(size_t(ch) * oh + oy) * ow + ox] = m;
            }
    return y;
}

// ---------------------------------------------------------------- forward

std::vector<float> forward(const Model& m, const float* input) {
    Tensor x; x.c = m.in_c; x.h = m.in_h; x.w = m.in_w;
    if (m.dtype == 0) x.f.assign(input, input + x.size());
    else { x.q.resize(x.size()); for (size_t i = 0; i < x.size(); i++) x.q[i] = requant(input[i], m.s_in); }

    for (const Layer& L : m.layers) {
        switch (L.op) {
            case CONV:   x = conv(x, L, m.dtype); break;
            case LINEAR: x = linear(x, L, m.dtype); break;
            case RELU:   if (!x.f.empty()) relu(x.f); else relu(x.q); break;
            case POOL:
                if (!x.f.empty()) x.f = pool(x.f, x.c, x.h, x.w, L.k); else x.q = pool(x.q, x.c, x.h, x.w, L.k);
                x.h /= L.k; x.w /= L.k; break;
            case FLATTEN: x.c = uint32_t(x.size()); x.h = x.w = 1; break;
        }
    }
    return x.f;   // last layer always emits float logits
}

}  // namespace ef

// ---------------------------------------------------------------- CLI

struct TestSet { std::vector<uint8_t> labels; std::vector<uint8_t> pixels; uint32_t n = 0; };

static TestSet load_mnist(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    TestSet t; ef::rd(in, &t.n, 1);
    t.labels.resize(t.n); t.pixels.resize(size_t(t.n) * 784);
    for (uint32_t i = 0; i < t.n; i++) { ef::rd(in, &t.labels[i], 1); ef::rd(in, &t.pixels[size_t(i) * 784], 784); }
    return t;
}

static void normalize(const uint8_t* px, float* out) {   // same as torchvision Normalize((0.1307,), (0.3081,))
    for (int i = 0; i < 784; i++) out[i] = (px[i] / 255.f - 0.1307f) / 0.3081f;
}

#ifndef EF_NO_MAIN
int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: edgeforge eval|bench|dump model.efm mnist_test.bin [arg]\n"); return 2; }
    std::string cmd = argv[1];
    ef::Model m = ef::load(argv[2]);
    TestSet t = load_mnist(argv[3]);
    std::vector<float> x(784);

    if (cmd == "eval") {
        uint32_t ok = 0;
        for (uint32_t i = 0; i < t.n; i++) {
            normalize(&t.pixels[size_t(i) * 784], x.data());
            auto y = ef::forward(m, x.data());
            if (std::max_element(y.begin(), y.end()) - y.begin() == t.labels[i]) ok++;
        }
        std::printf("%s: accuracy %.4f (%u/%u)\n", m.dtype ? "int8" : "fp32", double(ok) / t.n, ok, t.n);
    } else if (cmd == "bench") {
        int iters = argc > 4 ? std::atoi(argv[4]) : 5000;
        std::vector<double> us; us.reserve(iters);
        for (int i = 0; i < iters; i++) {
            normalize(&t.pixels[size_t(i % t.n) * 784], x.data());
            auto t0 = std::chrono::steady_clock::now();
            auto y = ef::forward(m, x.data());
            auto t1 = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() + (y[0] == 12345.f ? 1 : 0));
        }
        std::sort(us.begin(), us.end());
        double mean = std::accumulate(us.begin(), us.end(), 0.0) / us.size();
        std::printf("%s: %d iters  mean %.1f us  p50 %.1f us  p99 %.1f us\n", m.dtype ? "int8" : "fp32", iters,
                    mean, us[us.size() / 2], us[size_t(us.size() * 0.99)]);
    } else if (cmd == "dump") {
        uint32_t i = argc > 4 ? uint32_t(std::atoi(argv[4])) : 0;
        normalize(&t.pixels[size_t(i) * 784], x.data());
        for (float v : ef::forward(m, x.data())) std::printf("%.6f ", v);
        std::printf("\n");
    } else { std::fprintf(stderr, "unknown command\n"); return 2; }
    return 0;
}
#endif  // EF_NO_MAIN
