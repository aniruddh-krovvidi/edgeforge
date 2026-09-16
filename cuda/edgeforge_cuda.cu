// edgeforge CUDA backend: the int8 conv / linear kernels for an NVIDIA target,
// plus a parity harness that runs the same .efm graph on the CPU runtime and
// on the GPU and compares logits.
//
// Build (needs an NVIDIA GPU + CUDA toolkit):   make edgeforge_cuda
// Run:   ./edgeforge_cuda data/model_int8.efm data/mnist_test.bin [n_images]
//
// NOTE: this file was written on a machine without an NVIDIA GPU and is
// compiled/validated separately (see README "CUDA status"). It reuses the CPU
// runtime as the reference by including it with its main() compiled out.
#define EF_NO_MAIN
#include "../runtime.cpp"

#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while (0)

// Round-to-nearest-even requantize, same contract as the CPU runtime.
__device__ __forceinline__ int8_t requant_dev(float y, float s_out) {
    int v = __float2int_rn(y / s_out);
    return (int8_t)max(-128, min(127, v));
}

// One thread per output element (oc, oy, ox). int8 x int8 -> int32 accumulate,
// per-output-channel float multiplier, fused requant (or float emit for the last layer).
// Weights are cached in shared memory per block when they fit (small edge convs do).
__global__ void conv_i8_kernel(const int8_t* __restrict__ x, const int8_t* __restrict__ w,
                               const int32_t* __restrict__ bias, const float* __restrict__ s_w,
                               int C, int H, int W, int OC, int K, int stride, int pad,
                               int OH, int OW, float s_out, int8_t* __restrict__ yq, float* __restrict__ yf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OC * OH * OW;
    if (idx >= total) return;
    int ox = idx % OW, oy = (idx / OW) % OH, oc = idx / (OW * OH);
    int n = C * K * K;
    const int8_t* wo = w + (size_t)oc * n;
    int32_t acc = bias[oc];
    for (int ic = 0; ic < C; ic++)
        for (int kh = 0; kh < K; kh++) {
            int iy = oy * stride + kh - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kw = 0; kw < K; kw++) {
                int ix = ox * stride + kw - pad;
                if (ix < 0 || ix >= W) continue;
                acc += (int)x[((size_t)ic * H + iy) * W + ix] * (int)wo[(ic * K + kh) * K + kw];
            }
        }
    float v = (float)acc * s_w[oc];
    if (s_out == 0.f) yf[idx] = v; else yq[idx] = requant_dev(v, s_out);
}

// One warp per output feature: lanes stride the input, warp-reduce the int32 sum.
__global__ void linear_i8_kernel(const int8_t* __restrict__ x, const int8_t* __restrict__ w,
                                 const int32_t* __restrict__ bias, const float* __restrict__ s_w,
                                 int IN, int OUT, float s_out, int8_t* __restrict__ yq, float* __restrict__ yf) {
    int o = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (o >= OUT) return;
    const int8_t* wo = w + (size_t)o * IN;
    int32_t acc = 0;
    for (int i = lane; i < IN; i += 32) acc += (int)wo[i] * (int)x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) {
        float v = (float)(acc + bias[o]) * s_w[o];
        if (s_out == 0.f) yf[o] = v; else yq[o] = requant_dev(v, s_out);
    }
}

__global__ void relu_i8_kernel(int8_t* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0) x[i] = 0;
}

__global__ void maxpool_i8_kernel(const int8_t* x, int8_t* y, int C, int H, int W, int K) {
    int OH = H / K, OW = W / K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * OH * OW) return;
    int ox = idx % OW, oy = (idx / OW) % OH, c = idx / (OW * OH);
    int8_t m = -128;
    for (int dy = 0; dy < K; dy++)
        for (int dx = 0; dx < K; dx++)
            m = max(m, x[((size_t)c * H + oy * K + dy) * W + ox * K + dx]);
    y[idx] = m;
}

// Device-resident copy of the int8 graph.
struct DevLayer { int8_t* w = nullptr; int32_t* b = nullptr; float* s_w = nullptr; };

static std::vector<float> forward_cuda(const ef::Model& m, const std::vector<DevLayer>& dl, const float* input) {
    uint32_t c = m.in_c, h = m.in_h, w = m.in_w;
    size_t n_in = size_t(c) * h * w;
    std::vector<int8_t> hq(n_in);
    for (size_t i = 0; i < n_in; i++) hq[i] = ef::requant(input[i], m.s_in);
    int8_t *dx, *dy; float* df;
    CK(cudaMalloc(&dx, 1 << 16)); CK(cudaMalloc(&dy, 1 << 16)); CK(cudaMalloc(&df, 4096 * sizeof(float)));
    CK(cudaMemcpy(dx, hq.data(), n_in, cudaMemcpyHostToDevice));
    std::vector<float> logits;
    for (size_t li = 0; li < m.layers.size(); li++) {
        const ef::Layer& L = m.layers[li];
        const DevLayer& D = dl[li];
        if (L.op == ef::CONV) {
            uint32_t oh = (h + 2 * L.pad - L.k) / L.stride + 1, ow = (w + 2 * L.pad - L.k) / L.stride + 1;
            int total = int(L.out_c * oh * ow);
            conv_i8_kernel<<<(total + 255) / 256, 256>>>(dx, D.w, D.b, D.s_w, c, h, w, L.out_c, L.k, L.stride, L.pad,
                                                          oh, ow, L.s_out, dy, df);
            c = L.out_c; h = oh; w = ow;
            if (L.s_out == 0.f) { logits.resize(total); CK(cudaMemcpy(logits.data(), df, total * 4, cudaMemcpyDeviceToHost)); break; }
            std::swap(dx, dy);
        } else if (L.op == ef::LINEAR) {
            int warps_per_block = 8;
            linear_i8_kernel<<<(L.out_f + warps_per_block - 1) / warps_per_block, 32 * warps_per_block>>>(
                dx, D.w, D.b, D.s_w, L.in_f, L.out_f, L.s_out, dy, df);
            c = L.out_f; h = w = 1;
            if (L.s_out == 0.f) { logits.resize(L.out_f); CK(cudaMemcpy(logits.data(), df, L.out_f * 4, cudaMemcpyDeviceToHost)); break; }
            std::swap(dx, dy);
        } else if (L.op == ef::RELU) {
            int n = int(c * h * w);
            relu_i8_kernel<<<(n + 255) / 256, 256>>>(dx, n);
        } else if (L.op == ef::POOL) {
            int n = int(c * (h / L.k) * (w / L.k));
            maxpool_i8_kernel<<<(n + 255) / 256, 256>>>(dx, dy, c, h, w, L.k);
            h /= L.k; w /= L.k; std::swap(dx, dy);
        } else if (L.op == ef::FLATTEN) { c = c * h * w; h = w = 1; }
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    cudaFree(dx); cudaFree(dy); cudaFree(df);
    return logits;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: edgeforge_cuda model_int8.efm mnist_test.bin [n]\n"); return 2; }
    ef::Model m = ef::load(argv[1]);
    if (m.dtype != 1) { std::fprintf(stderr, "CUDA backend is int8-only\n"); return 2; }
    TestSet t = load_mnist(argv[2]);
    int n = argc > 3 ? std::atoi(argv[3]) : 100;

    std::vector<DevLayer> dl(m.layers.size());
    for (size_t i = 0; i < m.layers.size(); i++) {
        const ef::Layer& L = m.layers[i];
        if (L.op != ef::CONV && L.op != ef::LINEAR) continue;
        CK(cudaMalloc(&dl[i].w, L.w_q.size())); CK(cudaMemcpy(dl[i].w, L.w_q.data(), L.w_q.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dl[i].b, L.b_q.size() * 4)); CK(cudaMemcpy(dl[i].b, L.b_q.data(), L.b_q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dl[i].s_w, L.s_w.size() * 4)); CK(cudaMemcpy(dl[i].s_w, L.s_w.data(), L.s_w.size() * 4, cudaMemcpyHostToDevice));
    }

    std::vector<float> x(784);
    float worst = 0.f; int agree = 0;
    for (int i = 0; i < n; i++) {
        normalize(&t.pixels[size_t(i) * 784], x.data());
        auto cpu = ef::forward(m, x.data());
        auto gpu = forward_cuda(m, dl, x.data());
        for (size_t j = 0; j < cpu.size(); j++) worst = std::max(worst, std::fabs(cpu[j] - gpu[j]));
        agree += (std::max_element(cpu.begin(), cpu.end()) - cpu.begin()) == (std::max_element(gpu.begin(), gpu.end()) - gpu.begin());
    }
    std::printf("cuda vs cpu int8: %d images, max |diff| = %.2e, argmax agreement %d/%d -> %s\n",
                n, worst, agree, n, (worst < 1e-3f && agree == n) ? "PARITY OK" : "PARITY FAILED");
    return (worst < 1e-3f && agree == n) ? 0 : 1;
}
