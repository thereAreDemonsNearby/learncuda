#include <fmt/core.h>
#include <vector>
#include <cstddef>
#include <chrono>
#include <random>
#include "TimerGuard.h"

template <typename T>
inline void doNotOptimize(T const& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

constexpr size_t MaxMaskLen = 9;
__constant__ float Mask[MaxMaskLen]; // cuda constant memory

bool vectorCompare(std::vector<float> const& a, std::vector<float> const& b)
{
    if (a.size() != b.size()) {
        fmt::print(stderr, "size not equal\n");
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 0.01) {
            fmt::print(stderr, "a[{0}] = {1}, b[{0}] = {2}\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

std::vector<float> convolution1D_CPU(std::vector<float> const& a, std::vector<float> const& mask)
{
    // for checking correctness
    std::vector<float> res(a.size());
    for (size_t i = 0; i < res.size(); ++i) {
        res[i] = 0.0f;
        for (size_t j = 0; j < mask.size(); ++j) {
            int h = i + j - mask.size() / 2;
            res[i] += (h >= 0 && h < a.size() ? a[h] : 0.0f) * mask[j];
        }
    }

    return res;
}

std::vector<float> convolution1D_GPU(std::vector<float> const& a, std::vector<float> const& mask);
std::vector<float> convolution1D_GPU_constantMem(std::vector<float> const& a, std::vector<float> const& mask);
template <int BlockWidth>
std::vector<float> convolution1D_GPU_constantMem_sharedMem(std::vector<float> const& a, std::vector<float> const& mask);

int main()
{
    std::random_device rd;
    std::default_random_engine eg(rd());
    std::uniform_real_distribution<float> distrib;

    size_t N = 10000000;
    size_t M = 9;
    std::vector<float> a(N);
    std::vector<float> mask(M);

    for (auto& e : a) {
        e = distrib(eg);
    }

    for (auto& e : mask) {
        e = distrib(eg);
    }

    std::vector<float> res1, res2, res3, res4;

    {
        TimerGuard t("convolution1D cpu:");
        res1 = convolution1D_CPU(a, mask);
        doNotOptimize(res1);
    }

    {
        TimerGuard t("convolution1D gpu:");
        res2 = convolution1D_GPU(a, mask);
        doNotOptimize(res2);
    }

    {
        TimerGuard t("convolution1D gpu constant memory:");
        res3 = convolution1D_GPU_constantMem(a, mask);
        doNotOptimize(res3);
    }

    {
        TimerGuard t("convolution1D gpu constant memory & shared memory:");
        res4 = convolution1D_GPU_constantMem_sharedMem<1024>(a, mask);
        doNotOptimize(res4);
    }

    if (!vectorCompare(res1, res2)) {
        fmt::print(stderr, "wrong answer\n");
    }

    if (!vectorCompare(res1, res3)) {
        fmt::print(stderr, "wrong answer 1 3\n");
    }

    if (!vectorCompare(res1, res4)) {
        fmt::print(stderr, "wrong answer 1 4\n");
    }
}

__global__
void convolution1D_GPU_kernel(float const* a, float* p, float const* mask, int len, int masklen)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    
    float v = 0.0f;
    for (int j = 0; j < masklen; ++j) {
        int h = i + j - masklen / 2;
        v += (h >= 0 && h < len ? a[h] : 0.0f) * mask[j];
    }
    p[i] = v;
}

std::vector<float> convolution1D_GPU(std::vector<float> const& a, std::vector<float> const& mask)
{
    std::vector<float> result(a.size());
    
    cudaError_t err;
    float* d_a;
    float* d_m;
    float* d_r;

    if (err = cudaMalloc((void**)&d_a, sizeof(float) * a.size());
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMalloc failed: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMalloc((void**)&d_m, sizeof(float) * mask.size());
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMalloc failed: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMalloc((void**)&d_r, sizeof(float) * result.size());
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMalloc failed: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpy failed: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMemcpy(d_m, mask.data(), sizeof(float) * mask.size(), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpy failed: {}\n", cudaGetErrorString(err));
    }

    auto start = std::chrono::system_clock::now();
    convolution1D_GPU_kernel<<<ceil((double)a.size() / 1024), 1024>>>(d_a, d_r, d_m, a.size(), mask.size());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print("kernel only: {}s\n", std::chrono::duration<double>(end-start).count());

    if (err = cudaGetLastError();
        err != cudaSuccess) {
        fmt::print(stderr, "cuda kernel ran incorrectly: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMemcpy(result.data(), d_r, sizeof(float) * result.size(), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpy failed: {}\n", cudaGetErrorString(err));
    }

    cudaFree(d_a);
    cudaFree(d_m);
    cudaFree(d_r);

    return result;
}

__global__
void convolution1D_GPU_constantMem_kernel(float const* a, float* p, int len, int masklen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len)
        return;

    float pVal = 0.0f;
    int startingPoint = i - masklen / 2;
    for (int j = 0; j < masklen; ++j) {
        pVal += (startingPoint + j >= 0 && startingPoint + j < len ? a[startingPoint + j] : 0.0f) * Mask[j];
        // Mask : global constant 
    }

    p[i] = pVal;
}

std::vector<float> convolution1D_GPU_constantMem(std::vector<float> const& a, std::vector<float> const& mask)
{
    std::vector<float> result(a.size());
    
    cudaError_t err;
    float* d_a;
 
    float* d_r;

    if (err = cudaMalloc((void**)&d_a, sizeof(float) * a.size());
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMalloc failed: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMalloc((void**)&d_r, sizeof(float) * result.size());
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMalloc failed: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpy failed: {}\n", cudaGetErrorString(err));
    }

    // if (err = cudaMemcpy(d_m, mask.data(), sizeof(float) * mask.size(), cudaMemcpyHostToDevice);
    //     err != cudaSuccess) {
    //     fmt::print(stderr, "cudaMemcpy failed: {}\n", cudaGetErrorString(err));
    // }
    if (err = cudaMemcpyToSymbol(Mask, mask.data(), sizeof(float) * mask.size());
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpyToSymbol failed: {}\n", cudaGetErrorString(err));
    }

    auto start = std::chrono::system_clock::now();
    convolution1D_GPU_constantMem_kernel<<<ceil((double)a.size() / 1024), 1024>>>(d_a, d_r, a.size(), mask.size());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print("kernel only: {}s\n", std::chrono::duration<double>(end-start).count());

    if (err = cudaGetLastError();
        err != cudaSuccess) {
        fmt::print(stderr, "cuda kernel ran incorrectly: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMemcpy(result.data(), d_r, sizeof(float) * result.size(), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpy failed: {}\n", cudaGetErrorString(err));
    }

    cudaFree(d_a);
    cudaFree(d_r);

    return result;
}

template <int BlockWidth>
__global__
void convolution1D_GPU_constantMem_sharedMem_kernel(float const* a, float* p, int len, int masklen)
{
    __shared__ float a_tile[BlockWidth + MaxMaskLen - 1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i_prev = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    int i_next = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    // if (i >= len) {
    //     return;
    // }

    // inner cells
    a_tile[masklen / 2 + threadIdx.x] = i < len ? a[i] : 0;

    // left ghost cells
    if (threadIdx.x >= BlockWidth - masklen / 2) {
        if (i_prev >= 0) {
            a_tile[threadIdx.x - BlockWidth + masklen / 2] = a[i_prev];
        } else {
            a_tile[threadIdx.x - BlockWidth + masklen / 2] = 0.0f;
        }
    }

    // right ghost cells
    if (threadIdx.x < masklen / 2) {
        if (i_next < len) {
            a_tile[BlockWidth + masklen / 2 + threadIdx.x] = a[i_next];
        } else {
            a_tile[BlockWidth + masklen / 2 + threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

    float pVal = 0.0f;

    for (int j = 0; j < masklen; ++j) {
        pVal += a_tile[threadIdx.x + j] * Mask[j];
    }
    if (i < len)
        p[i] = pVal;
}

template <int BlockWidth>
std::vector<float> convolution1D_GPU_constantMem_sharedMem(std::vector<float> const& a, std::vector<float> const& mask)
{
    std::vector<float> result(a.size());
    
    cudaError_t err;
    float* d_a;
 
    float* d_r;

    if (err = cudaMalloc((void**)&d_a, sizeof(float) * a.size());
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMalloc failed: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMalloc((void**)&d_r, sizeof(float) * result.size());
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMalloc failed: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpy failed: {}\n", cudaGetErrorString(err));
    }

    // if (err = cudaMemcpy(d_m, mask.data(), sizeof(float) * mask.size(), cudaMemcpyHostToDevice);
    //     err != cudaSuccess) {
    //     fmt::print(stderr, "cudaMemcpy failed: {}\n", cudaGetErrorString(err));
    // }
    if (err = cudaMemcpyToSymbol(Mask, mask.data(), sizeof(float) * mask.size());
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpyToSymbol failed: {}\n", cudaGetErrorString(err));
    }

    auto start = std::chrono::system_clock::now();
    convolution1D_GPU_constantMem_sharedMem_kernel<BlockWidth><<<ceil((double)a.size() / BlockWidth), BlockWidth>>>(d_a, d_r, a.size(), mask.size());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print("kernel only: {}s\n", std::chrono::duration<double>(end-start).count());

    if (err = cudaGetLastError();
        err != cudaSuccess) {
        fmt::print(stderr, "cuda kernel ran incorrectly: {}\n", cudaGetErrorString(err));
    }

    if (err = cudaMemcpy(result.data(), d_r, sizeof(float) * result.size(), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        fmt::print(stderr, "cudaMemcpy failed: {}\n", cudaGetErrorString(err));
    }

    cudaFree(d_a);
    cudaFree(d_r);

    return result;    
}
