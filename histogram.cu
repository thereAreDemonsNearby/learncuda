#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "common.h"

std::vector<int> histogram_CPU(std::vector<char> const& text)
{
    std::vector<int> histo(7);
    for (auto c : text) {
        int ic = c - 'a';
        if (ic >=0 && ic < 26) {
            ++histo[ic/4];
        }
    }

    return histo;
}

template <typename Kernel>
std::vector<int> histogram_GPU(std::vector<char> const& text);

__global__
void histogram_GPU_kernel_trivial(char const* __restrict__ input, int* histo, int sz, int numBins)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int j = i; j < sz; j += blockDim.x * gridDim.x) {
        int c = input[j] - 'a';
        if (c >= 0 && c < 26) {
            atomicAdd(&(histo[c/4]), 1);
        }
    }
}

struct HistogramKernel_trivial
{
    void operator()(char const* input, int* histo, int sz, int numBins)
    {
        histogram_GPU_kernel_trivial<<<std::min(1024, (sz+1023)/1024), 1024>>>(input, histo, sz, numBins);
    }
};

__global__
void histogram_GPU_kernel_privatized(char const* __restrict__ input, int* histo, int sz, int numBins)
{
    extern __shared__ int histo_s[];
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int binIdx = threadIdx.x; binIdx < numBins; binIdx += blockDim.x) {
        histo_s[binIdx] = 0;
    }
    __syncthreads();

    for (int j = i; j < sz; j += blockDim.x * gridDim.x) {
        int c = input[j] - 'a';
        if (c >= 0 && c < 26) {
            atomicAdd(&histo_s[c/4], 1);
        }
    }
    __syncthreads();

    for (int binIdx = threadIdx.x; binIdx < numBins; binIdx += blockDim.x) {
        atomicAdd(&histo[binIdx], histo_s[binIdx]);
    }
}

struct HistogramKernel_privatized
{
    void operator()(char const* input, int* histo, int sz, int numBins)
    {
        histogram_GPU_kernel_privatized<<<std::min(1024, (sz+1023)/1024), 1024, numBins>>>(input, histo, sz, numBins);
    }
};

bool vectorCompare(std::vector<int> const& a, std::vector<int> const& b)
{
    if (a.size() != b.size()) {
        fmt::print(stderr, "size error\n");
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            fmt::print(stderr, "[{0}] {1} != {2}\n", i, a[i], b[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    size_t N = 1000000;
    if (argc > 1) {
        N = std::stoul(argv[1]);
    }

    std::mt19937 eng(std::random_device{}());
    // std::uniform_int_distribution<char> distrib('a', 'z');
    std::normal_distribution<> distrib(13, 2);
    std::vector<char> text(N);
    for (auto& c : text) {
        c = std::lround(distrib(eng)) + 'a';
    }

    std::vector<int> histo1, histo2, histo3;

    {
        TimerGuard t("histogram cpu:");
        histo1 = histogram_CPU(text);
        doNotOptimize(histo1);
    }

    {
        TimerGuard t("histogram gpu trivial:");
        histo2 = histogram_GPU<HistogramKernel_trivial>(text);
        doNotOptimize(histo2);
    }

    {
        TimerGuard t("histogram gpu privatized:");
        histo3 = histogram_GPU<HistogramKernel_privatized>(text);
        doNotOptimize(histo3);
    }

    if (!vectorCompare(histo1, histo2)) {
        fmt::print(stderr, "1 2 error\n");
        for (auto i : histo1) {
            fmt::print(stderr, "{} ", i);
        }
        fmt::print(stderr, "\n");

        for (auto i : histo2) {
            fmt::print(stderr, "{} ", i);
        }
        fmt::print(stderr, "\n");
 
    }

    if (!vectorCompare(histo1, histo3)) {
        fmt::print(stderr, "1 3 error\n");
    }
}




template <typename Kernel>
std::vector<int> histogram_GPU(std::vector<char> const& text)
{
    std::vector<int> histo(7, 0);

    cudaError_t err;    
    char* d_text;
    int* d_histo;
    err = cudaMalloc((void**)&d_text, sizeof(char) * text.size());
    checkCudaError(err);
    
    err = cudaMalloc((void**)&d_histo, sizeof(int) * histo.size());
    checkCudaError(err);

    err = cudaMemcpy(d_text, text.data(), sizeof(char) * text.size(), cudaMemcpyHostToDevice);
    checkCudaError(err);

    // init to 0
    err = cudaMemcpy(d_histo, histo.data(), sizeof(int) * histo.size(), cudaMemcpyHostToDevice);
    checkCudaError(err);

    auto start = std::chrono::system_clock::now();
    Kernel k;
    k(d_text, d_histo, text.size(), histo.size());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print(stderr, "kernel time: {}\n", std::chrono::duration<double>(end-start).count());
    
    err = cudaGetLastError();
    checkCudaError(err);

    err = cudaMemcpy(histo.data(), d_histo, sizeof(int) * histo.size(), cudaMemcpyDeviceToHost);
    checkCudaError(err);

    return histo;
}
