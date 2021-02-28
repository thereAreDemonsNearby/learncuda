#include <fmt/core.h>
#include <cstddef>
#include <vector>
#include <chrono>
#include <random>
#include "TimerGuard.h"

#define checkCudaError(err)                                             \
    if (err != cudaSuccess) {                                           \
        fmt::print(stderr, "{0} at line {1}\n", cudaGetErrorString(err), __LINE__); \
    }

template <typename T>
inline void doNotOptimize(T const& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

template <typename T>
bool vectorCompare(std::vector<T> const& a, std::vector<T> const& b)
{
    if (a.size() != b.size()) {
        fmt::print(stderr, "size not equal\n");
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 0.1) {
            fmt::print(stderr, "a[{0}] = {1}, b[{0}] = {2}\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

template <typename T>
std::vector<T> scan_CPU(std::vector<T> const& in) // inclusive scan
{
    std::vector<T> result = in;
    for (size_t i = 1; i < result.size(); ++i) {
        result[i] += result[i - 1];
    }
    return result;
}

template <typename T, size_t SectionSize>
__global__
void kogge_Stone_scanKernel(T const* in, T* out, T* s, size_t sz)
{
    // blockDim.x == SectionSize
    // double buffering must be used to ensure correctness
    __shared__ T section[SectionSize * 2];
    
    T* pin = &section[0];
    T* pout = &section[SectionSize];
    
    int i = SectionSize * blockIdx.x + threadIdx.x;

    if (i < sz) {
        pin[threadIdx.x] = in[i];
    }
    // section[threadIdx.x + blockDim.x] = i + blockDim.x < sz ? in[i + blockDim.x];
    // __syncthreads(); // unnecessary
        
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (threadIdx.x >= stride) {
            // section[threadIdx.x] += section[threadIdx.x - stride];
            pout[threadIdx.x] = pin[threadIdx.x] + pin[threadIdx.x - stride];
        } else {
            pout[threadIdx.x] = pin[threadIdx.x];
        }
        T* temp = pin;
        pin = pout;
        pout = temp;
    }
    // __syncthreads();
    
    if (i < sz) {
        // out[i] = section[threadIdx.x];
        out[i] = pin[threadIdx.x];
    }

    if (s != nullptr) {
        if (threadIdx.x == blockDim.x - 1) {
            // last thread
            s[blockIdx.x] = pin[SectionSize - 1];
        }
    }
}

template<typename T, size_t SectionSize>
struct Kogge_Stone_scanKernel
{
    static constexpr size_t threadsPerBlock = SectionSize;
    
    static void run(T const* in, T* out, T* s, size_t sz)
    {
        kogge_Stone_scanKernel<T, SectionSize><<<ceil((double)sz/SectionSize), threadsPerBlock>>>(in, out, s, sz);
    }
};

template <typename T, size_t SectionSize>
__global__
void brent_Kung_scanKernel(T const* in, T* out, T* s, size_t sz)
{
    // blockDim.x * 2 == SectionSize

    int i = SectionSize * blockIdx.x + threadIdx.x;
    // double buffering is not necessary
    __shared__ T section[SectionSize * 2];
    
    section[threadIdx.x] = i < sz ? in[i] : 0.0f;
    section[threadIdx.x + blockDim.x] = i + blockDim.x < sz ? in[i + blockDim.x] : 0.0f;

    // firstly build a reduction tree:
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int j = (threadIdx.x + 1) * stride * 2 - 1;
        if (j < SectionSize) {
            section[j] += section[j - stride];            
        }
    }

    // secondly do scan:
    for (int stride = SectionSize / 4; stride > 0; stride /= 2) {
        __syncthreads();
        int j = (threadIdx.x + 1) * stride * 2 - 1;
        if (j + stride < SectionSize) {
            section[j + stride] += section[j];
        }
    }
        
    if (i < sz) {
        out[i] = section[threadIdx.x];
    }
    if (i + blockDim.x < sz) {
        out[i + blockDim.x] = section[threadIdx.x + blockDim.x];
    }

    if (s != nullptr) {
        if (threadIdx.x == blockDim.x - 1) {
            // last thread
            s[blockIdx.x] = section[SectionSize - 1];
        }
    }
}

template <typename T, size_t SectionSize>
struct Brent_Kung_scanKernel
{
    static constexpr size_t threadsPerBlock = SectionSize / 2;
   
    static void run(T const* in, T* out, T* s, size_t sz)
    {
        brent_Kung_scanKernel<T, SectionSize><<<ceil((double)sz/SectionSize), threadsPerBlock>>>(in, out, s, sz);
    }
};

template<typename T, template <typename, size_t> typename SecScanFunc, size_t SectionSize>
std::vector<T> scan_GPU_hierachical(std::vector<T> const& in);

template<typename T, template <typename, size_t> typename SecScanFunc, size_t SectionSize>
std::vector<T> scan_GPU_streaming(std::vector<T> const& in);

int main(int argc, char** argv)
{
    size_t sz;
    if (argc > 1) {
        sz = std::stoul(argv[1]);
    } else {
        sz = 1000000;
    }
    
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<float> dist;
    
    std::vector<float> in(sz);
    for (auto& e : in) e = dist(eng);
    // int c = 0;
    // for (auto& e : in) e = c++;

    std::vector<float> res1, res2, res3;

    {
        TimerGuard tg("scan cpu:");
        res1 = scan_CPU(in);
        doNotOptimize(res1);
    }

    {
        TimerGuard tg("scan gpu koggle-stone 1024:");
        res2 = scan_GPU_hierachical<float, Kogge_Stone_scanKernel, 1024>(in);
        doNotOptimize(res2);
    }

    {
        TimerGuard tg("scan gpu koggle-stone 512:");
        auto res = scan_GPU_hierachical<float, Kogge_Stone_scanKernel, 512>(in);
        doNotOptimize(res);
    }

    {
        TimerGuard tg("scan gpu brent-kung 2048:");
        res3 = scan_GPU_hierachical<float, Brent_Kung_scanKernel, 2048>(in);
        doNotOptimize(res3);
    }

    {
        TimerGuard tg("scan gpu brent-kung 1024:");
        auto res = scan_GPU_hierachical<float, Brent_Kung_scanKernel, 1024>(in);
        doNotOptimize(res);
    }

    {
        TimerGuard tg("scan gpu brent-kung 512:");
        auto res = scan_GPU_hierachical<float, Brent_Kung_scanKernel, 1024>(in);
        doNotOptimize(res);
    }


    if (!vectorCompare(res1, res2)) {
        fmt::print(stderr, "1 2 error\n");
    }
}



template <typename T, size_t SectionSize>
__global__
void scan_GPU_hierachical_finalStep(T* out, T* s, size_t sz)
{
    int i = threadIdx.x + SectionSize * blockIdx.x;
    if (threadIdx.x == blockDim.x - 1 && (blockIdx.x + 1) * SectionSize - 1 < sz) {
        out[(blockIdx.x + 1) * SectionSize - 1] = s[blockIdx.x];
    }
    if (blockIdx.x >= 1) {
        T a = s[blockIdx.x - 1];
        for (int j = i; j < i + SectionSize; j += blockDim.x) {
            if (j < sz && j != (blockIdx.x + 1) * SectionSize - 1) {
                out[j] += a;
            }
        }
    }
}

/* in and out are device pointers */
template<typename T, template <typename, size_t> typename SecScanFunc, size_t SectionSize>
void scan_GPU_hierachical_impl(T const* in, T* out, size_t sz)
{
    cudaError_t err;
    size_t numSections = (sz + SectionSize - 1) / SectionSize;
    if (numSections > 1) {
        T* s;
        err = cudaMalloc((void**)&s, numSections * sizeof(T));
        checkCudaError(err);

        // 1. section scan
        SecScanFunc<T, SectionSize>::run(in, out, s, sz);       
        err = cudaGetLastError();
        checkCudaError(err);

        // 2. scan for s
        scan_GPU_hierachical_impl<T, SecScanFunc, SectionSize>(s, s, numSections);

        // 3. propagate
        scan_GPU_hierachical_finalStep<T, SectionSize><<<numSections, SecScanFunc<T, SectionSize>::threadsPerBlock>>>(out, s, sz);
        err = cudaGetLastError();
        checkCudaError(err);

        cudaFree(s);
    } else {
        SecScanFunc<T, SectionSize>::run(in, out, nullptr, sz);
        err = cudaGetLastError();
        checkCudaError(err);
    }
}

template<typename T, template <typename, size_t> typename SecScanFunc, size_t SectionSize>
std::vector<T> scan_GPU_hierachical(std::vector<T> const& in)
{
    std::vector<T> out(in.size());
    
    cudaError_t err;
    T* d_in;
    T* d_out;

    err = cudaMalloc((void**)&d_in, sizeof(T) * in.size());
    checkCudaError(err);

    err = cudaMalloc((void**)&d_out, sizeof(T) * out.size());
    checkCudaError(err);

    err = cudaMemcpy(d_in, in.data(), sizeof(T) * in.size(), cudaMemcpyHostToDevice);
    checkCudaError(err);

    auto start = std::chrono::system_clock::now();
    scan_GPU_hierachical_impl<T, SecScanFunc, SectionSize>(d_in, d_out, in.size());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print(stderr, "kernel execution time without PCIE transfer: {}\n", std::chrono::duration<double>(end-start).count());

    err = cudaMemcpy(out.data(), d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);
    checkCudaError(err);

    cudaFree(d_in);
    cudaFree(d_out);

    return out;
}
