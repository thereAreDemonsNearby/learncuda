#include <iostream>
#include <cstddef>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#define cudaErrorCheck(expr)                                            \
    do {                                                                \
        cudaError_t err;                                                \
        err = (expr);                                                   \
        if (err != cudaSuccess) {                                       \
            std::cerr << "line" << __LINE__ << ": cuda error: " << cudaGetErrorString(err) << "\n"; \
            exit(1);                                                    \
        }                                                               \
    } while(0)

void stream(size_t sz);

int main()
{
    stream((size_t)128 * 1024 * 1024);
}

__global__
void copy(float* dst, float* src, size_t sz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < sz; idx += blockDim.x * gridDim.x) {
        dst[idx] = src[idx];
    }
}

__global__
void add(float* dst, float* src1, float* src2, size_t sz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < sz; idx += blockDim.x * gridDim.x) {
        dst[idx] = src1[idx] + src2[idx];
    }
}

__global__
void scale(float* dst, float* src, float alpha, size_t sz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < sz; idx += blockDim.x * gridDim.x) {
        dst[idx] = alpha * src[idx];
    }
}

void stream(size_t sz)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist;
    std::vector<float> v1(sz), v2(sz);
    std::generate(v1.begin(), v1.end(), [&gen, &dist](){ return dist(gen); });
    std::generate(v2.begin(), v2.end(), [&gen, &dist](){ return dist(gen); });

    float *d1, *d2, *dr;
    
    cudaErrorCheck(cudaMalloc((void**)&d1, sz * sizeof(float)));
    cudaErrorCheck(cudaMalloc((void**)&d2, sz * sizeof(float)));
    cudaErrorCheck(cudaMalloc((void**)&dr, sz * sizeof(float)));

    cudaErrorCheck(cudaMemcpy(d1, v1.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d2, v2.data(), sz * sizeof(float), cudaMemcpyHostToDevice));
    
    std::chrono::system_clock::time_point start, end;
    double du;
    
    start = std::chrono::system_clock::now();
    copy<<<ceil(sz / 64.0), 64>>>(dr, d1, sz);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    du = std::chrono::duration<double>(end-start).count();
    std::cerr << "time elapsed for copy: " << du
              << "\nmem bandwidth: " << 2 * sz * sizeof(float) / du << "\n\n";

    start = std::chrono::system_clock::now();
    scale<<<ceil(sz / 64.0), 64>>>(dr, d1, 3.3f, sz);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    du = std::chrono::duration<double>(end-start).count();
    std::cerr << "time elapsed for scale: " << du
              << "\nmem bandwidth: " << 2 * sz * sizeof(float) / du << "\n\n";

    start = std::chrono::system_clock::now();
    add<<<ceil(sz / 64.0), 64>>>(dr, d1, d2, sz);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    du = std::chrono::duration<double>(end-start).count();
    std::cerr << "time elapsed for add: " << du
              << "\nmem bandwidth: " << 3 * sz * sizeof(float) / du << "\n\n";

    cudaFree(d1);
    cudaFree(d2);
    cudaFree(dr);
}
