#include <cmath>
#include <vector>
#include <iostream>
#include "TimerGuard.h"

__global__
void matMulVecKernel(float* resultVec, float* mat,
                     float* vec, size_t n, size_t m)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = 0.0;
        for (size_t i = 0; i < m; ++i) {
            t = t + mat[idx * m + i] * vec[i];
        }
        resultVec[idx] = t;
    }
}

void matMulVecGPU(float* h_res, float* h_mat, float* h_vec, size_t n, size_t m)
{
    float* d_mat;
    cudaError_t err;
    if ((err = cudaMalloc((void**)&d_mat, n * m * sizeof(float))) != cudaSuccess) {
        std::cerr << "cannot alloc mem for mat\n";
        std::exit(1);
    }
    if((err = cudaMemcpy(d_mat, h_mat, n * m * sizeof(float), cudaMemcpyHostToDevice))
       != cudaSuccess) {
        std::cerr << "cannot memcpy mat to device\n";
        std::exit(1);
    }

    float* d_vec;
    cudaMalloc((void**)&d_vec, m * sizeof(float));
    cudaMemcpy(d_vec, h_vec, m * sizeof(float), cudaMemcpyHostToDevice);

    float* d_res;
    cudaMalloc((void**)&d_res, n * sizeof(float));
    
    {
        TimerGuard tg("only the kernel: ");
        matMulVecKernel<<<std::ceil(n/1024.0), 1024>>>(d_res, d_mat, d_vec, n, m);
        std::cerr << "nonsense\n";
    }

    cudaMemcpy(h_res, d_res, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_res);
}

void matMulVecCPU(float* h_res, float* h_mat, float* h_vec, size_t n, size_t m)
{
    for (size_t i = 0; i < n; ++i) {
        h_res[i] = 0.0;
        for (size_t j = 0; j < m; ++j) {
            h_res[i] += h_mat[i * m + j] * h_vec[j];
        }
    }
}

template <typename Cont>
bool nearlyEq(Cont const& c1, Cont const& c2)
// requires c1.size() c1[]
{
    for (size_t i = 0; i < c1.size(); ++i) {
        if (std::abs(c1[i] - c2[i]) > 0.01)
            return false;
    }
    return true;
}


int main()
{
    constexpr size_t n = 20000, m = 50000;
    std::vector<float> mat(n * m, 1.1f);
    std::vector<float> vec(m, 2.2f);
    std::vector<float> res1(n, 0.0f);
    std::vector<float> res2(n, 0.0f);

    {
        TimerGuard tg("gpu: ");
        matMulVecGPU(res1.data(), mat.data(), vec.data(), n, m);
    }

    {
        TimerGuard tg("cpu: ");
        matMulVecCPU(res2.data(), mat.data(), vec.data(), n, m);
    }

    if (!nearlyEq(res1, res2)) {
        std::cout << "result not eq\n";
    }
}