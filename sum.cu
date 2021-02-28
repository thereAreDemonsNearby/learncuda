#include <iostream>
#include <cstddef>
#include "TimerGuard.h"

template <typename T>
T sum(T const* arr, size_t sz)
{
    T sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < sz; ++i) {
        sum += arr[i];
    }
}

template <typename T>
__global__
void sumKernel1(T const* arr, size_t sz);

template <typename T>
__global__
void sumKernel2(T const* arr, size_t sz);

template <typename T>
T sum_gpu1(T const* arr, size_t sz)
{
    cudaError_t err;
    T* arr_dev;
    if (err = cudaMalloc((void**)&arr_dev, sizeof(T) * sz);
        err != cudaSuccess) {
        std::cerr << "cannot alloc mem for arr_dev\n";
        std::exit(1);
    }

    if (err = cudaMemcpy(arr_dev, arr, sz * sizeof(T), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cannot copy to device\n";
        std::exit(1);
    }

    sumKernel1<T><<<, 1024>>>(arr_dev, sz);

    T sum;
    if (err = cudaMemcpy(&sum, arr_dev, sizeof(T), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cannot copy back to host\n";
        std::exit(1);
    }

    return sum;
}

template <typename T>
T sum_gpu2(T const* arr, size_t sz)
{
    cudaError_t err;
    T* arr_dev;
    if (err = cudaMalloc((void**)&arr_dev, sizeof(T) * sz);
        err != cudaSuccess) {
        std::cerr << "cannot alloc mem for arr_dev\n";
        std::exit(1);
    }

    if (err = cudaMemcpy(arr_dev, arr, sz * sizeof(T), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cannot copy to device\n";
        std::exit(1);
    }

    sumKernel2<T><<< ,>>>(arr_dev, sz);

    T sum;
    if (err = cudaMemcpy(&sum, arr_dev, sizeof(T), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cannot copy back to host\n";
        std::exit(1);
    }

    return sum;
}

int main()
{
    
}
