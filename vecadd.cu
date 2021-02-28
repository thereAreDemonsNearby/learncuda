#include <math.h>
#include <stdio.h>
#include <vector>

__global__
void vecAddKernel(float* A, float* B, float* C, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    float* d_A;
    float* d_B;
    float* d_C;
    int nbytes = n * sizeof(float);

    cudaError_t err;
    err = cudaMalloc((void**)&d_A, nbytes);
    if (err != cudaSuccess) {
        printf("cannot alloc memory: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaMalloc((void**)&d_B, nbytes);
    cudaMalloc((void**)&d_C, nbytes);

    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);
    
    // do the work
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
    
    cudaMemcpy(h_C, d_C, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#define N 10000

int main()
{
    std::vector<float> a(N, 1.1);
    std::vector<float> b(N, 1.1);
    std::vector<float> c(N);

    vecAdd(a.data(), b.data(), c.data(), N);

    for (int i = 0; i < N; ++i) {
        printf("%f\n", c[i]);
    }
}