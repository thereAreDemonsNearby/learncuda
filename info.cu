#include <stdio.h>

int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("device count: %d\n", devCount);
    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("ver: %d.%d\n", devProp.major, devProp.minor);
        printf("max threads per block: %d\n", (int)devProp.maxThreadsPerBlock);
        printf("number of SMs: %d\n", (int)devProp.multiProcessorCount);
        printf("warp size: %d\n", (int)devProp.warpSize);
        printf("registers available per SM: %d\n", (int)devProp.regsPerBlock);
        printf("shared memory available per SM: %d\n", (int)devProp.sharedMemPerBlock);
        printf("clock frequency: %d\n", (int)devProp.clockRate);
        printf("max threads dimBlock.x: %d\n", (int)devProp.maxThreadsDim[0]);
        printf("max threads dimBlock.y: %d\n", (int)devProp.maxThreadsDim[1]);
        printf("max threads dimBlock.z: %d\n", (int)devProp.maxThreadsDim[2]);
        printf("max blocks dimGrid.x: %d\n", (int)devProp.maxGridSize[0]);
        printf("max blocks dimGrid.y: %d\n", (int)devProp.maxGridSize[1]);
        printf("max blocks dimGrid.z: %d\n", (int)devProp.maxGridSize[2]);
        printf("\n");        
    }
}