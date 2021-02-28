#include <fmt/core.h>
#include <cstddef>
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
