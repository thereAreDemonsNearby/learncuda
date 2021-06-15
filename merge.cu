#include "common.h"
#include <vector>
#include <random>
#include <algorithm>
#include <stdio.h>

template <typename T>
__device__
__host__
void merge_sequential(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c);

template <typename T, typename MergeKernel>
void merge_GPU(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c);

struct MergeKernel_naive
{
    template <typename T>
    void operator()(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c);
};

struct MergeKernel_tiled
{
    template <typename T>
    void operator()(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c);
};

struct MergeKernel_tiled_circular
{
    template <typename T>
    void operator()(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c);
};

template <typename T>
__device__
__host__
int co_rank(int k, T const* __restrict__ a, int m, T const* __restrict__ b, int n);

int BLOCK_SIZE = 128;

int main(int argc, char** argv)
{
    int M = 100000, N = 100000;
    if (argc == 2) {
        M = std::stoi(argv[1]);
        N = M;
    } else if (argc == 3) {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
    }
    std::vector<int> a(M), b(N);

    std::mt19937 eng(std::random_device{}());
    std::uniform_int_distribution<int> distrib(-1000000, 1000000);
    for (auto& e : a) e = distrib(eng);
    for (auto& e : b) e = distrib(eng);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());

    // test co_rank
    // int i = co_rank(10600, a.data(), a.size(), b.data(), b.size());
    // fmt::print("i = {}\n", i);
    for (int i = 0; i <= M + N; ++i) {
        int j = co_rank(i, a.data(), a.size(), b.data(), b.size());
        if (j < 0) {
            fmt::print("j = {}\n", j);
        }
    }
    // return 0;

    std::vector<int> c1(M + N), c2(M+N), c3(M+N), c4(M+N);

    {
        TimerGuard t("merge cpu:");
        merge_sequential(a.data(), (int)a.size(), b.data(), (int)b.size(), c1.data());
    }
    if (!std::is_sorted(c1.begin(), c1.end())) {
        fmt::print(stderr, "merge error\n");
    }

    {
        TimerGuard t("merge gpu naive:");
        merge_GPU<int, MergeKernel_naive>(a.data(), a.size(), b.data(), b.size(), c2.data());
    }
    if (!std::is_sorted(c2.begin(), c2.end())) {
        fmt::print(stderr, "merge error\n");
    }

    {
        TimerGuard t("merge gpu tiled:");
        merge_GPU<int, MergeKernel_tiled>(a.data(), a.size(), b.data(), b.size(), c3.data());
    }
    if (!std::is_sorted(c3.begin(), c3.end())) {
        fmt::print(stderr, "merge error\n");
    }

    // {
    //     TimerGuard t("merge gpu tiled & circular:");
    //     merge_GPU<int, MergeKernel_tiled_circular>(a.data(), a.size(), b.data(), b.size(), c4.data());
    // }
    // if (!std::is_sorted(c4.begin(), c4.end())) {
    //     fmt::print(stderr, "merge error\n");
    // }
}

template <typename T>
__device__
__host__
int co_rank(int k, T const* __restrict__ a, int m, T const* __restrict__ b, int n)
{
    int i = min(k, m);
    int j = k - i;

    int i_low = max(0, k - n);
    int j_low = max(0, k - m);

    bool active = true;
    
    while (active) {
        // if (i < 0 || j < 0) {
        //     printf("fuck : i = %d, i_low = %d, j = %d, j_low = %d, k = %d\n", i, i_low, j, j_low, k);
        // }
        if (i > 0 && j < n && a[i - 1] > b[j]) {
            int delta = min((i - i_low + 1) / 2, n - j);
            // int delta = (i - i_low + 1) / 2;
            // if (delta > i) {
            //     printf("fuck : delta=%d, i = %d, i_low = %d, j = %d, j_low = %d, k = %d\n", delta, i, i_low, j, j_low, k);
            // }
            i -= delta;
            j_low = j;
            j += delta;
        } else if (j > 0 && i < m && b[j - 1] > a[i]) {
            int delta = min((j - j_low + 1) / 2, m - i);
            // int delta = (j - j_low + 1) / 2;
            j -= delta;
            i_low = i;
            i += delta;
        } else {
            active = false;
        }
    }

    return i; // j can be computed by k - i
}

template <typename T>
__device__
__host__
void merge_sequential(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c)
{
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (a[i] < b[j]) {
            c[k++] = a[i++];
        } else {
            c[k++] = b[j++];
        }
    }

    while (i < m) {
        c[k++] = a[i++];
    }

    while (j < n) {
        c[k++] = b[j++];
    }
}


template <typename T>
__global__
void merge_kernel_naive(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c)
{
    int numThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int c_curr = tid * ((m+n) / numThreads) + min(tid, (m+n) % numThreads);
    int c_next = (tid+1) * ((m+n) / numThreads) + min(tid+1, (m+n) % numThreads);

    int a_curr = co_rank(c_curr, a, m, b, n);
    int b_curr = c_curr - a_curr;

    int a_next = co_rank(c_next, a, m, b, n);
    int b_next = c_next - a_next;

    // if (tid <= 20) {
    //     printf("c_curr = %d, c_next = %d, a_curr = %d, a_next = %d, b_curr = %d, b_next = %d\n",
    //            c_curr, c_next, a_curr, a_next, b_curr, b_next);
    // }

    merge_sequential(&a[a_curr], a_next-a_curr, &b[b_curr], b_next-b_curr, &c[c_curr]);
}

template <typename T, int TILE_SIZE>
__global__
void merge_kernel_tiled(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c)
{
    __shared__ T a_S[TILE_SIZE]; // TILE_SIZE can be larger than blockDim.x
    __shared__ T b_S[TILE_SIZE];
    
    int bid = blockIdx.x;

    int c_blk_curr = bid * ((m+n) / gridDim.x) + min(bid, (m+n) % gridDim.x);
    int c_blk_next = (bid+1) * ((m+n) / gridDim.x) + min(bid+1, (m+n) % gridDim.x);
    if (threadIdx.x == 0) {
        a_S[0] = co_rank(c_blk_curr, a, m, b, n);
        a_S[1] = co_rank(c_blk_next, a, m, b, n);
    }
    __syncthreads();
    int a_blk_curr = a_S[0];
    int b_blk_curr = c_blk_curr - a_blk_curr;
    int a_blk_next = a_S[1];
    int b_blk_next = c_blk_next - a_blk_next;

    int c_blk_len = c_blk_next - c_blk_curr;
    int a_blk_len = a_blk_next - a_blk_curr;
    int b_blk_len = b_blk_next - b_blk_curr;
    
    int numIters = (c_blk_len + TILE_SIZE - 1) / TILE_SIZE;

    int a_consumed = 0;
    int b_consumed = 0;
    int c_completed = 0;
    for (int it = 0; it < numIters; ++it) {
        // collaboratively load a and b into shared memory:
        __syncthreads();
        for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
            if (a_blk_curr + a_consumed + t < a_blk_next) {
                a_S[t] = a[a_blk_curr + a_consumed + t];
            }
        }

        for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
            if (b_blk_curr + b_consumed + t < b_blk_next) {
                b_S[t] = b[b_blk_curr + b_consumed + t];
            }
        }
        __syncthreads();

        int c_thrd_curr = threadIdx.x * (TILE_SIZE / blockDim.x); // assume no remainder
        int c_thrd_next = (threadIdx.x + 1) * (TILE_SIZE / blockDim.x);

        c_thrd_curr = min(c_thrd_curr, c_blk_len - c_completed);
        c_thrd_next = min(c_thrd_next, c_blk_len - c_completed);

        int a_thrd_curr = co_rank(c_thrd_curr, a_S, min(TILE_SIZE, a_blk_len - a_consumed),
                                  b_S, min(TILE_SIZE, b_blk_len - b_consumed));
        int b_thrd_curr = c_thrd_curr - a_thrd_curr;

        int a_thrd_next = co_rank(c_thrd_next, a_S, min(TILE_SIZE, a_blk_len - a_consumed),
                                  b_S, min(TILE_SIZE, b_blk_len - b_consumed));
        int b_thrd_next = c_thrd_next - a_thrd_next;

        merge_sequential(&a_S[a_thrd_curr], a_thrd_next - a_thrd_curr,
                         &b_S[b_thrd_curr], b_thrd_next - b_thrd_curr,
                         &c[c_blk_curr + c_completed + c_thrd_curr]);


        a_consumed += co_rank(min(TILE_SIZE, c_blk_len-c_completed),
                              a_S, min(TILE_SIZE, a_blk_len - a_consumed),
                              b_S, min(TILE_SIZE, b_blk_len - b_consumed));
        // a_consumed += co_rank(TILE_SIZE, a_S, TILE_SIZE, b_S, TILE_SIZE);
        c_completed += TILE_SIZE;
        b_consumed += c_completed - a_consumed;
//        __syncthreads();
    }
}

template <typename T, size_t TILE_SIZE>
__device__
__host__
int co_rank_circular(int k, T const* __restrict__ a, int a_start, int m, T const* __restrict__ b, int b_start, int n)
{
    int i = min(k, m);
    int j = k - i;

    int i_low = max(0, k - n);
    int j_low = max(0, k - m);

    bool active = true;
    
    while (active) {
        // if (i < 0 || j < 0) {
        //     printf("fuck : i = %d, i_low = %d, j = %d, j_low = %d, k = %d\n", i, i_low, j, j_low, k);
        // }
        if (i > 0 && j < n && a[(i - 1 + a_start) % TILE_SIZE] > b[(j + b_start) % TILE_SIZE]) {
            int delta = min((i - i_low + 1) / 2, n - j);
            // if (delta > i) {
            //     printf("fuck : delta=%d, i = %d, i_low = %d, j = %d, j_low = %d, k = %d\n", delta, i, i_low, j, j_low, k);
            // }
            i -= delta;
            j_low = j;
            j += delta;
        } else if (j > 0 && i < m && b[(j - 1 + b_start) % TILE_SIZE] > a[(i + a_start) % TILE_SIZE]) {
            int delta = min((j - j_low + 1) / 2, m - i);
            j -= delta;
            i_low = i;
            i += delta;
        } else {
            active = false;
        }
    }

    return i; // j can be computed by k - i
}

template <size_t TILE_SIZE, typename T>
__device__
__host__
void merge_sequential_circular(T const* __restrict__ a, int a_start, int a_thrd_curr, int m,
                               T const* __restrict__ b, int b_start, int b_thrd_curr, int n, T* c)
{
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (a[i] < b[j]) {
            c[k++] = a[(i + a_start + a_thrd_curr) % TILE_SIZE];
            ++i;
        } else {
            c[k++] = b[(j + b_start + b_thrd_curr) % TILE_SIZE];
            ++j;
        }
    }

    while (i < m) {
        c[k++] = a[(i + a_start + a_thrd_curr) % TILE_SIZE];
        ++i;
    }

    while (j < n) {
        c[k++] = b[(j + b_start + b_thrd_curr) % TILE_SIZE];
        ++j;
    }
}

template <typename T, int TILE_SIZE>
__global__
void merge_kernel_tiled_circular(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c)
{
    __shared__ T a_S[TILE_SIZE]; // TILE_SIZE can be larger than blockDim.x
    __shared__ T b_S[TILE_SIZE];
    
    int bid = blockIdx.x;

    int c_blk_curr = bid * ((m+n) / gridDim.x) + min(bid, (m+n) % gridDim.x);
    int c_blk_next = (bid+1) * ((m+n) / gridDim.x) + min(bid+1, (m+n) % gridDim.x);
    if (threadIdx.x == 0) {
        a_S[0] = co_rank(c_blk_curr, a, m, b, n);
        a_S[1] = co_rank(c_blk_next, a, m, b, n);
    }
    __syncthreads();
    int a_blk_curr = a_S[0];
    int b_blk_curr = c_blk_curr - a_blk_curr;
    int a_blk_next = a_S[1];
    int b_blk_next = c_blk_next - a_blk_next;

    int c_blk_len = c_blk_next - c_blk_curr;
    int a_blk_len = a_blk_next - a_blk_curr;
    int b_blk_len = b_blk_next - b_blk_curr;
    
    int numIters = (c_blk_len + TILE_SIZE - 1) / TILE_SIZE;

    int a_consumed = 0;
    int b_consumed = 0;
    int c_completed = 0;

    int a_S_consumed = 0;
    int b_S_consumed = 0;
    int a_start = 0;
    int b_start = 0;
    for (int it = 0; it < numIters; ++it) {
        // collaboratively load a and b into shared memory:
        __syncthreads();
        for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
            if (a_blk_curr + a_consumed + t < a_blk_next) {
                a_S[(t + a_start) % TILE_SIZE] = a[a_blk_curr + a_consumed + t];
            }
        }

        for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
            if (b_blk_curr + b_consumed + t < b_blk_next) {
                b_S[(t + b_start) % TILE_SIZE] = b[b_blk_curr + b_consumed + t];
            }
        }
        __syncthreads();

        int c_thrd_curr = threadIdx.x * (TILE_SIZE / blockDim.x); // assume no remainder
        int c_thrd_next = (threadIdx.x + 1) * (TILE_SIZE / blockDim.x);

        c_thrd_curr = min(c_thrd_curr, c_blk_len - c_completed);
        c_thrd_next = min(c_thrd_next, c_blk_len - c_completed);

        int a_thrd_curr = co_rank_circular<T, TILE_SIZE>(c_thrd_curr,
                                                         a_S, a_start, min(TILE_SIZE, a_blk_len - a_consumed),
                                                         b_S, b_start, min(TILE_SIZE, b_blk_len - b_consumed));
        int b_thrd_curr = c_thrd_curr - a_thrd_curr;

        int a_thrd_next = co_rank_circular<T, TILE_SIZE>(c_thrd_next,
                                                         a_S, a_start, min(TILE_SIZE, a_blk_len - a_consumed),
                                                         b_S, b_start, min(TILE_SIZE, b_blk_len - b_consumed));
        int b_thrd_next = c_thrd_next - a_thrd_next;

        merge_sequential_circular<TILE_SIZE>(a_S, a_start, a_thrd_curr, a_thrd_next - a_thrd_curr,
                                             b_S, b_start, b_thrd_curr, b_thrd_next - b_thrd_curr,
                                             &c[c_blk_curr + c_completed + c_thrd_curr]);

        a_S_consumed = co_rank_circular<T, TILE_SIZE>(min(TILE_SIZE, c_blk_len-c_completed),
                                                      a_S, a_start, min(TILE_SIZE, a_blk_len - a_consumed),
                                                      b_S, b_start, min(TILE_SIZE, b_blk_len - b_consumed));
        a_consumed += a_S_consumed;
        // a_consumed += co_rank(TILE_SIZE, a_S, TILE_SIZE, b_S, TILE_SIZE);
        c_completed += min(TILE_SIZE, c_blk_len-c_completed); //TILE_SIZE;
        b_S_consumed = min(TILE_SIZE, c_blk_len-c_completed) - a_S_consumed;
        b_consumed += b_S_consumed;
        a_start = (a_start + a_S_consumed) % TILE_SIZE;
        b_start = (b_start + b_S_consumed) % TILE_SIZE;
//        __syncthreads();
    }
}

template <typename T, typename MergeKernel>
void merge_GPU(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c)
{
    cudaError_t err;
    T* d_a;
    T* d_b;
    T* d_c;

    err = cudaMalloc((void**)&d_a, sizeof(T) * m);
    checkCudaError(err);

    err = cudaMalloc((void**)&d_b, sizeof(T) * n);
    checkCudaError(err);

    err = cudaMalloc((void**)&d_c, sizeof(T) * (m + n));
    checkCudaError(err);

    err = cudaMemcpy(d_a, a, sizeof(T) * m, cudaMemcpyHostToDevice);
    checkCudaError(err);

    err = cudaMemcpy(d_b, b, sizeof(T) * n, cudaMemcpyHostToDevice);
    checkCudaError(err);

    auto start = std::chrono::system_clock::now();
    MergeKernel kernel;
    kernel(d_a, m, d_b, n, d_c);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print(stderr, "kernel time: {}\n", std::chrono::duration<double>(end-start).count());
    err = cudaGetLastError();
    checkCudaError(err);

    err = cudaMemcpy(c, d_c, sizeof(T) * (m + n), cudaMemcpyDeviceToHost);
    checkCudaError(err);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

template<typename T>
void MergeKernel_naive::operator()(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c)
{
    merge_kernel_naive<T><<<min((m+n+BLOCK_SIZE-1)/BLOCK_SIZE, 1024), BLOCK_SIZE>>>(a, m, b, n, c);
}

template<typename T>
void MergeKernel_tiled::operator()(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c)
{
    merge_kernel_tiled<T, 1024><<<min((m+n+BLOCK_SIZE-1)/BLOCK_SIZE, 1024), BLOCK_SIZE>>>(a, m, b, n, c);
}

template <typename T>
void MergeKernel_tiled_circular::operator()(T const* __restrict__ a, int m, T const* __restrict__ b, int n, T* c)
{
    merge_kernel_tiled_circular<T, 1024><<<min((m+n+BLOCK_SIZE-1)/BLOCK_SIZE, 1024), BLOCK_SIZE>>>(a, m, b, n, c);
}
