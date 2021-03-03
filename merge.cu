#include "common.h"
#include <vector>



int main()
{
    
}

template <typename T>
__device__
__host__
int co_rank(int k, T* a, int m, T* b, int n)
{
    int i = min(k, m);
    int j = k - i;

    int i_low = max(0, k - n);
    int j_low = max(0, k - m);

    bool active = true;
    
    while (active) {
        if (i > 0 && j < n && a[i - 1] > b[j]) {
            int delta = (i - i_low + 1) / 2;
            i -= delta;
            j_low = j;
            j += delta;
        } else if (j > 0 && i < m && b[j - 1] > a[i]) {
            int delta = (j - j_low + 1) / 2;
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
void merge_sequential(T* a, int m, T* b, int n, T* c)
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
void merge_kernel_naive(T* a, int m, T* b, int n, T* c)
{
    int numThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int c_curr = tid * ((m+n) / numThreads) + min(tid, (m+n) % numThreads);
    int c_next = (tid+1) * ((m+n) / numThreads) + min(tid+1, (m+n) % numThreads);

    int a_curr = co_rank(c_curr, a, m, b, n);
    int b_curr = c_curr - a_curr;

    int a_next = co_rank(c_next, a, m, b, n);
    int b_next = c_next - a_next;

    merge_sequential(&a[a_curr], a_next-a_curr, &b[b_curr], b_next-b_curr, &c[c_curr]);
}

template <typename T, size_t TILE_SIZE>
__global__
void merge_kernel_tiled(T* a, int m, T* b, int n, T* c)
{
    __shared__ a_S[TILE_SIZE]; // TILE_SIZE can be larger than blockDim.x
    __shared__ b_S[TILE_SIZE];
    
    int bid = blockIdx.x;

    int c_blk_curr = bid * ((m+n) / numThreads) + min(bid, (m+n) % numThreads);
    int c_blk_next = (bid+1) * ((m+n) / numThreads) + min(bid+1, (m+n) % numThreads);
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
        for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
            if (a_blk_curr + a_consumed + t < a_blk_next) {
                a_S[t] = a[a_blk_curr + a_consumed + t];
            }
        }

        for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
            if (b_blk_curr + b_consumed + t < b_blk_next) {
                b_S[t] = a[b_blk_curr + b_consumed + t];
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
                         &c[c_blk_curr + c_completed]);

        c_completed += TILE_SIZE;
        a_consumed += co_rank(TILE_SIZE, a_S, TILE_SIZE, b_S, TILE_SIZE);
        b_consumed += c_completed - a_consumed;
    }
}
