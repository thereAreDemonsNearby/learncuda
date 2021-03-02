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
        if (i > 0 && a[i - 1] > b[j]) {
            int delta = (i - i_low + 1) / 2;
            i -= delta;
            j_low = j;
            j += delta;
        } else if (j > 0 && b[j - 1] > a[i]) {
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
void merge_naive(T* a, int m, T* b, int n, T* c, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int c_curr = tid * (k / gridDim.x * blockDim.x);
    int c_next = max((tid + 1) * (k / gridDim.x * blockDim.x), k);

    int a_curr = co_rank(c_curr, a, m, b, n);
    int b_curr = c_curr - a_curr;

    int a_next = co_rank(c_next, a, m, b, n);
    int b_next = c_next - a_next;
}
