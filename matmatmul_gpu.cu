#include <cstddef>
#include <string>
#include <random>
#include <fmt/core.h>
#include "TimerGuard.h"
#include <boost/align/aligned_allocator.hpp>

template <typename T>
inline void doNotOptimize(T const& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

template <typename T>
class Mat
{
public:
    Mat(size_t r, size_t c)
        : rows_(r), cols_(c), data_(r * c)
    {
    }

    // Mat& operator=(Mat const& rhs) = default;

    T* operator[](size_t i)
    {
        return data_.data() + i * cols_;
    }

    const T* operator[](size_t i) const
    {
        return data_.data() + i * cols_;
    }

    bool operator==(Mat const& rhs) const
    {
        return rows_ == rhs.rows_ && cols_ == rhs.cols_
            && data_ == rhs.data_;
    }

    bool operator!=(Mat const& rhs) const
    {
        return !(*this == rhs);
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    T* data() { return data_.data(); }
    T const* data() const { return data_.data(); }


    size_t rows_, cols_;
    std::vector<T, boost::alignment::aligned_allocator<T, 32>> data_;
};

template <typename T>
bool matNearlyEqual(Mat<T> const& m1, Mat<T> const& m2)
{
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        fmt::print(stderr, "size wrong\n");
        return false;
    }

    for (size_t i = 0; i < m1.rows(); ++i) {
        for (size_t j = 0; j < m1.cols(); ++j) {
            if (std::fabs(m1[i][j] - m2[i][j]) > 0.1) {
                fmt::print(stderr, "{0} {1}, index[{2}][{3}]\ndump:\n", m1[i][j], m2[i][j], i, j);
                for (size_t ii = 0; ii < m1.rows() && ii < 10; ++ii) {
                    for (size_t jj = 0; jj < m1.cols() && jj < 10; ++jj) {
                        fmt::print(stderr, "[{0}][{1}]:{2} vs {3}\t", ii, jj, m1[ii][jj], m2[ii][jj]);
                    }
                    fmt::print(stderr, "\n");
                }
                return false;
            }
        }
    }
    return true;
}

template <typename T>
Mat<T> matMatMult_GPU_trivial(Mat<T> const& m1, Mat<T> const& m2);

template <typename T, size_t TILE_WIDTH>
Mat<T> matMatMult_GPU_tiled(Mat<T> const& m1, Mat<T> const& m2);

template <typename T, size_t TILE_WIDTH, size_t moreWork>
Mat<T> matMatMult_GPU_moreWork(Mat<T> const& m1, Mat<T> const& m2);

template <size_t BLOCK_WIDTH>
Mat<float> matMatMult_GPU_threadBlock2x2(Mat<float> const& a, Mat<float> const& b);

template <size_t BLOCK_WIDTH, size_t MoreWork>
Mat<float> matMatMult_GPU_threadBlock2x2_moreWork(Mat<float> const& m1, Mat<float> const& m2);

template <size_t TILE_WIDTH, size_t MoreWork>
Mat<float> matMatMult_GPU_threadBlock1x2_moreWork(Mat<float> const& m1, Mat<float> const& m2);

// #define test_func(info, func, m1, m2, refRes)          \
//     { \
//     { \
//     TimerGuard tg(info); \
//     auto res = func(m1, m2);  \
//     doNotOptimize(res); \
//     } \
//     if (!matNearlyEqual(res, refRes)) { fmt::print(stderr, "result error\n"); } \
//     }

template <typename Func, typename T>
inline void test_func(std::string const& info, Func func, Mat<T> m1, Mat<T> m2, Mat<T> refRes)
{
    Mat<T> res(m1.rows(), m2.cols());
    {
        TimerGuard tg(info);
        res = func(m1, m2);
        doNotOptimize(res);
    }
    if (!matNearlyEqual(res, refRes)) { fmt::print(stderr, "result error\n"); }
}

int main(int argc, char** argv)
{
    size_t N1 = 2000;
    size_t N2 = 2000;
    size_t N3 = 2000;
    if (argc == 4) {
        N1 = std::stoul(argv[1]);
        N2 = std::stoul(argv[2]);
        N3 = std::stoul(argv[3]);
    }
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_real_distribution<float> urd(-10.0, 10.0);

    Mat<float> m1(N1, N2);
    Mat<float> m2(N2, N3);

    Mat<float> res1(N1, N3);
    // Mat<float> res2(N1, N3);
    // Mat<float> res3(N1, N3);
    // Mat<float> res4(N1, N3);
    // Mat<float> res5(N1, N3);
    
    // Mat<float> res6(N1, N3);
    // Mat<float> res7(N1, N3);

    // do some random init first
    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            m1[i][j] = urd(dre);
        }
    }

    for (size_t i = 0; i < N2; ++i) {
        for (size_t j = 0; j < N3; ++j) {
            m2[i][j] = urd(dre);
        }
    }

    {
        TimerGuard tg("GPU trivial:");
        res1 = matMatMult_GPU_trivial(m1, m2);
        doNotOptimize(res1);
    }

    test_func("GPU tiled 32:", matMatMult_GPU_tiled<float, 32>, m1, m2, res1);
    test_func("GPU more work x1:", matMatMult_GPU_moreWork<float, 32, 1>, m1, m2, res1);
    test_func("GPU more work x2:", matMatMult_GPU_moreWork<float, 32, 2>, m1, m2, res1);
    test_func("GPU more work x4:", matMatMult_GPU_moreWork<float, 32, 4>, m1, m2, res1);
    test_func("GPU more work x8:", matMatMult_GPU_moreWork<float, 32, 8>, m1, m2, res1);
    test_func("GPU tiled 16 thread block 2x2:", matMatMult_GPU_threadBlock2x2<16>, m1, m2, res1);
    test_func("GPU tiled 32 thread block 2x2:", matMatMult_GPU_threadBlock2x2<32>, m1, m2, res1);
    test_func("GPU tiled 64 thread block 2x2:", matMatMult_GPU_threadBlock2x2<64>, m1, m2, res1);
    
    test_func("GPU tiled 32 thread block 2x2 more work x2:", matMatMult_GPU_threadBlock2x2_moreWork<32, 2>, m1, m2, res1);
    test_func("GPU tiled 32 thread block 2x2 more work x4:", matMatMult_GPU_threadBlock2x2_moreWork<32, 4>, m1, m2, res1);
    test_func("GPU tiled 32 thread block 2x2 more work x8:", matMatMult_GPU_threadBlock2x2_moreWork<32, 8>, m1, m2, res1);

    test_func("GPU tiled 32 thread block 1x2 more work x2:", matMatMult_GPU_threadBlock1x2_moreWork<32, 2>, m1, m2, res1);
    test_func("GPU tiled 32 thread block 1x2 more work x4:", matMatMult_GPU_threadBlock1x2_moreWork<32, 4>, m1, m2, res1);
    test_func("GPU tiled 32 thread block 1x2 more work x8:", matMatMult_GPU_threadBlock1x2_moreWork<32, 8>, m1, m2, res1);
}

template <typename T>
__global__
void matMatMultKernel_GPU_trivial(T const* __restrict a, T const* __restrict b,
                                  T* __restrict res, size_t n1, size_t n2, size_t n3)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y < n1 && x < n3) {
        T s = 0;
        for (size_t k = 0; k < n2; ++k) {
            s += a[y * n2 + k] * b[k * n3 + x];
        }
        res[y * n3 + x] = s;
    }
}

template <typename T>
Mat<T> matMatMult_GPU_trivial(Mat<T> const& m1, Mat<T> const& m2)
{
    if (m1.cols() != m2.rows()) {
        throw 0;
    }

    cudaError_t err;
    Mat<T> res(m1.rows(), m2.cols());

    T* d_m1;
    if (err = cudaMalloc((void**)&d_m1, m1.size() * sizeof(T));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::cerr << "err is " << cudaGetErrorString(err) << "\n";
    }

    T* d_m2;
    if (err = cudaMalloc((void**)&d_m2, m2.size() * sizeof(T));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
    }

    T* d_res;
    if (err = cudaMalloc((void**)&d_res, res.size() * sizeof(T));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
    }

    if (err = cudaMemcpy(d_m1, m1.data(), m1.size() * sizeof(T), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
    }
    if (err = cudaMemcpy(d_m2, m2.data(), m2.size() * sizeof(T), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
    }

    double p = 32;
    dim3 dimBlock(p, p, 1);
    dim3 dimGrid(ceil(res.cols() / p),
                 ceil(res.rows() / p), 1);
    auto start = std::chrono::system_clock::now();
    matMatMultKernel_GPU_trivial<T><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print(stderr, "kernel only time: {}\n", std::chrono::duration<double>(end-start).count());

    if (err = cudaGetLastError();
        err != cudaSuccess) {
        std::cerr << "cuda kernel ran incorrectly: " << cudaGetErrorString(err) << "\n";
    }

    if (err = cudaMemcpy(res.data(), d_res, res.size() * sizeof(T), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
    }

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);

    return res;
}


template <typename T, size_t TILE_WIDTH>
__global__
void matMatMultKernel_GPU_tiled(float const* __restrict a, float const* __restrict b,
                                float* __restrict c, size_t ni, size_t nk, size_t nj)
{
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ T aTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ T bTile[TILE_WIDTH][TILE_WIDTH];

    T cVal = 0;
    int row = by*TILE_WIDTH+ty;
    int col = bx*TILE_WIDTH+tx;
    for (int h = 0; h < (int)ceil((T)nk / TILE_WIDTH); ++h) {
        // load tiles from a and b
        if (row < ni && h*TILE_WIDTH+tx < nk)
            aTile[ty][tx] = a[row * nk + h*TILE_WIDTH+tx]; // a[by*TILE_WIDTH+ty][h*TILE_WIDTH+tx];
        else
            aTile[ty][tx] = 0;
        
        if (h*TILE_WIDTH+ty < nk && col < nj)
            bTile[ty][tx] = b[(h*TILE_WIDTH+ty) * nj + col]; // b[h*TILE_WIDTH+ty][bx*TILE_WIDTH+tx];
        else
            bTile[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            cVal += aTile[ty][k] * bTile[k][tx];
        }

        __syncthreads();
    }

    // c[by*TILE_WIDTH+ty][bx*TILE_WIDTH+tx]
    if (row < ni && col < nj)
        c[row * nj + col] = cVal;
}

template <typename T, size_t TILE_WIDTH>
Mat<T> matMatMult_GPU_tiled(Mat<T> const& m1, Mat<T> const& m2)
{
    if (m1.cols() != m2.rows()) {
        throw 0;
    }

    cudaError_t err;
    Mat<T> res(m1.rows(), m2.cols());

    T* d_m1;
    if (err = cudaMalloc((void**)&d_m1, m1.size() * sizeof(T));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::cerr << "err is " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }

    T* d_m2;
    if (err = cudaMalloc((void**)&d_m2, m2.size() * sizeof(T));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    T* d_res;
    if (err = cudaMalloc((void**)&d_res, res.size() * sizeof(T));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    if (err = cudaMemcpy(d_m1, m1.data(), m1.size() * sizeof(T), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }
    if (err = cudaMemcpy(d_m2, m2.data(), m2.size() * sizeof(T), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }

    // double p = TILE_WIDTH;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(ceil((double)res.cols() / TILE_WIDTH),
                 ceil((double)res.rows() / TILE_WIDTH), 1);
    auto start = std::chrono::system_clock::now();
    matMatMultKernel_GPU_tiled<T, TILE_WIDTH><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print(stderr, "kernel only time: {}\n", std::chrono::duration<double>(end-start).count());

    if (err = cudaGetLastError();
        err != cudaSuccess) {
        std::cerr << "cuda kernel ran incorrectly: " << cudaGetErrorString(err) << "\n";
    }

    if (err = cudaMemcpy(res.data(), d_res, res.size() * sizeof(T), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);

    return res;
}

template <typename T, size_t TILE_WIDTH, size_t moreWork>
__global__
void matMatMultKernel_GPU_moreWork(float const* __restrict a, float const* __restrict b, float* __restrict c,
                                   size_t ni, size_t nk, size_t nj)
{
    __shared__ float aTile[TILE_WIDTH][TILE_WIDTH+1]; // + 1?
    __shared__ float bTile[moreWork][TILE_WIDTH][TILE_WIDTH+1];

    float cVal[moreWork];
    #pragma unroll
    for (int i = 0; i < moreWork; ++i) {
        cVal[i] = 0;
    }

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * moreWork * TILE_WIDTH + tx; // col, col + TILE_WIDTH, col + 2 * TILE_WIDTH, ..., col + (moreWork-1)*TILE_WIDTH

    for (int ph = 0; ph < ceil((T)nk / TILE_WIDTH); ++ph) {

        if (row < ni && ph * TILE_WIDTH + tx < nj) {
            aTile[ty][tx] = a[row * nk + ph * TILE_WIDTH + tx];
        } else {
            aTile[ty][tx] = 0;
        }
        // load bTile
        #pragma unroll
        for (int mw = 0; mw < moreWork; ++mw) {
            if (ph * TILE_WIDTH + ty < nk && col + mw * TILE_WIDTH < nj) {
                bTile[mw][ty][tx] = b[(ph * TILE_WIDTH + ty) * nj + col + mw * TILE_WIDTH]; // b[ph * TILE_WIDTH + ty][col + mw * TILE_WIDTH]
            } else {
                bTile[mw][ty][tx] = 0;
            }
        }
        __syncthreads();

        // loop interchange?
        for (int k = 0; k < TILE_WIDTH; ++k) {
            #pragma unroll
            for (int mw = 0; mw < moreWork; ++mw) {
                cVal[mw] += aTile[ty][k] * bTile[mw][k][tx];
            }                
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mw = 0; mw < moreWork; ++mw) {
        if (row < ni && col + mw * TILE_WIDTH < nj) {
            c[row * nj + col + mw * TILE_WIDTH] = cVal[mw]; // c[row][col + mw * TILE_WIDTH]
        }
    }
}

template <typename T, size_t TILE_WIDTH, size_t moreWork>
Mat<T> matMatMult_GPU_moreWork(Mat<T> const& m1, Mat<T> const& m2)
{
    if (m1.cols() != m2.rows()) {
        throw 0;
    }

    cudaError_t err;
    Mat<T> res(m1.rows(), m2.cols());

    T* d_m1;
    if (err = cudaMalloc((void**)&d_m1, m1.size() * sizeof(T));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::cerr << "err is " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }

    T* d_m2;
    if (err = cudaMalloc((void**)&d_m2, m2.size() * sizeof(T));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    T* d_res;
    if (err = cudaMalloc((void**)&d_res, res.size() * sizeof(T));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    if (err = cudaMemcpy(d_m1, m1.data(), m1.size() * sizeof(T), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }
    if (err = cudaMemcpy(d_m2, m2.data(), m2.size() * sizeof(T), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }

    double p = TILE_WIDTH;
    dim3 dimBlock(p, p, 1);
    dim3 dimGrid(ceil(ceil(res.cols() / p) / moreWork),
                 ceil(res.rows() / p), 1);
    auto start = std::chrono::system_clock::now();
    matMatMultKernel_GPU_moreWork<T, TILE_WIDTH, moreWork><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print(stderr, "kernel only time: {}\n", std::chrono::duration<double>(end-start).count());

    if (err = cudaGetLastError();
        err != cudaSuccess) {
        std::cerr << "cuda kernel ran incorrectly: " << cudaGetErrorString(err) << "\n";
    }

    if (err = cudaMemcpy(res.data(), d_res, res.size() * sizeof(T), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
    }

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);

    return res;
}


// single precision only
template <size_t TILE_WIDTH>
__global__
void matMatMultKernel_GPU_threadBlock2x2(float const* __restrict a, float const* __restrict b,
                                         float* __restrict c, size_t ni, size_t nk, size_t nj)
{
    // one thread owns a 2x2 block of c   
    // __shared__ float aTile[TILE_WIDTH][TILE_WIDTH];
    // __shared__ float bTile[TILE_WIDTH][TILE_WIDTH];    
    __shared__ float2 aTile[TILE_WIDTH][TILE_WIDTH/2];
    __shared__ float2 bTile[TILE_WIDTH][TILE_WIDTH/2];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = by * TILE_WIDTH + 2 * ty;
    int col = bx * TILE_WIDTH + 2 * tx;

    // one thread will generate four results
    float2 cLine1 = {0.0f, 0.0f};
    float2 cLine2 = {0.0f, 0.0f};
    for (int ph = 0; ph < (int)ceil((float)nk / TILE_WIDTH); ++ph) {
        // load aTile and bTile
        // too much branch...
        // load aTile:
        if (row < ni) {
            if (ph * TILE_WIDTH + 2 * tx + 1 < nk) {
                aTile[ty * 2][tx] = *reinterpret_cast<float2 const*>(&a[row * nk + ph * TILE_WIDTH + 2 * tx]); // a[row][2*(ph * TILE_WIDTH + tx)]
            } else if (ph * TILE_WIDTH + 2 * tx < nk) {
                aTile[ty * 2][tx] = {a[row * nk + ph * TILE_WIDTH + 2 * tx], 0.0f};
            } else {
                aTile[ty * 2][tx] = {0.0f, 0.0f};
            }            
        } else {
            aTile[ty * 2][tx] = {0.0f, 0.0f};
        }
        
        if (row + 1 < ni) {
            if (ph * TILE_WIDTH + tx * 2 + 1 < nk) {
                aTile[ty * 2 + 1][tx] = *reinterpret_cast<float2 const*>(&a[(row + 1) * nk + ph * TILE_WIDTH + tx * 2]); // a[row + 1][2*(ph * TILE_WIDTH+tx)];
            } else if (2*(ph * TILE_WIDTH + tx) < nk) {
                aTile[ty * 2 + 1][tx] = {a[(row + 1) * nk + ph * TILE_WIDTH + tx * 2], 0.0f};
            } else {
                aTile[ty * 2 + 1][tx] = {0.0f, 0.0f};
            }
        } else {
            aTile[ty * 2 + 1][tx] = {0.0f, 0.0f};
        }

        // load bTile:
        if (ph * TILE_WIDTH + ty * 2 < nk) {
            if (col + 1 < nj) {
                bTile[ty * 2][tx] = *reinterpret_cast<float2 const*>(&b[(ph * TILE_WIDTH + ty * 2) * nj + col]); // b[2*(ph * TILE_WIDTH + ty)][col]
            } else if (col < nj) {
                bTile[ty * 2][tx] = {b[(ph * TILE_WIDTH + ty * 2) * nj + col], 0.0f};
            } else {
                bTile[ty * 2][tx] = {0.0f, 0.0f};
            }                
        } else {
            bTile[ty * 2][tx] = {0.0f, 0.0f};
        }

        if (ph * TILE_WIDTH + ty * 2 + 1 < nk) {
            if (col + 1 < nj) {
                bTile[ty * 2 + 1][tx] = *reinterpret_cast<float2 const*>(&b[(ph * TILE_WIDTH + ty * 2 + 1) * nj + col]); // b[2*(ph * TILE_WIDTH + ty) + 1][col];
            } else if (col < nj) {
                bTile[ty * 2 + 1][tx] = {b[(ph * TILE_WIDTH + ty * 2 + 1) * nj + col], 0.0f};
            } else {
                bTile[ty * 2 + 1][tx] = {0.0f, 0.0f};
            }                

        } else {
            bTile[ty * 2 + 1][tx] = {0.0f, 0.0f};
        }
        
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH / 2; ++k) {
            // a[row][2*k] a[row][2*k+1] a[row+1][2*k] a[row+1][2*k+1]
            // b[2*k][col] b[2*k][col+1] b[2*k+1][col] b[2*k+1][col+1]
            cLine1.x += aTile[2*ty][k].x * bTile[2*k][tx].x + aTile[2*ty][k].y * bTile[2*k+1][tx].x;
            cLine1.y += aTile[2*ty][k].x * bTile[2*k][tx].y + aTile[2*ty][k].y * bTile[2*k+1][tx].y;
            cLine2.x += aTile[2*ty+1][k].x * bTile[2*k][tx].x + aTile[2*ty+1][k].y * bTile[2*k+1][tx].x;
            cLine2.y += aTile[2*ty+1][k].x * bTile[2*k][tx].y + aTile[2*ty+1][k].y * bTile[2*k+1][tx].y;
        }

        __syncthreads();
    }

    // c[row][col]
    if (row < ni && col + 1 < nj) {
        *reinterpret_cast<float2*>(&c[row * nj + col]) = cLine1;
    } else if (row < ni && col < nj) {
        c[row * nj + col] = cLine1.x;
    }

    if (row + 1 < ni && col + 1 < nj) {
        *reinterpret_cast<float2*>(&c[(row + 1) * nj + col]) = cLine2;
    } else if (row + 1 < ni && col < nj) {
        c[(row + 1) * nj + col] = cLine2.x;
    }
}

template <size_t TILE_WIDTH>
Mat<float> matMatMult_GPU_threadBlock2x2(Mat<float> const& m1, Mat<float> const& m2)
{
    if (m1.cols() != m2.rows()) {
        throw 0;
    }

    cudaError_t err;
    Mat<float> res(m1.rows(), m2.cols());

    float* d_m1;
    if (err = cudaMalloc((void**)&d_m1, m1.size() * sizeof(float));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::cerr << "err is " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }

    float* d_m2;
    if (err = cudaMalloc((void**)&d_m2, m2.size() * sizeof(float));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    float* d_res;
    if (err = cudaMalloc((void**)&d_res, res.size() * sizeof(float));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    if (err = cudaMemcpy(d_m1, m1.data(), m1.size() * sizeof(float), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }
    if (err = cudaMemcpy(d_m2, m2.data(), m2.size() * sizeof(float), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }

    double p = TILE_WIDTH;
    dim3 dimBlock(TILE_WIDTH/2, TILE_WIDTH/2, 1);
    dim3 dimGrid(ceil(res.cols() / p),
                 ceil(res.rows() / p), 1);
    auto start = std::chrono::system_clock::now();
    matMatMultKernel_GPU_threadBlock2x2<TILE_WIDTH><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print(stderr, "kernel only time: {}\n", std::chrono::duration<double>(end-start).count());

    if (err = cudaGetLastError();
        err != cudaSuccess) {
        std::cerr << "cuda kernel ran incorrectly: " << cudaGetErrorString(err) << "\n";
    }

    if (err = cudaMemcpy(res.data(), d_res, res.size() * sizeof(float), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
    }

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);

    return res;
}


template <size_t TILE_WIDTH, size_t MoreWork>
__global__
void matMatMultKernel_GPU_threadBlock2x2_moreWork(float const* __restrict a, float const* __restrict b,
                                                  float* __restrict c, size_t ni, size_t nk, size_t nj)
{
    // one thread owns a 2x2 block of c
    // __shared__ float aTile[TILE_WIDTH][TILE_WIDTH];
    // __shared__ float bTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float2 aTile[TILE_WIDTH][TILE_WIDTH/2+1];
    __shared__ float2 bTile[MoreWork][TILE_WIDTH][TILE_WIDTH/2];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = by * TILE_WIDTH + ty * 2;
    // int col = bx * TILE_WIDTH + tx * 2; // col, col + TILE_WIDTH, col + 2*TILE_WIDTH, ...
    int col = bx * MoreWork * TILE_WIDTH + 2 * tx;

    // one thread will generate four results
    float2 cLine1[MoreWork];
    float2 cLine2[MoreWork];
    #pragma unroll
    for (int i = 0; i < MoreWork; ++i) {
        cLine1[i] = {0.0f, 0.0f};
        cLine2[i] = {0.0f, 0.0f};
    }
    
    for (int ph = 0; ph < (int)ceil((float)nk / TILE_WIDTH); ++ph) {
        // load aTile and bTile
        // too much branch...
        // load aTile:
        if (row < ni) {
            if (ph * TILE_WIDTH + tx * 2 + 1 < nk) {
                aTile[ty * 2][tx] = *reinterpret_cast<float2 const*>(&a[row * nk + ph * TILE_WIDTH + tx * 2]); // a[row][2*(ph * TILE_WIDTH + tx)]
            } else if (ph * TILE_WIDTH + tx * 2 < nk) {
                aTile[ty * 2][tx] = {a[row * nk + ph * TILE_WIDTH + tx * 2], 0.0f};
            } else {
                aTile[ty * 2][tx] = {0.0f, 0.0f};
            }            
        } else {
            aTile[ty * 2][tx] = {0.0f, 0.0f};
        }
        
        if (row + 1 < ni) {
            if (ph * TILE_WIDTH + tx * 2 + 1 < nk) {
                aTile[ty * 2 + 1][tx] = *reinterpret_cast<float2 const*>(&a[(row + 1) * nk + ph * TILE_WIDTH + tx * 2]); // a[row + 1][2*(ph * TILE_WIDTH+tx)];
            } else if (ph * TILE_WIDTH + tx * 2 < nk) {
                aTile[ty * 2 + 1][tx] = {a[(row + 1) * nk + ph * TILE_WIDTH + tx * 2], 0.0f};
            } else {
                aTile[ty * 2 + 1][tx] = {0.0f, 0.0f};
            }
        } else {
            aTile[ty * 2 + 1][tx] = {0.0f, 0.0f};
        }
        
        #pragma unroll
        for (int mw = 0; mw < MoreWork; ++mw) {
            // load bTile:
            if (ph * TILE_WIDTH + ty * 2 < nk) {
                if (col + mw * TILE_WIDTH + 1 < nj) {
                    bTile[mw][ty * 2][tx] = *reinterpret_cast<float2 const*>(&b[(ph * TILE_WIDTH + ty * 2) * nj + col + mw * TILE_WIDTH]); // b[2*(ph * TILE_WIDTH + ty)][col]
                } else if (col + mw * TILE_WIDTH < nj) {
                    bTile[mw][ty * 2][tx] = {b[(ph * TILE_WIDTH + ty * 2) * nj + col + mw * TILE_WIDTH], 0.0f};
                } else {
                    bTile[mw][ty * 2][tx] = {0.0f, 0.0f};
                }                
            } else {
                bTile[mw][ty * 2][tx] = {0.0f, 0.0f};
            }

            if (ph * TILE_WIDTH + ty * 2 + 1 < nk) {
                if (col + mw * TILE_WIDTH + 1 < nj) {
                    bTile[mw][ty * 2 + 1][tx] = *reinterpret_cast<float2 const*>(&b[(ph * TILE_WIDTH + ty * 2 + 1) * nj + col + mw * TILE_WIDTH]); // b[2*(ph * TILE_WIDTH + ty) + 1][col];
                } else if (col + mw * TILE_WIDTH < nj) {
                    bTile[mw][ty * 2 + 1][tx] = {b[(ph * TILE_WIDTH + ty * 2 + 1) * nj + col + mw * TILE_WIDTH], 0.0f};
                } else {
                    bTile[mw][ty * 2 + 1][tx] = {0.0f, 0.0f};
                }

            } else {
                bTile[mw][ty * 2 + 1][tx] = {0.0f, 0.0f};
            }
        }
        
        __syncthreads();

        #pragma unroll
        for (int mw = 0; mw < MoreWork; ++mw) {
            for (int k = 0; k < TILE_WIDTH / 2; ++k) {
                // a[row][2*k] a[row][2*k+1] a[row+1][2*k] a[row+1][2*k+1]
                // b[2*k][col] b[2*k][col+1] b[2*k+1][col] b[2*k+1][col+1]
                cLine1[mw].x += aTile[2*ty][k].x * bTile[mw][2*k][tx].x + aTile[2*ty][k].y * bTile[mw][2*k+1][tx].x;
                cLine1[mw].y += aTile[2*ty][k].x * bTile[mw][2*k][tx].y + aTile[2*ty][k].y * bTile[mw][2*k+1][tx].y;
                cLine2[mw].x += aTile[2*ty+1][k].x * bTile[mw][2*k][tx].x + aTile[2*ty+1][k].y * bTile[mw][2*k+1][tx].x;
                cLine2[mw].y += aTile[2*ty+1][k].x * bTile[mw][2*k][tx].y + aTile[2*ty+1][k].y * bTile[mw][2*k+1][tx].y;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int mw = 0; mw < MoreWork; ++mw) {
        // c[row][col]
        if (row < ni && col + mw * TILE_WIDTH + 1 < nj) {
            *reinterpret_cast<float2*>(&c[row * nj + col + mw * TILE_WIDTH]) = cLine1[mw];
        } else if (row < ni && col + mw * TILE_WIDTH < nj) {
            c[row * nj + col + mw * TILE_WIDTH] = cLine1[mw].x;
        }

        if (row + 1 < ni && col + mw * TILE_WIDTH + 1 < nj) {
            *reinterpret_cast<float2*>(&c[(row + 1) * nj + col + mw * TILE_WIDTH]) = cLine2[mw];
        } else if (row + 1 < ni && col + mw * TILE_WIDTH < nj) {
            c[(row + 1) * nj + col + mw * TILE_WIDTH] = cLine2[mw].x;
        }
    }
}

template <size_t TILE_WIDTH, size_t MoreWork>
Mat<float> matMatMult_GPU_threadBlock2x2_moreWork(Mat<float> const& m1, Mat<float> const& m2)
{
    if (m1.cols() != m2.rows()) {
        throw 0;
    }

    cudaError_t err;
    Mat<float> res(m1.rows(), m2.cols());

    float* d_m1;
    if (err = cudaMalloc((void**)&d_m1, m1.size() * sizeof(float));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::cerr << "err is " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }

    float* d_m2;
    if (err = cudaMalloc((void**)&d_m2, m2.size() * sizeof(float));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    float* d_res;
    if (err = cudaMalloc((void**)&d_res, res.size() * sizeof(float));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    if (err = cudaMemcpy(d_m1, m1.data(), m1.size() * sizeof(float), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }
    if (err = cudaMemcpy(d_m2, m2.data(), m2.size() * sizeof(float), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }

    double p = TILE_WIDTH;
    dim3 dimBlock(TILE_WIDTH/2, TILE_WIDTH/2, 1);
    dim3 dimGrid(std::ceil(std::ceil(res.cols() / p) / MoreWork),
                 std::ceil(res.rows() / p), 1);
    auto start = std::chrono::system_clock::now();
    matMatMultKernel_GPU_threadBlock2x2_moreWork<TILE_WIDTH, MoreWork><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print(stderr, "kernel only time: {}\n", std::chrono::duration<double>(end-start).count());

    if (err = cudaGetLastError();
        err != cudaSuccess) {
        std::cerr << "cuda kernel ran incorrectly: " << cudaGetErrorString(err) << "\n";
    }

    if (err = cudaMemcpy(res.data(), d_res, res.size() * sizeof(float), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
    }

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);

    return res;
}


template <size_t TILE_WIDTH, size_t MoreWork>
__global__
void matMatMultKernel_GPU_threadBlock1x2_moreWork(float const* __restrict a, float const* __restrict b,
                                                  float* __restrict c, size_t ni, size_t nk, size_t nj)
{
    // one thread owns a 2x1 block of c
    // __shared__ float aTile[TILE_WIDTH][TILE_WIDTH];
    // __shared__ float bTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float2 aTile[TILE_WIDTH][TILE_WIDTH/2 + 1];
    __shared__ float2 bTile[MoreWork][TILE_WIDTH][TILE_WIDTH/2];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = by * TILE_WIDTH + ty;
    // int col = bx * TILE_WIDTH + tx * 2; // col, col + TILE_WIDTH, col + 2*TILE_WIDTH, ...
    int col = bx * MoreWork * TILE_WIDTH + 2 * tx;

    // one thread will generate four results
    float2 cLine[MoreWork];
    #pragma unroll
    for (int i = 0; i < MoreWork; ++i) {
        cLine[i] = {0.0f, 0.0f};
    }
    
    for (int ph = 0; ph < (int)ceil((float)nk / TILE_WIDTH); ++ph) {
        // load aTile and bTile
        // too much branch...
        // load aTile:
        if (row < ni) {
            if (ph * TILE_WIDTH + tx * 2 + 1 < nk) {
                aTile[ty][tx] = *reinterpret_cast<float2 const*>(&a[row * nk + ph * TILE_WIDTH + tx * 2]);
            } else if (ph * TILE_WIDTH + tx * 2 < nk) {
                aTile[ty][tx] = {a[row * nk + ph * TILE_WIDTH + tx * 2], 0.0f};
            } else {
                aTile[ty][tx] = {0.0f, 0.0f};
            }            
        } else {
            aTile[ty][tx] = {0.0f, 0.0f};
        }
                
        #pragma unroll
        for (int mw = 0; mw < MoreWork; ++mw) {
            // load bTile:
            if (ph * TILE_WIDTH + ty < nk) {
                if (col + mw * TILE_WIDTH + 1 < nj) {
                    bTile[mw][ty][tx] = *reinterpret_cast<float2 const*>(&b[(ph * TILE_WIDTH + ty) * nj + col + mw * TILE_WIDTH]); // b[2*(ph * TILE_WIDTH + ty)][col]
                } else if (col + mw * TILE_WIDTH < nj) {
                    bTile[mw][ty][tx] = {b[(ph * TILE_WIDTH + ty) * nj + col + mw * TILE_WIDTH], 0.0f};
                } else {
                    bTile[mw][ty][tx] = {0.0f, 0.0f};
                }                
            } else {
                bTile[mw][ty][tx] = {0.0f, 0.0f};
            }

        }
        
        __syncthreads();

        #pragma unroll
        for (int mw = 0; mw < MoreWork; ++mw) {
            
            for (int k = 0; k < TILE_WIDTH / 2; ++k) {
                // aTile[ty][k].{x,y} & bTile[2*k{,+1}][tx].{x,y}
                cLine[mw].x += aTile[ty][k].x * bTile[mw][2*k][tx].x + aTile[ty][k].y * bTile[mw][2*k+1][tx].x;
                cLine[mw].y += aTile[ty][k].x * bTile[mw][2*k][tx].y + aTile[ty][k].y * bTile[mw][2*k+1][tx].y;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int mw = 0; mw < MoreWork; ++mw) {
        // c[row][col]
        if (row < ni && col + mw * TILE_WIDTH + 1 < nj) {
            *reinterpret_cast<float2*>(&c[row * nj + col + mw * TILE_WIDTH]) = cLine[mw];
        } else if (row < ni && col + mw * TILE_WIDTH < nj) {
            c[row * nj + col + mw * TILE_WIDTH] = cLine[mw].x;
        }
    }
}

template <size_t TILE_WIDTH, size_t MoreWork>
Mat<float> matMatMult_GPU_threadBlock1x2_moreWork(Mat<float> const& m1, Mat<float> const& m2)
{
    if (m1.cols() != m2.rows()) {
        throw 0;
    }

    cudaError_t err;
    Mat<float> res(m1.rows(), m2.cols());

    float* d_m1;
    if (err = cudaMalloc((void**)&d_m1, m1.size() * sizeof(float));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::cerr << "err is " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }

    float* d_m2;
    if (err = cudaMalloc((void**)&d_m2, m2.size() * sizeof(float));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    float* d_res;
    if (err = cudaMalloc((void**)&d_res, res.size() * sizeof(float));
        err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        std::exit(1);
    }

    if (err = cudaMemcpy(d_m1, m1.data(), m1.size() * sizeof(float), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }
    if (err = cudaMemcpy(d_m2, m2.data(), m2.size() * sizeof(float), cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        std::exit(1);
    }

    double p = TILE_WIDTH;
    dim3 dimBlock(TILE_WIDTH/2, TILE_WIDTH, 1);
    dim3 dimGrid(std::ceil(std::ceil(res.cols() / p) / MoreWork),
                 std::ceil(res.rows() / p), 1);
    auto start = std::chrono::system_clock::now();
    matMatMultKernel_GPU_threadBlock1x2_moreWork<TILE_WIDTH, MoreWork><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    fmt::print(stderr, "kernel only time: {}\n", std::chrono::duration<double>(end-start).count());

    if (err = cudaGetLastError();
        err != cudaSuccess) {
        std::cerr << "cuda kernel ran incorrectly: " << cudaGetErrorString(err) << "\n";
    }

    if (err = cudaMemcpy(res.data(), d_res, res.size() * sizeof(float), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
    }

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);

    return res;
}