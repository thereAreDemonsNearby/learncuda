#include <cstddef>
#include <string>
#include <random>
#include <fmt/core.h>
#include "TimerGuard.h"
#include <boost/align/aligned_allocator.hpp>
#include <cublas_v2.h>

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

Mat<float> matMatMult_cublas(Mat<float> const& m1, Mat<float> const& m2);

template <typename T, size_t TILE_WIDTH>
Mat<T> matMatMult_GPU_tiled(Mat<T> const& m1, Mat<T> const& m2);

template <typename T, size_t TILE_WIDTH, size_t moreWork>
Mat<T> matMatMult_GPU_moreWork(Mat<T> const& m1, Mat<T> const& m2);

template <size_t BLOCK_WIDTH>
Mat<float> matMatMult_GPU_threadBlock2x2(Mat<float> const& a, Mat<float> const& b);

template <size_t BLOCK_WIDTH, size_t MoreWork>
Mat<float> matMatMult_GPU_threadBlock2x2_moreWork(Mat<float> const& m1, Mat<float> const& m2);

template <int LDCBLK, int A_TILE_WIDTH, int VEC_LEN=LDCBLK>
Mat<float> matMatMult_Volkov_Demmel(Mat<float> const& m1, Mat<float> const& m2);

template <size_t TILE_IJ, size_t TILE_K, size_t REGBLK_WIDTH>
Mat<float> matMatMult_regblk(Mat<float> const& m1, Mat<float> const& m2);



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
    size_t N1 = 2048;
    size_t N2 = 2048;
    size_t N3 = 2048;
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

    // test_func("GPU tiled 32:", matMatMult_GPU_tiled<float, 32>, m1, m2, res1);
    // test_func("GPU more work x1:", matMatMult_GPU_moreWork<float, 32, 1>, m1, m2, res1);
    // test_func("GPU more work x2:", matMatMult_GPU_moreWork<float, 32, 2>, m1, m2, res1);
    // test_func("GPU more work x4:", matMatMult_GPU_moreWork<float, 32, 4>, m1, m2, res1);
    // test_func("GPU more work x8:", matMatMult_GPU_moreWork<float, 32, 8>, m1, m2, res1);
    // test_func("GPU tiled 16 thread block 2x2:", matMatMult_GPU_threadBlock2x2<16>, m1, m2, res1);
    // test_func("GPU tiled 32 thread block 2x2:", matMatMult_GPU_threadBlock2x2<32>, m1, m2, res1);
    // test_func("GPU tiled 64 thread block 2x2:", matMatMult_GPU_threadBlock2x2<64>, m1, m2, res1);

    test_func("cublas: :", matMatMult_cublas, m1, m2, res1);
    
    // test_func("GPU tiled 32 thread block 2x2 more work x2:", matMatMult_GPU_threadBlock2x2_moreWork<32, 2>, m1, m2, res1);
    // test_func("GPU tiled 32 thread block 2x2 more work x4:", matMatMult_GPU_threadBlock2x2_moreWork<32, 4>, m1, m2, res1);
    // test_func("GPU tiled 32 thread block 2x2 more work x8:", matMatMult_GPU_threadBlock2x2_moreWork<32, 8>, m1, m2, res1);

    // test_func("Volkov & Demmel 64x16 (origin):", matMatMult_Volkov_Demmel<64, 16>, m1, m2, res1);
    // test_func("Volkov & Demmel 32x32:", matMatMult_Volkov_Demmel<32, 32>, m1, m2, res1);
    // test_func("Volkov & Demmel 64x32:", matMatMult_Volkov_Demmel<64, 32>, m1, m2, res1);
    // test_func("Volkov & Demmel 128x32:", matMatMult_Volkov_Demmel<128, 32>, m1, m2, res1);
    // // test_func("Volkov & Demmel 128x64:", matMatMult_Volkov_Demmel<128, 64>, m1, m2, res1);

    test_func("Volkov & Demmel 256x32:", matMatMult_Volkov_Demmel<256, 32>, m1, m2, res1);
    test_func("Volkov & Demmel 512x32:", matMatMult_Volkov_Demmel<512, 32>, m1, m2, res1);


    test_func("Thread block 16x16, reg block 8x8: ", matMatMult_regblk<128, 8, 8>, m1, m2, res1);
    // 256x*: similar performance as 128x*
    // test_func("Volkov & Demmel 256x32:", matMatMult_Volkov_Demmel<256, 32>, m1, m2, res1);
    // test_func("Volkov & Demmel 256x64:", matMatMult_Volkov_Demmel<256, 64>, m1, m2, res1);

    // test_func("Volkov & Demmel 128x32 (thrd grp 64):", matMatMult_Volkov_Demmel<128, 32, 64>, m1, m2, res1);
    // test_func("Volkov & Demmel 256x32 (thrd grp 64):", matMatMult_Volkov_Demmel<256, 32, 64>, m1, m2, res1);
    // test_func("Volkov & Demmel 256x32 (thrd grp 128):", matMatMult_Volkov_Demmel<256, 32, 128>, m1, m2, res1);
    // test_func("Volkov & Demmel 512x32 (thrd grp 64):", matMatMult_Volkov_Demmel<512, 32, 64>, m1, m2, res1);
    // test_func("Volkov & Demmel 512x32 (thrd grp 128):", matMatMult_Volkov_Demmel<512, 32, 128>, m1, m2, res1);
    // test_func("Volkov & Demmel 512x32 (thrd grp 256):", matMatMult_Volkov_Demmel<512, 32, 256>, m1, m2, res1);
    // test_func("Volkov & Demmel 512x32 (thrd grp 512):", matMatMult_Volkov_Demmel<512, 32, 512>, m1, m2, res1);

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

Mat<float> matMatMult_cublas(Mat<float> const& m1, Mat<float> const& m2)
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

    // initialize cublas
    cublasHandle_t handle;
    cublasStatus_t status;
    if (status = cublasCreate_v2(&handle);
        status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasCreate failed\n";
        std::exit(1);
    }

    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    auto start = std::chrono::system_clock::now();

    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   m2.cols(), m1.rows(), m1.cols(),
                   &alpha, d_m2, m2.cols(), d_m1, m1.cols(), &beta, d_res, m2.cols());

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


// a row major version for the matrix multiplication algorithm in Volkov&Demmel
// default (in the paper):
// thread block dim: 64x1x1
// C block: 16x64
// A block: 16x16
// B block: no tiling
template <int LDCBLK, int A_TILE_WIDTH, int VEC_LEN=LDCBLK>
__global__
void matMatMultKernel_Volkov_Demmel(float const* a, float const* b, float* __restrict c,
                                    size_t const ni, size_t const nk, size_t const nj)
{
    static_assert(LDCBLK % VEC_LEN == 0);
    static_assert(VEC_LEN % A_TILE_WIDTH == 0 && A_TILE_WIDTH % (VEC_LEN / A_TILE_WIDTH) == 0);

    // constexpr int ITERS = LDCBLK / VEC_LEN;
    
    int by = A_TILE_WIDTH * blockIdx.y;
    int bx = LDCBLK * blockIdx.x; // blockDim.x == LDCBLK
    int tx = threadIdx.x; // [0, VEC_LEN)
    
    float cRegs[LDCBLK / VEC_LEN][A_TILE_WIDTH] = {0.0f};
    __shared__ float aTile[A_TILE_WIDTH][A_TILE_WIDTH+1];

    int aRowBase = by + tx / A_TILE_WIDTH;
    
    for (int pk = 0; pk < ceil(nk / float(A_TILE_WIDTH)); ++pk) {
        // TODO load a into shared mem
        // a will be transposed
        // 16/4=4
        constexpr int REP = A_TILE_WIDTH * A_TILE_WIDTH / VEC_LEN;
        constexpr int STEP = VEC_LEN / A_TILE_WIDTH;
        #pragma unroll
        for (int i = 0; i < REP; ++i) {
            // a[by + tx/16 + i * 4][pk * 16 + tx % 16]
            int aRow = aRowBase + i * STEP;
            int aCol = pk * A_TILE_WIDTH + tx % A_TILE_WIDTH;
            if (aRow < ni && aCol < nk) {
                aTile[tx % A_TILE_WIDTH][i * STEP + tx / A_TILE_WIDTH]
                    = a[aRow * nk + aCol];
            } else {
                aTile[tx % A_TILE_WIDTH][i * STEP + tx / A_TILE_WIDTH]
                    = 0.0f;
            }

            // what if i don't transpose: slower
            // if (aRow < ni && aCol < nk) {
            //     aTile[i * STEP + tx / A_TILE_WIDTH][tx % A_TILE_WIDTH]
            //         = a[aRow * nk + aCol];
            // } else {
            //     aTile[i * STEP + tx / A_TILE_WIDTH][tx % A_TILE_WIDTH]
            //         = 0.0f;
            // }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < A_TILE_WIDTH; ++i) {
            // for (int x = tx; x < LDCBLK; x += VEC_LEN)

            float bRegs[LDCBLK / VEC_LEN];
            
            #pragma unroll
            for (int ix = 0; ix < LDCBLK / VEC_LEN; ++ix) {
                // float bReg;
                if (pk * A_TILE_WIDTH + i < nk && bx + tx + ix * VEC_LEN < nj) {
                    bRegs[ix] = b[(pk * A_TILE_WIDTH + i) * nj + bx + tx + ix * VEC_LEN];
                } else {
                    bRegs[ix] = 0.0f;
                }
            }

            #pragma unroll
            for (int j = 0; j < A_TILE_WIDTH; ++j) {
                #pragma unroll
                for (int ix = 0; ix < LDCBLK / VEC_LEN; ++ix) {
                    cRegs[ix][j] += bRegs[ix] * aTile[i][j]; // aTile[j][i], j: row, i: col
                }
            }
        }

        __syncthreads();
    }
    
    // store back c
    #pragma unroll
    for (int i = 0; i < A_TILE_WIDTH; ++i) {
        for (int ix = 0; ix < LDCBLK / VEC_LEN; ++ix) {
            if (by + i < ni && bx + tx + ix * VEC_LEN < nj) {
                c[(by + i) * nj + bx + tx + ix * VEC_LEN] = cRegs[ix][i];
            }
        }
    }
}

template <int LDCBLK, int A_TILE_WIDTH, int VEC_LEN>
Mat<float> matMatMult_Volkov_Demmel(Mat<float> const& m1, Mat<float> const& m2)
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
    
    dim3 dimBlock(VEC_LEN, 1, 1);
    dim3 dimGrid(std::ceil(res.cols() / float(LDCBLK)),
                 std::ceil(res.rows() / float(A_TILE_WIDTH)), 1);
    
    auto start = std::chrono::system_clock::now();
    matMatMultKernel_Volkov_Demmel<LDCBLK, A_TILE_WIDTH, VEC_LEN><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());
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

// what if I use float2:

template <int LDCBLK, int A_TILE_WIDTH>
__global__
void matMatMultKernel_Volkov_Demmel_float2(float const* a, float const* b, float* __restrict c,
                                           size_t const ni, size_t const nk, size_t const nj)
{
    constexpr int VEC_LEN = LDCBLK / 2;
    int by = A_TILE_WIDTH * blockIdx.y;
    int bx = LDCBLK * blockIdx.x; // blockDim.x == VEC_LEN
    int tx = threadIdx.x; // [0, VEC_LEN)
    
    float2 cRegs[A_TILE_WIDTH] = {0.0f};
    __shared__ float aTile[A_TILE_WIDTH][A_TILE_WIDTH+1];

    int aRowBase = by + tx / A_TILE_WIDTH;
    
    for (int pk = 0; pk < ceil(nk / float(A_TILE_WIDTH)); ++pk) {
        // TODO load a into shared mem
        // a will be transposed
        // 16/4=4
        constexpr int REP = A_TILE_WIDTH * A_TILE_WIDTH / VEC_LEN;
        constexpr int STEP = VEC_LEN / A_TILE_WIDTH;
        int aCol = pk * A_TILE_WIDTH + tx % A_TILE_WIDTH;
        #pragma unroll
        for (int i = 0; i < REP; ++i) {
            // a[by + tx/16 + i * 4][pk * 16 + tx % 16]
            int aRow = aRowBase + i * STEP;            
            if (aRow < ni && aCol < nk) {
                aTile[tx % A_TILE_WIDTH][i * STEP + tx / A_TILE_WIDTH]
                    = a[aRow * nk + aCol];
            } else {
                aTile[tx % A_TILE_WIDTH][i * STEP + tx / A_TILE_WIDTH]
                    = 0.0f;
            }
        }

        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < A_TILE_WIDTH; ++i) {
            float2 bReg;
            // if (pk * A_TILE_WIDTH + i < nk && bx + tx < nj) {
            bReg = *reinterpret_cast<float2 const*>(&b[(pk * A_TILE_WIDTH + i) * nj + bx + tx * 2]);
            // } else {
            //     bReg = 0.0f;
            // }

            #pragma unroll
            for (int j = 0; j < A_TILE_WIDTH; ++j) {
                cRegs[j].x += bReg.x * aTile[i][j]; // aTile[j][i], j: row, i: col
                cRegs[j].y += bReg.y * aTile[i][j];
            }
        }

        __syncthreads();
    }
    
    // store back c
    #pragma unroll
    for (int i = 0; i < A_TILE_WIDTH; ++i) {
        // if (by + i < ni && bx + tx < nj) {
        *reinterpret_cast<float2*>(&c[(by + i) * nj + bx + tx * 2]) = cRegs[i];
        // }
    }
}

template <size_t TILE_IJ, size_t TILE_K, size_t REGBLK_WIDTH>
__global__
void matMatMultKernel_regblk(float const* a, float const* b, float* __restrict c,
                             size_t const ni, size_t const nk, size_t const nj)
{
    // no bound check for now
    static_assert(TILE_IJ % REGBLK_WIDTH == 0);
    // static_assert(TILE_K % REGBLK_WIDTH == 0);
    static_assert(REGBLK_WIDTH % 4 == 0, "I want to try float4");

    // assert(nk % 4 == 0);
    // assert(nj % 4 == 0);

    // float4 tileA[TILE_IJ][TILE_K / 4];
    // float4 tileB[TILE_K][TILE_IJ / 4];
    __shared__ float tileA[TILE_K][TILE_IJ];
    __shared__ float tileB[TILE_K][TILE_IJ];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int flatTid = ty * blockDim.x + tx;

    float cRegs[REGBLK_WIDTH * REGBLK_WIDTH] = {0.0f};

    // if (ty == 0 && tx == 0 && by == 0 && bx == 0) {
    //     printf("%f %f %f %f\n\n", a[0], a[nk], a[nk * 2], a[nk * 3]);
    // }

    constexpr int THRDS_IN_TB = (TILE_IJ * TILE_IJ) / (REGBLK_WIDTH * REGBLK_WIDTH);
    constexpr int LOAD_INSTR_LEN = 4;
    constexpr int LOAD_REPS = (TILE_IJ * TILE_K) / THRDS_IN_TB;
    for (int pk = 0; pk < ceil(nk / (double)TILE_K); ++pk) {
        // load A and B into shared memory:
        // load A:
        // int col = flatTid % (TILE_K / LOAD_LEN);
        // for (int r = 0; r < LOAD_REPS; ++r) {
        //     int row = flatTid / (TILE_K / LOAD_LEN) + r * (THRDS_IN_TB / (TILE_K / LOAD_LEN));
        //     tileA[row][col]
        //         = static_cast<float4 const*>(a)[(by * TILE_IJ + row) * (nk/LOAD_LEN) + pk * (TILE_K / LOAD_LEN) + col];
        // }
        // need to transpose A when loading from shared mem, so cannot use float4
        int col = flatTid % TILE_K;
        for (int r = 0; r < LOAD_REPS; ++r) {
            // a[by * TILE_IJ + row][pk * TILE_K + col]
            int row = flatTid / TILE_K + r * (THRDS_IN_TB / TILE_K);
            tileA[col][row] = a[(by * TILE_IJ + row) * nk + pk * TILE_K + col]; // transposed
        }
        // load B:
        // col = flatTid % (TILE_IJ / LOAD_LEN);
        // for (int r = 0; r < LOAD_REPS; ++r) {
        //     int row = flatTid / (TILE_IJ / LOAD_LEN) + r * (THRDS_IN_TB / (TILE_IJ / LOAD_LEN));
        //     tileB[row][col]
        //         = static_cast<float4 const*>(a)[(pk * TILE_IJ + row) * (nj/LOAD_LEN) + bx * (TILE_K / LOAD_LEN) + col];
        // }
        col = flatTid % TILE_IJ;
        for (int r = 0; r < LOAD_REPS; ++r) {
            int row = flatTid / TILE_IJ + r * (THRDS_IN_TB / TILE_IJ);
            tileB[row][col] = b[(pk * TILE_K + row) * nj + bx * TILE_IJ + col];
        }
        
        __syncthreads();

        // compute c
        float4 aRegs[REGBLK_WIDTH / LOAD_INSTR_LEN];
        float4 bRegs[REGBLK_WIDTH / LOAD_INSTR_LEN];
        for (int k = 0; k < TILE_K; ++k) { // 8
            #pragma unroll
            for (int i = 0; i < REGBLK_WIDTH / LOAD_INSTR_LEN; ++i) { // 2
                aRegs[i] = *reinterpret_cast<float4*>(&tileA[k][ty * REGBLK_WIDTH + i * LOAD_INSTR_LEN]);
                // if (pk == 0 && ty == 0 && tx == 0 && by == 0 && bx == 0) {
                //     printf("%f %f %f %f\n", aRegs[i].x, aRegs[i].y, aRegs[i].z, aRegs[i].w);
                // }
            }
            #pragma unroll
            for (int i = 0; i < REGBLK_WIDTH / LOAD_INSTR_LEN; ++i) { // 2
                bRegs[i] = *reinterpret_cast<float4*>(&tileB[k][tx * REGBLK_WIDTH + i * LOAD_INSTR_LEN]);
            }

            #pragma unroll
            for (int i = 0; i < REGBLK_WIDTH / LOAD_INSTR_LEN; ++i) { // 2
                float4 aReg = aRegs[i];
                #pragma unroll
                for (int j = 0; j < REGBLK_WIDTH / LOAD_INSTR_LEN; ++j) { // 2
                    float4 bReg = bRegs[j];
                    cRegs[(i * LOAD_INSTR_LEN + 0) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 0] += aReg.x * bReg.x;
                    cRegs[(i * LOAD_INSTR_LEN + 0) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 1] += aReg.x * bReg.y;
                    cRegs[(i * LOAD_INSTR_LEN + 0) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 2] += aReg.x * bReg.z;
                    cRegs[(i * LOAD_INSTR_LEN + 0) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 3] += aReg.x * bReg.w;

                    cRegs[(i * LOAD_INSTR_LEN + 1) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 0] += aReg.y * bReg.x;
                    cRegs[(i * LOAD_INSTR_LEN + 1) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 1] += aReg.y * bReg.y;
                    cRegs[(i * LOAD_INSTR_LEN + 1) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 2] += aReg.y * bReg.z;
                    cRegs[(i * LOAD_INSTR_LEN + 1) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 3] += aReg.y * bReg.w;

                    cRegs[(i * LOAD_INSTR_LEN + 2) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 0] += aReg.z * bReg.x;
                    cRegs[(i * LOAD_INSTR_LEN + 2) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 1] += aReg.z * bReg.y;
                    cRegs[(i * LOAD_INSTR_LEN + 2) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 2] += aReg.z * bReg.z;
                    cRegs[(i * LOAD_INSTR_LEN + 2) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 3] += aReg.z * bReg.w;

                    cRegs[(i * LOAD_INSTR_LEN + 3) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 0] += aReg.w * bReg.x;
                    cRegs[(i * LOAD_INSTR_LEN + 3) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 1] += aReg.w * bReg.y;
                    cRegs[(i * LOAD_INSTR_LEN + 3) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 2] += aReg.w * bReg.z;
                    cRegs[(i * LOAD_INSTR_LEN + 3) * REGBLK_WIDTH + j * LOAD_INSTR_LEN + 3] += aReg.w * bReg.w;
                }
            }
        }
        __syncthreads();
    }

    // write C back
    // non-coalesced
    for (int i = 0; i < REGBLK_WIDTH; ++i) {
        for (int j = 0; j < REGBLK_WIDTH; ++j) {
            c[(by * TILE_IJ + ty * REGBLK_WIDTH + i) * nj + (bx * TILE_IJ + tx * REGBLK_WIDTH + j)]
                = cRegs[i * REGBLK_WIDTH + j];
        }
    }
}

template <size_t TILE_IJ, size_t TILE_K, size_t REGBLK_WIDTH>
Mat<float> matMatMult_regblk(Mat<float> const& m1, Mat<float> const& m2)
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

    dim3 dimBlock(TILE_IJ / REGBLK_WIDTH, TILE_IJ / REGBLK_WIDTH, 1);
    dim3 dimGrid(std::ceil(res.cols() / double(TILE_IJ)),
                 std::ceil(res.rows() / double(TILE_IJ)), 1);
    
    auto start = std::chrono::system_clock::now();
    matMatMultKernel_regblk<TILE_IJ, TILE_K, REGBLK_WIDTH><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());
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
