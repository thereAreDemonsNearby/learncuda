#include "TimerGuard.h"
#include <vector>
#include <cstddef>
#include <cmath>
#include <random>
#include <iostream>
#include <exception>

constexpr size_t TILE_WIDTH = 32;

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
    std::vector<T> data_;
};

template <typename T>
bool matNearlyEqual(Mat<T> const& m1, Mat<T> const& m2)
{
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        std::cerr << "size wrong\n";
        return false;
    }

    for (size_t i = 0; i < m1.rows(); ++i) {
        for (size_t j = 0; j < m1.cols(); ++j) {
            if (std::fabs(m1[i][j] - m2[i][j]) > 0.1) {
                std::cerr << m1[i][j] << ' ' << m2[i][j] << '\n';
                return false;
            }
        }
    }
    return true;
}


template <typename T>
Mat<T> matMatMult_CPU_trivial(Mat<T> const& m1, Mat<T> const& m2);

template <typename T>
Mat<T> matMatMult_CPU_tiled(Mat<T> const& m1, Mat<T> const& m2);

template <typename T>
Mat<T> matMatMult_GPU_trivial(Mat<T> const& m1, Mat<T> const& m2);

template <typename T>
Mat<T> matMatMult_GPU_tiled(Mat<T> const& m1, Mat<T> const& m2);

// template <>
// Mat<float> matMatMult_GPU_tiled(Mat<float> const& m1, Mat<float> const& m2);

int main(int argc, char* argv[])
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

    Mat<float> res(N1, N3);

    Mat<float> res2(N1, N3);

    Mat<float> res3(N1, N3);

    Mat<float> res4(N1, N3);

    // do some random init first
    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            m1[i][j] = urd(dre);
            // m1[i][j] = 1.0;
        }
    }

    for (size_t i = 0; i < N2; ++i) {
        for (size_t j = 0; j < N3; ++j) {
            m2[i][j] = urd(dre);
            // m2[i][j] = 1.0;
        }
    }

    // {
    //     TimerGuard tg("matMatMult_CPU_trivial");
    //     res = matMatMult_CPU_trivial(m1, m2);
    // }
    // {
    //     TimerGuard tg("matMatMult_CPU_tiled");
    //     res4 = matMatMult_CPU_tiled(m1, m2);
    // }

    {
        TimerGuard tg("matMatMult_GPU_trivial");
        res2 = matMatMult_GPU_trivial(m1, m2);
    }

    {
        TimerGuard tg("matMatMult_GPU_tiled");
        res3 = matMatMult_GPU_tiled(m1, m2);
    }

    if (!matNearlyEqual(res, res2)) {
        std::cerr << "gpu trivial Wrong" << std::endl;
    }
    if (!matNearlyEqual(res, res3)) {
        std::cerr << "gpu tiled Wrong" << std::endl;
    }

    std::cout << '\n';
}

template <typename T>
Mat<T> matMatMult_CPU_trivial(Mat<T> const& m1, Mat<T> const& m2)
{
    if (m1.cols() != m2.rows()) {
        throw 0;
    }
    Mat<T> res(m1.rows(), m2.cols());
    auto ni = m1.rows();
    auto nj = m2.cols();
    auto nk = m1.cols();

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            T sum = 0;
            for (size_t k = 0; k < nk; ++k) {
                sum = sum + m1[i][k] * m2[k][j];
            }
            res[i][j] = sum;
        }
    }

    return res;
}

template <typename T>
Mat<T> matMatMult_CPU_tiled(Mat<T> const& m1, Mat<T> const& m2)
{
    Mat<T> res(m1.rows(), m2.cols());
    if (m1.cols() != m2.rows()) {
        throw 0;
    }

    // 64/4 = 16, so 16 floats per cache line
    size_t blockWidth = 32;
    for (size_t bi = 0; bi < res.rows() / blockWidth; ++bi) {
        for (size_t bj = 0; bj < res.cols() / blockWidth; ++bj) {
            for (size_t bk = 0; bk < m1.cols() / blockWidth; ++bk) {
                
                for (size_t i = 0; i < blockWidth; ++i) {
                    for (size_t j = 0; j < blockWidth; ++j) {
                        T s = res[bi * blockWidth + i][bj * blockWidth + j];
                        for (size_t k = 0; k < blockWidth; ++k) {
                            s = s + m1[bi * blockWidth + i][bk * blockWidth + k]
                                * m2[bk * blockWidth + k][bj * blockWidth + j];
                        }
                        res[bi * blockWidth + i][bj * blockWidth + j] = s;
                    }
                }                
            }
            // if m1.cols() is not times of blockWidth:
            size_t kk = (m1.cols() / blockWidth) * blockWidth;
            if (kk < m1.cols()) {
                for (size_t i = 0; i < blockWidth; ++i) {
                    for (size_t j = 0; j < blockWidth; ++j) {
                        T s = res[bi * blockWidth + i][bj * blockWidth + j];
                        for (size_t k = kk; k < m1.cols(); ++k) {
                            s = s + m1[bi * blockWidth + i][k]
                                * m2[k][bj * blockWidth + j];
                        }
                        res[bi * blockWidth + i][bj * blockWidth + j] = s;
                    }
                }
            }
        }
        // if res.cols() (aka m2.cols()) is not times of blockWidth
        size_t jj = (res.cols() / blockWidth) * blockWidth;
        if (jj < res.cols()) {
            for (size_t i = bi * blockWidth; i < (bi + 1) * blockWidth; ++i) {
                for (size_t j = jj; j < res.cols(); ++j) {
                    T s = 0; // todo
                    for (size_t k = 0; k < m1.cols(); ++k) {
                        s = s + m1[i][k] * m2[k][j];
                    }
                    res[i][j] = s;
                }
            }
        }
    }
    size_t ii = (res.rows() / blockWidth) * blockWidth;
    for (size_t i = ii; i < res.rows(); ++i) {
        for (size_t j = 0; j < res.cols(); ++j) {
            T s = 0;
            for (size_t k = 0; k < m1.cols(); ++k) {
                s = s + m1[i][k] * m2[k][j];
            }
            res[i][j] = s;
        }
    }

    return res;
}

// m1[n1][n2], m2[n2][n3], res[n1][n3]
template <typename T>
__global__
void matMatMultKernel1(T* m1, T* m2, T* res,
                       size_t n1, size_t n2, size_t n3)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n1 && col < n3) {
        T s = 0.0f;
        for (size_t k = 0; k < n2; ++k) {
            s = s + m1[row * n2 + k] * m2[k * n3 + col];
        }
        res[row * n3 + col] = s;
    }
}

template <typename T>
__global__
void matMatMultKernel2(T* m1, T* m2, T* res,
                       size_t n1, size_t n2, size_t n3)
{
    __shared__ T mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ T nds[TILE_WIDTH][TILE_WIDTH];
    
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    size_t row = by * TILE_WIDTH + ty;
    size_t col = bx * TILE_WIDTH + tx;

    T s = 0;
    for (size_t ph = 0; ph < std::ceil(n2 / (T)TILE_WIDTH); ++ph) {
        if (row < n1 && (ph * TILE_WIDTH + tx) < n2)
            mds[ty][tx] = m1[row * n2 + ph * TILE_WIDTH + tx]; // m1[row][ph * TILE_WIDTH + tx]
        else
            mds[ty][tx] = 0.0f;
        if ((ph * TILE_WIDTH + ty) < n2 && col < n3)
            nds[ty][tx] = m2[(ph * TILE_WIDTH + ty) * n3 + col]; // m2[ph * TILE_WIDTH + ty][col]
        else
            nds[ty][tx] = 0.0f;
        
        __syncthreads(); // sync threads in one block
        
        for (size_t k = 0; k < TILE_WIDTH; ++k) {
            s = s + mds[ty][k] * nds[k][tx];
        }
        __syncthreads();
    }

    if (row < n1 && col < n3)
        res[row * n3 + col] = s;
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
    matMatMultKernel1<T><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());    

    if (err = cudaMemcpy(res.data(), d_res, res.size() * sizeof(T), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
    }

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);

    return res;
}

template <typename T>
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

    double p = TILE_WIDTH;
    dim3 dimBlock(p, p, 1);
    dim3 dimGrid(ceil(res.cols() / p),
                 ceil(res.rows() / p), 1);
    matMatMultKernel2<T><<<dimGrid, dimBlock>>>(d_m1, d_m2, d_res, m1.rows(), m1.cols(), m2.cols());    

    if (err = cudaMemcpy(res.data(), d_res, res.size() * sizeof(T), cudaMemcpyDeviceToHost);
        err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
    }

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);

    return res;
}
