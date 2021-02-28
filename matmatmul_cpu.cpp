#include <cstddef>
#include <string>
#include <random>
#include <fmt/core.h>
#include "TimerGuard.h"
#include <boost/align/aligned_allocator.hpp>
//#ifdef __AVX2__
#include <x86intrin.h>
//#endif

template <typename T>
inline void doNotOptimize(T const& value)
{
    asm volatile("" : : "r,m"(value) : "memory");
}

int BLOCKWIDTH = 48; // assume to be 8x

class Mat
{
public:
    Mat(size_t r, size_t c)
        : rows_(r), cols_(c), data_(r * c)
    {
    }

    Mat(size_t r, size_t c, float e)
        : rows_(r), cols_(c), data_(r * c, e)
    {
    }

    // Mat& operator=(Mat const& rhs) = default;

    float* operator[](size_t i)
    {
        return data_.data() + i * cols_;
    }

    const float* operator[](size_t i) const
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
    float* data() { return data_.data(); }
    float const* data() const { return data_.data(); }


    size_t rows_, cols_;
    std::vector<float, boost::alignment::aligned_allocator<float, 32>> data_;
};

bool matNearlyEqual(Mat const& m1, Mat const& m2)
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

Mat matMult_cpu_ijk(Mat const& a, Mat const& b)
{
    int n1 = a.rows(), n2 = a.cols(), n3 = b.cols();
    Mat out(n1, n3);
    #pragma omp parallel for
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n3; ++j) {
            float c = 0.0f;
            for (int k = 0; k < n2; ++k) {
                c += a[i][k] * b[k][j];
            }
            out[i][j] = c;
        }
    }
    
    return out;
}

Mat matMult_cpu_jik(Mat const& a, Mat const& b)
{
    int n1 = a.rows(), n2 = a.cols(), n3 = b.cols();
    Mat out(n1, n3);
    #pragma omp parallel for
    for (int j = 0; j < n3; ++j) {
        for (int i = 0; i < n1; ++i) {
            float c = 0.0f;
            for (int k = 0; k < n2; ++k) {
                c += a[i][k] * b[k][j];
            }
            out[i][j] = c;
        }
    }
    
    return out;
}

Mat matMult_cpu_kij(Mat const& a, Mat const& b)
{
    int n1 = a.rows(), n2 = a.cols(), n3 = b.cols();
    Mat out(n1, n3, 0.0f);

    for (int k = 0; k < n2; ++k) {
        #pragma omp parallel for
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n3; ++j) {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return out;
}

Mat matMult_cpu_ikj(Mat const& a, Mat const& b)
{
    Mat out(a.rows(), b.cols(), 0.0f);
    for (int i = 0; i < a.rows(); ++i) {
        for (int k = 0; k < a.cols(); ++k) {
            for (int j = 0; j < b.cols(); ++j) {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return out;
}

Mat matMult_cpu_ijk_tiled(Mat const& a, Mat const& b)
{
    Mat out(a.rows(), b.cols(), 0.0f);
#pragma omp parallel for
    for (size_t bi = 0; bi < a.rows() / BLOCKWIDTH; ++bi) {
        for (size_t bj = 0; bj < b.cols() / BLOCKWIDTH; ++bj) {
            for (size_t bk = 0; bk < a.cols() / BLOCKWIDTH; ++bk) {

                for (size_t i = bi * BLOCKWIDTH; i < (bi + 1) * BLOCKWIDTH; ++i) {
                    for (size_t j = bj * BLOCKWIDTH; j < (bj + 1) * BLOCKWIDTH; ++j) {
                        float c = out[i][j];
                        for (size_t k = bk * BLOCKWIDTH; k < (bk + 1) * BLOCKWIDTH; ++k) {
                            c += a[i][k] * b[k][j];
                        }
                        out[i][j] = c;
                    }
                }
                
            }

            // in case of a.cols() is not times of BLOCKWIDTH
            size_t kk = a.cols() / BLOCKWIDTH * BLOCKWIDTH;
            if (kk < a.cols()) {    
                for (size_t i = bi * BLOCKWIDTH; i < (bi + 1) * BLOCKWIDTH; ++i) {
                    for (size_t j = bj * BLOCKWIDTH; j < (bj + 1) * BLOCKWIDTH; ++j) {
                        float c = out[i][j];
                        for (size_t k = kk; k < a.cols(); ++k) {
                            c += a[i][k] * b[k][j];
                        }
                        out[i][j] = c;
                    }
                }
            }
        }

        // in case of b.cols() is not times of BLOCKWIDTH
        size_t jj = b.cols() / BLOCKWIDTH * BLOCKWIDTH;
        if (jj < b.cols()) {
            for (size_t i = bi * BLOCKWIDTH; i < (bi + 1) * BLOCKWIDTH; ++i) {
                for (size_t j = jj; j < b.cols(); ++j) {
                    float c = 0.0f;
                    for (size_t k = 0; k < a.cols(); ++k) {
                        c += a[i][k] * b[k][j];
                    }
                    out[i][j] = c;
                }
            }
        }
    }
    size_t ii = a.rows() / BLOCKWIDTH * BLOCKWIDTH;

    #pragma omp parallel for
    for (size_t i = ii; i < a.rows(); ++i) {
        for (size_t j = 0; j < b.cols(); ++j) {
            float c = 0.0f;
            for (size_t k = 0; k < a.cols(); ++k) {
                c += a[i][k] * b[k][j];
            }
            out[i][j] = c;
        }
    }

    return out;
}

template<size_t BI, size_t BK, size_t BJ>
struct IKJ_DefaultKernel
{
    void operator()(float const* __restrict a, float const* __restrict b,  float* __restrict c,
                    size_t n1, size_t n2, size_t n3)
    {
        for (size_t i = 0; i < BI; ++i) {
            for (size_t k = 0; k < BK; ++k) {
                for (size_t j = 0; j < BJ; ++j) {
                    // c[i][j] += a[i][k] * b[k][j]
                    c[i * n3 + j] += a[i * n2 + k] * b[k * n3 + j];
                }
            }
        }
    }
};


template<size_t BI, size_t BK, size_t BJ>
struct IKJ_AVX2Kernel
{
    void operator()(float const* __restrict a, float const* __restrict b,  float* __restrict c,
                    size_t n1, size_t n2, size_t n3)
    {
        constexpr size_t UNROLL_I = 2;
        // assume BI is multiple of 8
        __m256 c_lines[UNROLL_I * BJ / (32 / sizeof(float))]; // 2 rows of c
        for (size_t i = 0; i < BI; i += UNROLL_I) {
            // load c:            
            for (size_t ii = 0; ii < UNROLL_I; ++ii) {
                for (size_t jj = 0; jj < BJ; jj += (32 / sizeof(float))) {
                    // c[i + ii][jj]
                    c_lines[ii * BJ / (32 / sizeof(float)) + jj / (32 / sizeof(float))] = _mm256_loadu_ps(&c[(i + ii) * n3 + jj]);
                }
            }

            for (size_t k = 0; k < BK; ++k) {
                // a[i][k], a[i + 1][k]
                // __m256 a1 = _mm256_load_ps(a + i * n2 + k);
                // __m256 a2 = _mm256_load_ps(a + (i + 1) * n2 + k);

                __m256 b_line[BJ / (32 / sizeof(float))];
                for (size_t j = 0; j < BJ / (32 / sizeof(float)); ++j) {
                    b_line[j] = _mm256_loadu_ps(b + k * n3 + j * (32 / sizeof(float)));
                }

                // for (size_t j = 0; j < BJ / (32 / sizeof(float)); ++j) {
                //     c_lines[j] = _mm256_fmadd_ps(a1, b_line[j], c_lines[j]);
                //     c_lines[]
                // }
                for (size_t ii = 0; ii < UNROLL_I; ++ii) {
                    // __m256 a_vec = _mm256_loadu_ps(a + (i + ii) * n2 + k);
                    __m256 a_vec = _mm256_broadcast_ss(a + (i + ii) * n2 + k);
                    for (size_t j = 0; j < BJ / (32 / sizeof(float)); ++j) {
                        c_lines[ii * BJ / (32 / sizeof(float)) + j] = _mm256_fmadd_ps(a_vec, b_line[j],
                                                                                       c_lines[ii * BJ / (32 / sizeof(float)) + j]);
                        // c_lines[ii * BJ / (32 / sizeof(float)) + j] = _mm256_add_ps(_mm256_mul_ps(a_vec, b_line[j]),
                        //                                                             c_lines[ii * BJ / (32 / sizeof(float)) + j]);
                    }
                }
            }

            // store c:
            for (size_t ii = 0; ii < UNROLL_I; ++ii) {
                for (size_t jj = 0; jj < BJ / (32 / sizeof(float)); ++jj) {
                    // c[i + ii][jj]
                    _mm256_storeu_ps(c + (i + ii) * n3 + jj * (32 / sizeof(float)), c_lines[ii * BJ / (32 / sizeof(float)) + jj]);
                }
            }
        }
    }
};



// template <size_t BI, size_t BK, size_t BJ, typename Kernel = IKJ_DefaultKernel<BI, BK, BJ>>
template <size_t BI, size_t BK, size_t BJ, template<size_t, size_t, size_t > typename Kernel = IKJ_DefaultKernel>
Mat matMult_cpu_ikj_tiled(Mat const& a, Mat const& b)
{
    // easier to simd
    Mat out(a.rows(), b.cols(), 0.0f);
    #pragma omp parallel for
    for (size_t bi = 0; bi < a.rows() / BI; ++bi) {
        for (size_t bk = 0; bk < a.cols() / BK; ++bk) {
            for (size_t bj = 0; bj < b.cols() / BJ; ++bj) {

                // place to call the tiled kernel
                Kernel<BI, BK, BJ> kernel;
                kernel(&a[bi * BI][bk * BK], &b[bk * BK][bj * BJ], &out[bi * BI][bj * BJ],
                       a.rows(), a.cols(), b.cols());
                // for (size_t i = bi * BI; i < (bi + 1) * BI; ++i) {
                //     for (size_t k = bk * BK; k < (bk + 1) * BK; ++k) {
                //         for (size_t j = bj * BJ; j < (bj + 1) * BJ; ++j) {
                //             out[i][j] += a[i][k] * b[k][j];
                //         }
                //     }
                // }
                
            }
            size_t jj = b.cols() / BJ * BJ;
            if (jj < b.cols()) {
                for (size_t i = bi * BI; i < (bi + 1) * BI; ++i) {
                    for (size_t k = bk * BK; k < (bk + 1) * BK; ++k) {
                        for (size_t j = jj; j < b.cols(); ++j) {
                            out[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
        size_t kk = a.cols() / BK * BK;
        if (kk < a.cols()) {
            for (size_t i = bi * BI; i < (bi + 1) * BI; ++i) {
                for (size_t k = kk; k < a.cols(); ++k) {
                    for (size_t j = 0; j < b.cols(); ++j) {
                        out[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
    }
    size_t ii = a.rows() / BI * BI;
    #pragma omp parallel for
    for (size_t i = ii; i < a.rows(); ++i) {
        for (size_t k = 0; k < a.cols(); ++k) {
            for (size_t j = 0; j < b.cols(); ++j) {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return out;
}

Mat matMult_cpu_kij_tiled(Mat const& a, Mat const& b)
{
    Mat out(a.rows(), b.cols(), 0.0f);
    
    for (size_t bk = 0; bk < a.cols() / BLOCKWIDTH; ++bk) {
        for (size_t bi = 0; bi < a.rows() / BLOCKWIDTH; ++bi) {   
            for (size_t bj = 0; bj < b.cols() / BLOCKWIDTH; ++bj) {
                
                for (size_t k = bk * BLOCKWIDTH; k < (bk + 1) * BLOCKWIDTH; ++k) {            
                    for (size_t i = bi * BLOCKWIDTH; i < (bi + 1) * BLOCKWIDTH; ++i) {    
                        for (size_t j = bj * BLOCKWIDTH; j < (bj + 1) * BLOCKWIDTH; ++j) {
                            out[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
                
            }
            size_t jj = b.cols() / BLOCKWIDTH * BLOCKWIDTH;
            
            if (jj < b.cols()) {
                for (size_t k = bk * BLOCKWIDTH; k < (bk + 1) * BLOCKWIDTH; ++k) {
                    #pragma omp parallel for
                    for (size_t i = bi * BLOCKWIDTH; i < (bi + 1) * BLOCKWIDTH; ++i) {
                        for (size_t j = jj; j < b.cols(); ++j) {
                            out[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }

        size_t ii = a.rows() / BLOCKWIDTH * BLOCKWIDTH;
        for (size_t k = bk * BLOCKWIDTH; k < (bk + 1) * BLOCKWIDTH; ++k) {
            #pragma omp parallel for
            for (size_t i = ii; i < a.rows(); ++i) {                
                for (size_t j = 0; j < b.cols(); ++j) {
                    out[i][j] += a[i][k] * b[k][j];
                }
            }
        }

    }
    
    size_t kk = a.cols() / BLOCKWIDTH * BLOCKWIDTH;
    if (kk < a.cols()) {
        for (size_t k = kk; k < a.cols(); ++k) {
            #pragma omp parallel for
            for (size_t i = 0; i < a.rows(); ++i) {
                for (size_t j = 0; j < b.cols(); ++j) {
                    out[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    return out;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fmt::print(stderr, "not enough arguments\n");
        std::exit(1);
    }
    size_t N1 = std::stoul(argv[1]),
        N2 = std::stoul(argv[2]),
        N3 = std::stoul(argv[3]);
    if (argc == 5) {
        BLOCKWIDTH = std::stoul(argv[4]);
    }
    Mat mat1(N1, N2), mat2(N2, N3);
    Mat out1(N1, N3), out2(N1, N3), out3(N1, N3), out4(N1, N3), out5(N1, N3), out6(N1, N3), out7(N1, N3);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> distrib1{0.0f, 10.0f};
    std::normal_distribution<float> distrib2{5.0f, 2.0f};

    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            mat1[i][j] = distrib1(gen);
        }
    }
    for (size_t i = 0; i < N2; ++i) {
        for (size_t j = 0; j < N3; ++j) {
            mat2[i][j] = distrib2(gen);
        }
    }

    
    // {
    //     TimerGuard tg{"matMult_cpu_ijk:"};
    //     out1 = matMult_cpu_ijk(mat1, mat2);
    // }
    // {
    //     TimerGuard tg{"matMult_cpu_jik:"};
    //     out2 = matMult_cpu_jik(mat1, mat2);
    // }
    // {
    //     TimerGuard tg{"matMult_cpu_kij:"};
    //     out3 = matMult_cpu_kij(mat1, mat2);
    // }
    {
        TimerGuard tg{"matMult_cpu_ikj:"};
        out4 = matMult_cpu_ikj(mat1, mat2);
        doNotOptimize(out4);
    }
    {
        TimerGuard tg{"matMult_cpu_ijk_tiled:"};
        out5 = matMult_cpu_ijk_tiled(mat1, mat2);
        doNotOptimize(out5);
    }
    {
        TimerGuard tg{"matMult_cpu_ikj_tiled:"};
        out6 = matMult_cpu_ikj_tiled<48, 48, 48>(mat1, mat2);
        doNotOptimize(out6);
    }
    {
        TimerGuard tg{"matMult_cpu_ikj_tiled manual avx2:"};
        out7 = matMult_cpu_ikj_tiled<48, 48, 48, IKJ_AVX2Kernel>(mat1, mat2);
        doNotOptimize(out7);
    }
    // {
    //     TimerGuard tg{"matMult_cpu_kij_tiled:"};
    //     out7 = matMult_cpu_ikj_tiled(mat1, mat2);
    // }

    // if (!matNearlyEqual(out1, out2)) {
    //     fmt::print(stderr, "1 2 error\n");
    // }
    // if (!matNearlyEqual(out1, out3)) {
    //     fmt::print(stderr, "1 3 error\n");
    // }
    if (!matNearlyEqual(out4, out5)) {
        fmt::print(stderr, "4 5 error\n");
    }
    if (!matNearlyEqual(out4, out6)) {
        fmt::print(stderr, "4 6 error\n");
    }
    if (!matNearlyEqual(out4, out7)) {
        fmt::print(stderr, "4 7 error\n");
    }
}


