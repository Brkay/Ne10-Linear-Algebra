
#include <cmath>
#include <Util/TimerUtil.h>
#include <Thread/Logger.h>
#include <Util/Ne10Util.h>
#include <Util/MatrixUtil.h>
#include <omp.h>

int32_t Ne10Util::MatrixMultFloat(const float *A, const size_t &n, const size_t &m1, const float *B, const size_t &m2,
                                  const size_t &k, float *C)
{
    if (m1 != m2)
    {
        return -1;
    }
    constexpr uint32 blockSize = 4;
    //  n, m, k must be multiples of 4.

    // rows of A

    omp_set_num_threads(4); // 4 different cores!
    pinThreadToCore();

// A -> (nxm)
// B -> (mxk)
// C -> (nxk)
#pragma omp parallel
    {
// Define variables in the inner-scope so that they become private to each thread!
#pragma omp for
        for (size_t i_idx = 0; i_idx < n; i_idx += blockSize) // Loop over rows of C (and A)
        {
            for (size_t k_idx = 0; k_idx < k; k_idx += blockSize) // Loop over columns of C (and B)
            {
                float32x4_t A0;
                float32x4_t A1;
                float32x4_t A2;
                float32x4_t A3;

                // rows of B
                float32x4_t B0;
                float32x4_t B1;
                float32x4_t B2;
                float32x4_t B3;

                // rows of C

                // Zero accumulators for C values
                float32x4_t C0 = vmovq_n_f32(0);
                float32x4_t C1 = vmovq_n_f32(0);
                float32x4_t C2 = vmovq_n_f32(0);
                float32x4_t C3 = vmovq_n_f32(0);
                for (size_t j_idx = 0; j_idx < m1; j_idx += blockSize) // Multiply and accumulate over block of A and B
                {
                    // Compute base index to 4x4 block
                    const size_t A_idx = i_idx * m1 + j_idx; // A_{i,j} sub-block matrix that is 4x4.
                    const size_t B_idx = j_idx * k + k_idx;  // B_{j,k} sub-block matrix that is 4x4.
                    // C_{i,k} (4x4) = Sum_{k = 0 to 4} (A_{i,j} B_{j,k}).
                    // We will calculate each sub-product with ne10 instructions.
                    // At each loop, four different row calculations are made for each sub-product. Then, they are summed
                    // together.

                    A0 = vld1q_f32(A + A_idx); // Travel Rows of A_{i,k}
                    A1 = vld1q_f32(A + A_idx + m1);
                    A2 = vld1q_f32(A + A_idx + m1 * 2);
                    A3 = vld1q_f32(A + A_idx + m1 * 3);

                    B0 = vld1q_f32(B + B_idx);
                    B1 = vld1q_f32(B + B_idx + k);
                    B2 = vld1q_f32(B + B_idx + k * 2);
                    B3 = vld1q_f32(B + B_idx + k * 3);

                    C0 = vfmaq_laneq_f32(C0, B0, A0, 0);
                    C0 = vfmaq_laneq_f32(C0, B1, A0, 1);
                    C0 = vfmaq_laneq_f32(C0, B2, A0, 2);
                    C0 = vfmaq_laneq_f32(C0, B3, A0, 3);

                    C1 = vfmaq_laneq_f32(C1, B0, A1, 0);
                    C1 = vfmaq_laneq_f32(C1, B1, A1, 1);
                    C1 = vfmaq_laneq_f32(C1, B2, A1, 2);
                    C1 = vfmaq_laneq_f32(C1, B3, A1, 3);

                    C2 = vfmaq_laneq_f32(C2, B0, A2, 0);
                    C2 = vfmaq_laneq_f32(C2, B1, A2, 1);
                    C2 = vfmaq_laneq_f32(C2, B2, A2, 2);
                    C2 = vfmaq_laneq_f32(C2, B3, A2, 3);

                    C3 = vfmaq_laneq_f32(C3, B0, A3, 0);
                    C3 = vfmaq_laneq_f32(C3, B1, A3, 1);
                    C3 = vfmaq_laneq_f32(C3, B2, A3, 2);
                    C3 = vfmaq_laneq_f32(C3, B3, A3, 3);
                }
                const size_t C_idx = k * i_idx + k_idx; // C{i,k}
                vst1q_f32(C + C_idx, C0);
                vst1q_f32(C + C_idx + k, C1); // You need to write by rows, hole row must be traversed!!
                vst1q_f32(C + C_idx + k * 2, C2);
                vst1q_f32(C + C_idx + k * 3, C3);
            }
        }
    }
    return 0;
}

int32_t Ne10Util::MatrixMultComplexFloat(const float *A_real, const float *A_imag, const size_t &n, const size_t &m1,
                                         const float *B_real, const float *B_imagFloat, const size_t &m2, const size_t &k, float *C_real, float *C_imag)
{
    // Important! There are 32 128-bit NEON registers for ARM v8-A AArch64. We exactly use 32 registers! Since we do not
    // have more registers, we need to overwrite some of the used register values. This hardens to readibility.
    // Check the paper "Acceleration of complex matrix multiplication using arbitrary precision floating-point arithmetic"
    if (m1 != m2)
    {
        return -1;
    }
    constexpr uint32 blockSize = 4;

    float32x4_t A0_real, A0_imag;
    float32x4_t A1_real, A1_imag;
    float32x4_t A2_real, A2_imag;
    float32x4_t A3_real, A3_imag;

    float32x4_t B0_real, B0_imag;
    float32x4_t B1_real, B1_imag;
    float32x4_t B2_real, B2_imag;
    float32x4_t B3_real, B3_imag;

    float32x4_t C0_real, C0_imag;
    float32x4_t C1_real, C1_imag;
    float32x4_t C2_real, C2_imag;
    float32x4_t C3_real, C3_imag;

    // A -> (nxm)
    // B -> (mxk)
    // C -> (nxk)

    for (size_t i_idx = 0; i_idx < n; i_idx += blockSize)
    {
        for (size_t k_idx = 0; k_idx < k; k_idx += blockSize)
        {

            C0_real = vmovq_n_f32(0);
            C0_imag = vmovq_n_f32(0);

            C1_real = vmovq_n_f32(0);
            C1_imag = vmovq_n_f32(0);

            C2_real = vmovq_n_f32(0);
            C2_imag = vmovq_n_f32(0);

            C3_real = vmovq_n_f32(0);
            C3_imag = vmovq_n_f32(0);

            // Needed for 3M method (Karatsuba). We are doing complex multiplications, so we need cross multiplications!
            float32x4_t temp0_A;
            float32x4_t temp0_B;

            float32x4_t temp1_A;
            float32x4_t temp1_B;

            float32x4_t temp2_A;
            float32x4_t temp2_B;

            float32x4_t temp3_A;
            float32x4_t temp3_B;

            for (size_t j_idx = 0; j_idx < m1; j_idx += blockSize)
            {
                const size_t A_idx = i_idx * m1 + j_idx;
                const size_t B_idx = j_idx * k + k_idx;

                A0_real = vld1q_f32(A_real + A_idx); // Travel Rows of A_{i,k}
                A1_real = vld1q_f32(A_real + A_idx + m1);
                A2_real = vld1q_f32(A_real + A_idx + m1 * 2);
                A3_real = vld1q_f32(A_real + A_idx + m1 * 3);

                A0_imag = vld1q_f32(A_imag + A_idx);
                A1_imag = vld1q_f32(A_imag + A_idx + m1);
                A2_imag = vld1q_f32(A_imag + A_idx + m1 * 2);
                A3_imag = vld1q_f32(A_imag + A_idx + m1 * 3);

                B0_real = vld1q_f32(B_real + B_idx);
                B1_real = vld1q_f32(B_real + B_idx + k);
                B2_real = vld1q_f32(B_real + B_idx + k * 2);
                B3_real = vld1q_f32(B_real + B_idx + k * 3);

                B0_imag = vld1q_f32(B_imagFloat + B_idx);
                B1_imag = vld1q_f32(B_imagFloat + B_idx + k);
                B2_imag = vld1q_f32(B_imagFloat + B_idx + k * 2);
                B3_imag = vld1q_f32(B_imagFloat + B_idx + k * 3);

                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///
                /// CALCULATE SUMMATION AND CROSS MULTIPLICATION
                // First calculate the cross value because there are not enough registers!
                // Summation part
                temp0_A = vaddq_f32(A0_imag, A0_real);
                temp1_A = vaddq_f32(A1_imag, A1_real);
                temp2_A = vaddq_f32(A2_imag, A2_real);
                temp3_A = vaddq_f32(A3_imag, A3_real);

                temp0_B = vaddq_f32(B0_imag, B0_real);
                temp1_B = vaddq_f32(B1_imag, B1_real);
                temp2_B = vaddq_f32(B2_imag, B2_real);
                temp3_B = vaddq_f32(B3_imag, B3_real);

                // Cross multiplication, since we do not have enough registers, write on A{index}_real registers.
                A0_real = vmovq_n_f32(0);
                A0_real = vfmaq_laneq_f32(A0_real, temp0_B, temp0_A, 0);
                A0_real = vfmaq_laneq_f32(A0_real, temp1_B, temp0_A, 1);
                A0_real = vfmaq_laneq_f32(A0_real, temp2_B, temp0_A, 2);
                A0_real = vfmaq_laneq_f32(A0_real, temp3_B, temp0_A, 3);

                A1_real = vmovq_n_f32(0);
                A1_real = vfmaq_laneq_f32(A1_real, temp0_B, temp1_A, 0);
                A1_real = vfmaq_laneq_f32(A1_real, temp1_B, temp1_A, 1);
                A1_real = vfmaq_laneq_f32(A1_real, temp2_B, temp1_A, 2);
                A1_real = vfmaq_laneq_f32(A1_real, temp3_B, temp1_A, 3);

                A2_real = vmovq_n_f32(0);
                A2_real = vfmaq_laneq_f32(A2_real, temp0_B, temp2_A, 0);
                A2_real = vfmaq_laneq_f32(A2_real, temp1_B, temp2_A, 1);
                A2_real = vfmaq_laneq_f32(A2_real, temp2_B, temp2_A, 2);
                A2_real = vfmaq_laneq_f32(A2_real, temp3_B, temp2_A, 3);

                A3_real = vmovq_n_f32(0);
                A3_real = vfmaq_laneq_f32(A3_real, temp0_B, temp3_A, 0);
                A3_real = vfmaq_laneq_f32(A3_real, temp1_B, temp3_A, 1);
                A3_real = vfmaq_laneq_f32(A3_real, temp2_B, temp3_A, 2);
                A3_real = vfmaq_laneq_f32(A3_real, temp3_B, temp3_A, 3);
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///
                /// CALCULATE T2
                // Now use  temp{index}_B to calculate pure  imaginary multiplication of A and B
                // Pure Imaginary part as a matrix multiplication (T2 = Im(A) Im(B))
                temp0_B = vmovq_n_f32(0);
                temp0_B = vfmaq_laneq_f32(temp0_B, B0_imag, A0_imag, 0);
                temp0_B = vfmaq_laneq_f32(temp0_B, B1_imag, A0_imag, 1);
                temp0_B = vfmaq_laneq_f32(temp0_B, B2_imag, A0_imag, 2);
                temp0_B = vfmaq_laneq_f32(temp0_B, B3_imag, A0_imag, 3);

                temp1_B = vmovq_n_f32(0);
                temp1_B = vfmaq_laneq_f32(temp1_B, B0_imag, A1_imag, 0);
                temp1_B = vfmaq_laneq_f32(temp1_B, B1_imag, A1_imag, 1);
                temp1_B = vfmaq_laneq_f32(temp1_B, B2_imag, A1_imag, 2);
                temp1_B = vfmaq_laneq_f32(temp1_B, B3_imag, A1_imag, 3);

                temp2_B = vmovq_n_f32(0);
                temp2_B = vfmaq_laneq_f32(temp2_B, B0_imag, A2_imag, 0);
                temp2_B = vfmaq_laneq_f32(temp2_B, B1_imag, A2_imag, 1);
                temp2_B = vfmaq_laneq_f32(temp2_B, B2_imag, A2_imag, 2);
                temp2_B = vfmaq_laneq_f32(temp2_B, B3_imag, A2_imag, 3);

                temp3_B = vmovq_n_f32(0);
                temp3_B = vfmaq_laneq_f32(temp3_B, B0_imag, A3_imag, 0);
                temp3_B = vfmaq_laneq_f32(temp3_B, B1_imag, A3_imag, 1);
                temp3_B = vfmaq_laneq_f32(temp3_B, B2_imag, A3_imag, 2);
                temp3_B = vfmaq_laneq_f32(temp3_B, B3_imag, A3_imag, 3);
                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///
                // A{index}_real = Cross Value - T2.
                A0_real = vsubq_f32(A0_real, temp0_B);
                A1_real = vsubq_f32(A1_real, temp1_B);
                A2_real = vsubq_f32(A2_real, temp2_B);
                A3_real = vsubq_f32(A3_real, temp3_B);

                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                /// CALCULATE T1
                // We load Re(A) values again because we used it in the cross-calculation part.

                temp0_A = vld1q_f32(A_real + A_idx);
                temp1_A = vld1q_f32(A_real + A_idx + m1);
                temp2_A = vld1q_f32(A_real + A_idx + m1 * 2);
                temp3_A = vld1q_f32(A_real + A_idx + m1 * 3);

                // ATTENTION!!!! We use B{index}_imag for calculation on purpose!
                B0_imag = vmovq_n_f32(0);
                B1_imag = vmovq_n_f32(0);
                B2_imag = vmovq_n_f32(0);
                B3_imag = vmovq_n_f32(0);

                // Pure Real part as a matrix multiplication (T1 = Re(A) Re(B))
                B0_imag = vfmaq_laneq_f32(B0_imag, B0_real, temp0_A, 0);
                B0_imag = vfmaq_laneq_f32(B0_imag, B1_real, temp0_A, 1);
                B0_imag = vfmaq_laneq_f32(B0_imag, B2_real, temp0_A, 2);
                B0_imag = vfmaq_laneq_f32(B0_imag, B3_real, temp0_A, 3);

                B1_imag = vfmaq_laneq_f32(B1_imag, B0_real, temp1_A, 0);
                B1_imag = vfmaq_laneq_f32(B1_imag, B1_real, temp1_A, 1);
                B1_imag = vfmaq_laneq_f32(B1_imag, B2_real, temp1_A, 2);
                B1_imag = vfmaq_laneq_f32(B1_imag, B3_real, temp1_A, 3);

                B2_imag = vfmaq_laneq_f32(B2_imag, B0_real, temp2_A, 0);
                B2_imag = vfmaq_laneq_f32(B2_imag, B1_real, temp2_A, 1);
                B2_imag = vfmaq_laneq_f32(B2_imag, B2_real, temp2_A, 2);
                B2_imag = vfmaq_laneq_f32(B2_imag, B3_real, temp2_A, 3);

                B3_imag = vfmaq_laneq_f32(B3_imag, B0_real, temp3_A, 0);
                B3_imag = vfmaq_laneq_f32(B3_imag, B1_real, temp3_A, 1);
                B3_imag = vfmaq_laneq_f32(B3_imag, B2_real, temp3_A, 2);
                B3_imag = vfmaq_laneq_f32(B3_imag, B3_real, temp3_A, 3);

                // T1 = B{index}_imag = Re(A) * Re(B). It is on purpose!! There is no available temporary register!

                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                /// CALCULATE Im(AB)
                // (Cross value - T2) - T1
                A0_real = vsubq_f32(A0_real, B0_imag);
                A1_real = vsubq_f32(A1_real, B1_imag);
                A2_real = vsubq_f32(A2_real, B2_imag);
                A3_real = vsubq_f32(A3_real, B3_imag);

                // This addition is required because we are doing block-matrix multiplication!
                C0_imag = vaddq_f32(C0_imag, A0_real);
                C1_imag = vaddq_f32(C1_imag, A1_real);
                C2_imag = vaddq_f32(C2_imag, A2_real);
                C3_imag = vaddq_f32(C3_imag, A3_real);

                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                /// CALCULATE Re(AB)
                // T1 - T2
                B0_imag = vsubq_f32(B0_imag, temp0_B);
                B1_imag = vsubq_f32(B1_imag, temp1_B);
                B2_imag = vsubq_f32(B2_imag, temp2_B);
                B3_imag = vsubq_f32(B3_imag, temp3_B);

                C0_real = vaddq_f32(C0_real, B0_imag);
                C1_real = vaddq_f32(C1_real, B1_imag);
                C2_real = vaddq_f32(C2_real, B2_imag);
                C3_real = vaddq_f32(C3_real, B3_imag);
            }

            const size_t C_idx = k * i_idx + k_idx; // C{i,k}
            vst1q_f32(C_real + C_idx, C0_real);
            vst1q_f32(C_real + C_idx + k, C1_real);
            vst1q_f32(C_real + C_idx + k * 2, C2_real);
            vst1q_f32(C_real + C_idx + k * 3, C3_real);

            vst1q_f32(C_imag + C_idx, C0_imag);
            vst1q_f32(C_imag + C_idx + k, C1_imag);
            vst1q_f32(C_imag + C_idx + k * 2, C2_imag);
            vst1q_f32(C_imag + C_idx + k * 3, C3_imag);
        }
    }
    return 0;
}

int32_t Ne10Util::VectorxVectorElementwiseMultiplicationFloat(const float *A, const float *B,
                                                              const size_t &vectorSize, float *C)
{
    if (0 != vectorSize % 4)
    {
        return -1;
    }
    constexpr uint32 registerElementSize = 4;
    for (uint32 i = 0; i < vectorSize; i += registerElementSize)
    {
        float32x4_t vecA = vld1q_f32(A + i);
        float32x4_t vecB = vld1q_f32(B + i);
        float32x4_t temp = vmulq_f32(vecA, vecB);
        vst1q_f32(C + i, temp);
    }
    return 0;
}

int32_t Ne10Util::VectorxVectorElementwiseMultiplicationDouble(const double *A, const double *B,
                                                               const size_t &vectorSize, double *C)
{
    if (0 != vectorSize % 2)
    {
        return -1;
    }
    constexpr uint32 registerElementSize = 2;
    for (uint32 i = 0; i < vectorSize; i += registerElementSize)
    {
        float64x2_t vecA = vld1q_f64(A + i);
        float64x2_t vecB = vld1q_f64(B + i);
        float64x2_t temp = vmulq_f64(vecA, vecB);
        vst1q_f64(C + i, temp);
    }
    return 0;
}

int32_t Ne10Util::VectorxVectorElementwiseMultiplicationFloat1x16(const float *A, const float *B, float *C)
{
    float32x4_t vecA0 = vld1q_f32(A + 0);
    float32x4_t vecA1 = vld1q_f32(A + 4);
    float32x4_t vecA2 = vld1q_f32(A + 8);
    float32x4_t vecA3 = vld1q_f32(A + 12);

    float32x4_t vecB0 = vld1q_f32(B + 0);
    ;
    float32x4_t vecB1 = vld1q_f32(B + 4);
    ;
    float32x4_t vecB2 = vld1q_f32(B + 8);
    ;
    float32x4_t vecB3 = vld1q_f32(B + 12);
    ;

    float32x4_t vecC0 = vmulq_f32(vecA0, vecB0);
    float32x4_t vecC1 = vmulq_f32(vecA1, vecB1);
    float32x4_t vecC2 = vmulq_f32(vecA2, vecB2);
    float32x4_t vecC3 = vmulq_f32(vecA3, vecB3);

    vst1q_f32(C + 0, vecC0);
    vst1q_f32(C + 4, vecC1);
    vst1q_f32(C + 8, vecC2);
    vst1q_f32(C + 12, vecC3);

    return 0;
}

int32_t Ne10Util::VectorxVectorElementwiseMultiplicationDouble1x16(const double *A, const double *B, double *C)
{
    float64x2_t vecA0 = vld1q_f64(A + 0);
    float64x2_t vecA1 = vld1q_f64(A + 2);
    float64x2_t vecA2 = vld1q_f64(A + 4);
    float64x2_t vecA3 = vld1q_f64(A + 6);

    float64x2_t vecA4 = vld1q_f64(A + 8);
    float64x2_t vecA5 = vld1q_f64(A + 10);
    float64x2_t vecA6 = vld1q_f64(A + 12);
    float64x2_t vecA7 = vld1q_f64(A + 14);

    float64x2_t vecB0 = vld1q_f64(B + 0);
    float64x2_t vecB1 = vld1q_f64(B + 2);
    float64x2_t vecB2 = vld1q_f64(B + 4);
    float64x2_t vecB3 = vld1q_f64(B + 6);

    float64x2_t vecB4 = vld1q_f64(B + 8);
    float64x2_t vecB5 = vld1q_f64(B + 10);
    float64x2_t vecB6 = vld1q_f64(B + 12);
    float64x2_t vecB7 = vld1q_f64(B + 14);

    float64x2_t vecC0 = vmulq_f64(vecA0, vecB0);
    float64x2_t vecC1 = vmulq_f64(vecA1, vecB1);
    float64x2_t vecC2 = vmulq_f64(vecA2, vecB2);
    float64x2_t vecC3 = vmulq_f64(vecA3, vecB3);

    float64x2_t vecC4 = vmulq_f64(vecA4, vecB4);
    float64x2_t vecC5 = vmulq_f64(vecA5, vecB5);
    float64x2_t vecC6 = vmulq_f64(vecA6, vecB6);
    float64x2_t vecC7 = vmulq_f64(vecA7, vecB7);

    vst1q_f64(C + 0, vecC0);
    vst1q_f64(C + 2, vecC1);
    vst1q_f64(C + 4, vecC2);
    vst1q_f64(C + 6, vecC3);

    vst1q_f64(C + 8, vecC4);
    vst1q_f64(C + 10, vecC5);
    vst1q_f64(C + 12, vecC6);
    vst1q_f64(C + 14, vecC7);

    return 0;
}

int32_t Ne10Util::VectorMatrixMultiplicationFloat(const float *A, const float *B, const size_t &m1,
                                                  const size_t &m2, float *C)
{
    // m1 is the column number of A and the row number of B. A->(1xm1), B->(m1xm2)
    // m1 must be bigger than 4
    // m2 must be bigger of 4
    constexpr size_t registerElementSize = 4;
    if (0 != m2 % registerElementSize || 0 != m1 % registerElementSize)
    {
        return -1;
    }

    for (size_t columnIdx = 0; columnIdx < m2; columnIdx += registerElementSize)
    {
        float32x4_t vecC0 = vmovq_n_f32(0);
        for (size_t rowIndex = 0; rowIndex < m1; ++rowIndex)
        {
            float32x4_t vecA = vdupq_n_f32(*(A + rowIndex));
            float32x4_t vecB = vld1q_f32(B + rowIndex * m2 + columnIdx);

            vecC0 = vfmaq_f32(vecC0, vecB, vecA);
        }
        vst1q_f32(C + columnIdx, vecC0);
    }
}

int32_t Ne10Util::VectorMatrixMultiplicationDouble(const double *A, const double *B, const size_t &m1,
                                                   const size_t &m2, double *C)
{
    // m1 is the column number of A and the row number of B. A->(1xm1), B->(m1xm2)
    // m1 must be bigger than 2
    // m2 must be bigger of 2
    constexpr size_t registerElementSize = 2;
    if (0 != m2 % registerElementSize || 0 != m1 % registerElementSize)
    {
        return -1;
    }

    for (size_t columnIdx = 0; columnIdx < m2; columnIdx += registerElementSize)
    {
        float64x2_t vecC0 = vmovq_n_f64(0);
        for (size_t rowIndex = 0; rowIndex < m1; ++rowIndex)
        {
            float64x2_t vecA = vdupq_n_f64(*(A + rowIndex));
            float64x2_t vecB = vld1q_f64(B + rowIndex * m2 + columnIdx);

            vecC0 = vfmaq_f64(vecC0, vecB, vecA);
        }
        vst1q_f64(C + columnIdx, vecC0);
    }
    return 0;
}

int32_t Ne10Util::Vector1x16MatrixMultiplicationFloat16x16(const float *A, const float *B, float *C)
{

    // I wrote this function for high optimization (no any unnecessary multiplications and for loop) and little bit of fun.
    constexpr size_t M = 16;

    float32x4_t vecC0 = vmovq_n_f32(0);
    float32x4_t vecC1 = vmovq_n_f32(0);
    float32x4_t vecC2 = vmovq_n_f32(0);
    float32x4_t vecC3 = vmovq_n_f32(0);

    for (size_t idx = 0; idx < M; idx += 4)
    {
        float32x4_t vecA = vld1q_f32(A + idx);

        // First (4x16'th Block) Row. (1'st -> 16 elements)
        float32x4_t vecB0_0 = vld1q_f32(B + idx * M);
        float32x4_t vecB0_1 = vld1q_f32(B + idx * M + 4);
        float32x4_t vecB0_2 = vld1q_f32(B + idx * M + 8);
        float32x4_t vecB0_3 = vld1q_f32(B + idx * M + 12);

        // Second (4x16'th Block) Row. (2'nd -> 16 elements)
        float32x4_t vecB1_0 = vld1q_f32(B + idx * M + 16);
        float32x4_t vecB1_1 = vld1q_f32(B + idx * M + 20);
        float32x4_t vecB1_2 = vld1q_f32(B + idx * M + 24);
        float32x4_t vecB1_3 = vld1q_f32(B + idx * M + 28);

        // Third (4x16'th Block) Row. (3'rd -> 16 elements)
        float32x4_t vecB2_0 = vld1q_f32(B + idx * M + 32);
        float32x4_t vecB2_1 = vld1q_f32(B + idx * M + 36);
        float32x4_t vecB2_2 = vld1q_f32(B + idx * M + 40);
        float32x4_t vecB2_3 = vld1q_f32(B + idx * M + 44);

        // Fourth (4x16'th Block) Row. (4'th -> 16 elements)
        float32x4_t vecB3_0 = vld1q_f32(B + idx * M + 48);
        float32x4_t vecB3_1 = vld1q_f32(B + idx * M + 52);
        float32x4_t vecB3_2 = vld1q_f32(B + idx * M + 56);
        float32x4_t vecB3_3 = vld1q_f32(B + idx * M + 60);

        vecC0 = vfmaq_laneq_f32(vecC0, vecB0_0, vecA, 0);
        vecC1 = vfmaq_laneq_f32(vecC1, vecB0_1, vecA, 0);
        vecC2 = vfmaq_laneq_f32(vecC2, vecB0_2, vecA, 0);
        vecC3 = vfmaq_laneq_f32(vecC3, vecB0_3, vecA, 0);

        vecC0 = vfmaq_laneq_f32(vecC0, vecB1_0, vecA, 1);
        vecC1 = vfmaq_laneq_f32(vecC1, vecB1_1, vecA, 1);
        vecC2 = vfmaq_laneq_f32(vecC2, vecB1_2, vecA, 1);
        vecC3 = vfmaq_laneq_f32(vecC3, vecB1_3, vecA, 1);

        vecC0 = vfmaq_laneq_f32(vecC0, vecB2_0, vecA, 2);
        vecC1 = vfmaq_laneq_f32(vecC1, vecB2_1, vecA, 2);
        vecC2 = vfmaq_laneq_f32(vecC2, vecB2_2, vecA, 2);
        vecC3 = vfmaq_laneq_f32(vecC3, vecB2_3, vecA, 2);

        vecC0 = vfmaq_laneq_f32(vecC0, vecB3_0, vecA, 3);
        vecC1 = vfmaq_laneq_f32(vecC1, vecB3_1, vecA, 3);
        vecC2 = vfmaq_laneq_f32(vecC2, vecB3_2, vecA, 3);
        vecC3 = vfmaq_laneq_f32(vecC3, vecB3_3, vecA, 3);
    }

    vst1q_f32(C, vecC0);
    vst1q_f32(C + 4, vecC1);
    vst1q_f32(C + 8, vecC2);
    vst1q_f32(C + 12, vecC3);
    return 0;
}

int32_t Ne10Util::Vector1x16MatrixMultiplicationDouble16x16(const double *A, const double *B, double *C)
{
    constexpr size_t M = 16;

    float64x2_t vecC0 = vmovq_n_f64(0);
    float64x2_t vecC1 = vmovq_n_f64(0);
    float64x2_t vecC2 = vmovq_n_f64(0);
    float64x2_t vecC3 = vmovq_n_f64(0);

    float64x2_t vecC4 = vmovq_n_f64(0);
    float64x2_t vecC5 = vmovq_n_f64(0);
    float64x2_t vecC6 = vmovq_n_f64(0);
    float64x2_t vecC7 = vmovq_n_f64(0);

    for (size_t idx = 0; idx < M; idx += 2)
    {
        float64x2_t vecA = vld1q_f64(A + idx);

        float64x2_t vecB0_0 = vld1q_f64(B + idx * M);
        float64x2_t vecB0_1 = vld1q_f64(B + idx * M + 2);
        float64x2_t vecB0_2 = vld1q_f64(B + idx * M + 4);
        float64x2_t vecB0_3 = vld1q_f64(B + idx * M + 6);

        float64x2_t vecB0_4 = vld1q_f64(B + idx * M + 8);
        float64x2_t vecB0_5 = vld1q_f64(B + idx * M + 10);
        float64x2_t vecB0_6 = vld1q_f64(B + idx * M + 12);
        float64x2_t vecB0_7 = vld1q_f64(B + idx * M + 14);

        float64x2_t vecB1_0 = vld1q_f64(B + idx * M + 16);
        float64x2_t vecB1_1 = vld1q_f64(B + idx * M + 18);
        float64x2_t vecB1_2 = vld1q_f64(B + idx * M + 20);
        float64x2_t vecB1_3 = vld1q_f64(B + idx * M + 22);

        float64x2_t vecB1_4 = vld1q_f64(B + idx * M + 24);
        float64x2_t vecB1_5 = vld1q_f64(B + idx * M + 26);
        float64x2_t vecB1_6 = vld1q_f64(B + idx * M + 28);
        float64x2_t vecB1_7 = vld1q_f64(B + idx * M + 30);

        vecC0 = vfmaq_laneq_f64(vecC0, vecB0_0, vecA, 0);
        vecC1 = vfmaq_laneq_f64(vecC1, vecB0_1, vecA, 0);
        vecC2 = vfmaq_laneq_f64(vecC2, vecB0_2, vecA, 0);
        vecC3 = vfmaq_laneq_f64(vecC3, vecB0_3, vecA, 0);
        vecC4 = vfmaq_laneq_f64(vecC4, vecB0_4, vecA, 0);
        vecC5 = vfmaq_laneq_f64(vecC5, vecB0_5, vecA, 0);
        vecC6 = vfmaq_laneq_f64(vecC6, vecB0_6, vecA, 0);
        vecC7 = vfmaq_laneq_f64(vecC7, vecB0_7, vecA, 0);

        vecC0 = vfmaq_laneq_f64(vecC0, vecB1_0, vecA, 1);
        vecC1 = vfmaq_laneq_f64(vecC1, vecB1_1, vecA, 1);
        vecC2 = vfmaq_laneq_f64(vecC2, vecB1_2, vecA, 1);
        vecC3 = vfmaq_laneq_f64(vecC3, vecB1_3, vecA, 1);
        vecC4 = vfmaq_laneq_f64(vecC4, vecB1_4, vecA, 1);
        vecC5 = vfmaq_laneq_f64(vecC5, vecB1_5, vecA, 1);
        vecC6 = vfmaq_laneq_f64(vecC6, vecB1_6, vecA, 1);
        vecC7 = vfmaq_laneq_f64(vecC7, vecB1_7, vecA, 1);
    }
    vst1q_f64(C, vecC0);
    vst1q_f64(C + 2, vecC1);
    vst1q_f64(C + 4, vecC2);
    vst1q_f64(C + 6, vecC3);
    vst1q_f64(C + 8, vecC4);
    vst1q_f64(C + 10, vecC5);
    vst1q_f64(C + 12, vecC6);
    vst1q_f64(C + 14, vecC7);

    return 0;
}

int32_t Ne10Util::Vector1x16MatrixMultiplicationComplexFloat16x16(const float *A_real, const float *A_imag,
                                                                  const float *B_real, const float *B_imag, float *C_real, float *C_imag)
{
    constexpr size_t vectorSize = 16;
    float32x4_t vecC0_real = vmovq_n_f32(0);
    float32x4_t vecC1_real = vmovq_n_f32(0);
    float32x4_t vecC2_real = vmovq_n_f32(0);
    float32x4_t vecC3_real = vmovq_n_f32(0);

    float32x4_t vecC0_imag = vmovq_n_f32(0);
    float32x4_t vecC1_imag = vmovq_n_f32(0);
    float32x4_t vecC2_imag = vmovq_n_f32(0);
    float32x4_t vecC3_imag = vmovq_n_f32(0);
    for (size_t idx = 0; idx < vectorSize; idx += 2)
    {
        float32x4_t vecA_real = vld1q_f32(A_real + idx);
        float32x4_t vecA_imag = vld1q_f32(A_imag + idx);
        float32x4_t tempA = vaddq_f32(vecA_real, vecA_imag);

        float32x4_t vecB0_0_real = vld1q_f32(B_real + idx * vectorSize);
        float32x4_t vecB0_1_real = vld1q_f32(B_real + idx * vectorSize + 4);
        float32x4_t vecB0_2_real = vld1q_f32(B_real + idx * vectorSize + 8);
        float32x4_t vecB0_3_real = vld1q_f32(B_real + idx * vectorSize + 12);

        float32x4_t vecB1_0_real = vld1q_f32(B_real + idx * vectorSize + 16);
        float32x4_t vecB1_1_real = vld1q_f32(B_real + idx * vectorSize + 20);
        float32x4_t vecB1_2_real = vld1q_f32(B_real + idx * vectorSize + 24);
        float32x4_t vecB1_3_real = vld1q_f32(B_real + idx * vectorSize + 28);

        float32x4_t vecB0_0_imag = vld1q_f32(B_imag + idx * vectorSize);
        float32x4_t vecB0_1_imag = vld1q_f32(B_imag + idx * vectorSize + 4);
        float32x4_t vecB0_2_imag = vld1q_f32(B_imag + idx * vectorSize + 8);
        float32x4_t vecB0_3_imag = vld1q_f32(B_imag + idx * vectorSize + 12);

        float32x4_t vecB1_0_imag = vld1q_f32(B_imag + idx * vectorSize + 16);
        float32x4_t vecB1_1_imag = vld1q_f32(B_imag + idx * vectorSize + 20);
        float32x4_t vecB1_2_imag = vld1q_f32(B_imag + idx * vectorSize + 24);
        float32x4_t vecB1_3_imag = vld1q_f32(B_imag + idx * vectorSize + 28);

        // Former Row!
        // Imag(B) + Re(B)
        float32x4_t tempB0 = vaddq_f32(vecB0_0_real, vecB0_0_imag);
        float32x4_t tempB1 = vaddq_f32(vecB0_1_real, vecB0_1_imag);
        float32x4_t tempB2 = vaddq_f32(vecB0_2_real, vecB0_2_imag);
        float32x4_t tempB3 = vaddq_f32(vecB0_3_real, vecB0_3_imag);

        // Former Element in the register duplicated
        float32x4_t tempAScalar = vdupq_n_f32(vgetq_lane_f32(tempA, 0));

        // Cross Term
        tempB0 = vmulq_f32(tempAScalar, tempB0);
        tempB1 = vmulq_f32(tempAScalar, tempB1);
        tempB2 = vmulq_f32(tempAScalar, tempB2);
        tempB3 = vmulq_f32(tempAScalar, tempB3);

        // Cross Term (above, cumulative sum with vecC...)!
        vecC0_imag = vaddq_f32(tempB0, vecC0_imag);
        vecC1_imag = vaddq_f32(tempB1, vecC1_imag);
        vecC2_imag = vaddq_f32(tempB2, vecC2_imag);
        vecC3_imag = vaddq_f32(tempB3, vecC3_imag);

        // Calculate t1
        tempA = vdupq_n_f32(vgetq_lane_f32(vecA_real, 0));

        // t1
        vecB0_0_real = vmulq_f32(tempA, vecB0_0_real);
        vecB0_1_real = vmulq_f32(tempA, vecB0_1_real);
        vecB0_2_real = vmulq_f32(tempA, vecB0_2_real);
        vecB0_3_real = vmulq_f32(tempA, vecB0_3_real);

        // Calculate t2
        tempA = vdupq_n_f32(vgetq_lane_f32(vecA_imag, 0));

        // t2

        vecB0_0_imag = vmulq_f32(tempA, vecB0_0_imag);
        vecB0_1_imag = vmulq_f32(tempA, vecB0_1_imag);
        vecB0_2_imag = vmulq_f32(tempA, vecB0_2_imag);
        vecB0_3_imag = vmulq_f32(tempA, vecB0_3_imag);

        // Cross Term =  Cross Term - t1
        vecC0_imag = vsubq_f32(vecC0_imag, vecB0_0_real);
        vecC1_imag = vsubq_f32(vecC1_imag, vecB0_1_real);
        vecC2_imag = vsubq_f32(vecC2_imag, vecB0_2_real);
        vecC3_imag = vsubq_f32(vecC3_imag, vecB0_3_real);

        // Finally Imaginary Part = Cross Term - t2
        vecC0_imag = vsubq_f32(vecC0_imag, vecB0_0_imag);
        vecC1_imag = vsubq_f32(vecC1_imag, vecB0_1_imag);
        vecC2_imag = vsubq_f32(vecC2_imag, vecB0_2_imag);
        vecC3_imag = vsubq_f32(vecC3_imag, vecB0_3_imag);

        // Cumulative Real Part Calculation:
        vecC0_real = vaddq_f32(vecB0_0_real, vecC0_real);
        vecC1_real = vaddq_f32(vecB0_1_real, vecC1_real);
        vecC2_real = vaddq_f32(vecB0_2_real, vecC2_real);
        vecC3_real = vaddq_f32(vecB0_3_real, vecC3_real);

        // Real Part =  t1 - t2
        vecC0_real = vsubq_f32(vecC0_real, vecB0_0_imag);
        vecC1_real = vsubq_f32(vecC1_real, vecB0_1_imag);
        vecC2_real = vsubq_f32(vecC2_real, vecB0_2_imag);
        vecC3_real = vsubq_f32(vecC3_real, vecB0_3_imag);

        /// TRUE!!!
        /////////////////////////////////////
        /// Now the Second Row that taken to register
        ///
        tempB0 = vaddq_f32(vecB1_0_real, vecB1_0_imag);
        tempB1 = vaddq_f32(vecB1_1_real, vecB1_1_imag);
        tempB2 = vaddq_f32(vecB1_2_real, vecB1_2_imag);
        tempB3 = vaddq_f32(vecB1_3_real, vecB1_3_imag);

        // Former Element in the register duplicated
        tempA = vaddq_f32(vecA_real, vecA_imag);
        tempAScalar = vdupq_n_f32(vgetq_lane_f32(tempA, 1));

        // Cross Term
        tempB0 = vmulq_f32(tempAScalar, tempB0);
        tempB1 = vmulq_f32(tempAScalar, tempB1);
        tempB2 = vmulq_f32(tempAScalar, tempB2);
        tempB3 = vmulq_f32(tempAScalar, tempB3);

        // Cross Term (above, cumulative sum with vecC...)!
        vecC0_imag = vaddq_f32(tempB0, vecC0_imag);
        vecC1_imag = vaddq_f32(tempB1, vecC1_imag);
        vecC2_imag = vaddq_f32(tempB2, vecC2_imag);
        vecC3_imag = vaddq_f32(tempB3, vecC3_imag);

        // Calculate t1
        tempA = vdupq_n_f32(vgetq_lane_f32(vecA_real, 1));

        // t1
        vecB1_0_real = vmulq_f32(tempA, vecB1_0_real);
        vecB1_1_real = vmulq_f32(tempA, vecB1_1_real);
        vecB1_2_real = vmulq_f32(tempA, vecB1_2_real);
        vecB1_3_real = vmulq_f32(tempA, vecB1_3_real);

        // Calculate t2
        tempA = vdupq_n_f32(vgetq_lane_f32(vecA_imag, 1));

        // t2

        vecB1_0_imag = vmulq_f32(tempA, vecB1_0_imag);
        vecB1_1_imag = vmulq_f32(tempA, vecB1_1_imag);
        vecB1_2_imag = vmulq_f32(tempA, vecB1_2_imag);
        vecB1_3_imag = vmulq_f32(tempA, vecB1_3_imag);

        // Cross Term =  Cross Term - t1
        vecC0_imag = vsubq_f32(vecC0_imag, vecB1_0_real);
        vecC1_imag = vsubq_f32(vecC1_imag, vecB1_1_real);
        vecC2_imag = vsubq_f32(vecC2_imag, vecB1_2_real);
        vecC3_imag = vsubq_f32(vecC3_imag, vecB1_3_real);

        // Finally Imaginary Part = Cross Term - t2
        vecC0_imag = vsubq_f32(vecC0_imag, vecB1_0_imag);
        vecC1_imag = vsubq_f32(vecC1_imag, vecB1_1_imag);
        vecC2_imag = vsubq_f32(vecC2_imag, vecB1_2_imag);
        vecC3_imag = vsubq_f32(vecC3_imag, vecB1_3_imag);

        // Cumulative Real Part Calculation:
        vecC0_real = vaddq_f32(vecB1_0_real, vecC0_real);
        vecC1_real = vaddq_f32(vecB1_1_real, vecC1_real);
        vecC2_real = vaddq_f32(vecB1_2_real, vecC2_real);
        vecC3_real = vaddq_f32(vecB1_3_real, vecC3_real);

        // Real Part =  t1 - t2
        vecC0_real = vsubq_f32(vecC0_real, vecB1_0_imag);
        vecC1_real = vsubq_f32(vecC1_real, vecB1_1_imag);
        vecC2_real = vsubq_f32(vecC2_real, vecB1_2_imag);
        vecC3_real = vsubq_f32(vecC3_real, vecB1_3_imag);
    }
    vst1q_f32(C_real + 0, vecC0_real);
    vst1q_f32(C_real + 4, vecC1_real);
    vst1q_f32(C_real + 8, vecC2_real);
    vst1q_f32(C_real + 12, vecC3_real);

    vst1q_f32(C_imag + 0, vecC0_imag);
    vst1q_f32(C_imag + 4, vecC1_imag);
    vst1q_f32(C_imag + 8, vecC2_imag);
    vst1q_f32(C_imag + 12, vecC3_imag);

    return 0;
}

int32_t Ne10Util::Vector1x16MatrixMultiplicationComplexDouble16x16(const double *A_real, const double *A_imag,
                                                                   const double *B_real, const double *B_imag, double *C_real, double *C_imag)
{

    constexpr size_t vectorSize = 16;
    constexpr size_t processedIdx2Elements = 8;

    for (size_t idx2 = 0; idx2 < 2; idx2++)
    {
        float64x2_t vecC0_real = vmovq_n_f64(0);
        float64x2_t vecC1_real = vmovq_n_f64(0);
        float64x2_t vecC2_real = vmovq_n_f64(0);
        float64x2_t vecC3_real = vmovq_n_f64(0);

        float64x2_t vecC0_imag = vmovq_n_f64(0);
        float64x2_t vecC1_imag = vmovq_n_f64(0);
        float64x2_t vecC2_imag = vmovq_n_f64(0);
        float64x2_t vecC3_imag = vmovq_n_f64(0);

        for (size_t idx = 0; idx < vectorSize; idx += 2)
        {
            float64x2_t vecA_real = vld1q_f64(A_real + idx);
            float64x2_t vecA_imag = vld1q_f64(A_imag + idx);
            float64x2_t tempA = vaddq_f64(vecA_real, vecA_imag);

            float64x2_t vecB0_0_real = vld1q_f64(B_real + idx2 * processedIdx2Elements + idx * vectorSize);
            float64x2_t vecB0_1_real = vld1q_f64(B_real + idx2 * processedIdx2Elements + idx * vectorSize + 2);
            float64x2_t vecB0_2_real = vld1q_f64(B_real + idx2 * processedIdx2Elements + idx * vectorSize + 4);
            float64x2_t vecB0_3_real = vld1q_f64(B_real + idx2 * processedIdx2Elements + idx * vectorSize + 6);

            float64x2_t vecB1_0_real = vld1q_f64(B_real + idx2 * processedIdx2Elements + idx * vectorSize + 16);
            float64x2_t vecB1_1_real = vld1q_f64(B_real + idx2 * processedIdx2Elements + idx * vectorSize + 18);
            float64x2_t vecB1_2_real = vld1q_f64(B_real + idx2 * processedIdx2Elements + idx * vectorSize + 20);
            float64x2_t vecB1_3_real = vld1q_f64(B_real + idx2 * processedIdx2Elements + idx * vectorSize + 22);

            float64x2_t vecB0_0_imag = vld1q_f64(B_imag + idx2 * processedIdx2Elements + idx * vectorSize);
            float64x2_t vecB0_1_imag = vld1q_f64(B_imag + idx2 * processedIdx2Elements + idx * vectorSize + 2);
            float64x2_t vecB0_2_imag = vld1q_f64(B_imag + idx2 * processedIdx2Elements + idx * vectorSize + 4);
            float64x2_t vecB0_3_imag = vld1q_f64(B_imag + idx2 * processedIdx2Elements + idx * vectorSize + 6);

            float64x2_t vecB1_0_imag = vld1q_f64(B_imag + idx2 * processedIdx2Elements + idx * vectorSize + 16);
            float64x2_t vecB1_1_imag = vld1q_f64(B_imag + idx2 * processedIdx2Elements + idx * vectorSize + 18);
            float64x2_t vecB1_2_imag = vld1q_f64(B_imag + idx2 * processedIdx2Elements + idx * vectorSize + 20);
            float64x2_t vecB1_3_imag = vld1q_f64(B_imag + idx2 * processedIdx2Elements + idx * vectorSize + 22);

            // Former Row!
            // Imag(B) + Re(B)
            float64x2_t tempB0 = vaddq_f64(vecB0_0_real, vecB0_0_imag);
            float64x2_t tempB1 = vaddq_f64(vecB0_1_real, vecB0_1_imag);
            float64x2_t tempB2 = vaddq_f64(vecB0_2_real, vecB0_2_imag);
            float64x2_t tempB3 = vaddq_f64(vecB0_3_real, vecB0_3_imag);

            // Former Element in the register duplicated
            float64x2_t tempAScalar = vdupq_n_f64(vgetq_lane_f64(tempA, 0));

            // Cross Term
            tempB0 = vmulq_f64(tempAScalar, tempB0);
            tempB1 = vmulq_f64(tempAScalar, tempB1);
            tempB2 = vmulq_f64(tempAScalar, tempB2);
            tempB3 = vmulq_f64(tempAScalar, tempB3);

            // Cross Term (above, cumulative sum with vecC...)!
            vecC0_imag = vaddq_f64(tempB0, vecC0_imag);
            vecC1_imag = vaddq_f64(tempB1, vecC1_imag);
            vecC2_imag = vaddq_f64(tempB2, vecC2_imag);
            vecC3_imag = vaddq_f64(tempB3, vecC3_imag);

            // Calculate t1
            tempA = vdupq_n_f64(vgetq_lane_f64(vecA_real, 0));

            // t1
            vecB0_0_real = vmulq_f64(tempA, vecB0_0_real);
            vecB0_1_real = vmulq_f64(tempA, vecB0_1_real);
            vecB0_2_real = vmulq_f64(tempA, vecB0_2_real);
            vecB0_3_real = vmulq_f64(tempA, vecB0_3_real);

            // Calculate t2
            tempA = vdupq_n_f64(vgetq_lane_f64(vecA_imag, 0));

            // t2

            vecB0_0_imag = vmulq_f64(tempA, vecB0_0_imag);
            vecB0_1_imag = vmulq_f64(tempA, vecB0_1_imag);
            vecB0_2_imag = vmulq_f64(tempA, vecB0_2_imag);
            vecB0_3_imag = vmulq_f64(tempA, vecB0_3_imag);

            // Cross Term =  Cross Term - t1
            vecC0_imag = vsubq_f64(vecC0_imag, vecB0_0_real);
            vecC1_imag = vsubq_f64(vecC1_imag, vecB0_1_real);
            vecC2_imag = vsubq_f64(vecC2_imag, vecB0_2_real);
            vecC3_imag = vsubq_f64(vecC3_imag, vecB0_3_real);

            // Finally Imaginary Part = Cross Term - t2
            vecC0_imag = vsubq_f64(vecC0_imag, vecB0_0_imag);
            vecC1_imag = vsubq_f64(vecC1_imag, vecB0_1_imag);
            vecC2_imag = vsubq_f64(vecC2_imag, vecB0_2_imag);
            vecC3_imag = vsubq_f64(vecC3_imag, vecB0_3_imag);

            // Cumulative Real Part Calculation:
            vecC0_real = vaddq_f64(vecB0_0_real, vecC0_real);
            vecC1_real = vaddq_f64(vecB0_1_real, vecC1_real);
            vecC2_real = vaddq_f64(vecB0_2_real, vecC2_real);
            vecC3_real = vaddq_f64(vecB0_3_real, vecC3_real);

            // Real Part =  t1 - t2
            vecC0_real = vsubq_f64(vecC0_real, vecB0_0_imag);
            vecC1_real = vsubq_f64(vecC1_real, vecB0_1_imag);
            vecC2_real = vsubq_f64(vecC2_real, vecB0_2_imag);
            vecC3_real = vsubq_f64(vecC3_real, vecB0_3_imag);

            /// TRUE!!!
            /////////////////////////////////////
            /// Now the Second Row that taken to register
            ///
            tempB0 = vaddq_f64(vecB1_0_real, vecB1_0_imag);
            tempB1 = vaddq_f64(vecB1_1_real, vecB1_1_imag);
            tempB2 = vaddq_f64(vecB1_2_real, vecB1_2_imag);
            tempB3 = vaddq_f64(vecB1_3_real, vecB1_3_imag);

            // Former Element in the register duplicated
            tempA = vaddq_f64(vecA_real, vecA_imag);
            tempAScalar = vdupq_n_f64(vgetq_lane_f64(tempA, 1));

            // Cross Term
            tempB0 = vmulq_f64(tempAScalar, tempB0);
            tempB1 = vmulq_f64(tempAScalar, tempB1);
            tempB2 = vmulq_f64(tempAScalar, tempB2);
            tempB3 = vmulq_f64(tempAScalar, tempB3);

            // Cross Term (above, cumulative sum with vecC...)!
            vecC0_imag = vaddq_f64(tempB0, vecC0_imag);
            vecC1_imag = vaddq_f64(tempB1, vecC1_imag);
            vecC2_imag = vaddq_f64(tempB2, vecC2_imag);
            vecC3_imag = vaddq_f64(tempB3, vecC3_imag);

            // Calculate t1
            tempA = vdupq_n_f64(vgetq_lane_f64(vecA_real, 1));

            // t1
            vecB1_0_real = vmulq_f64(tempA, vecB1_0_real);
            vecB1_1_real = vmulq_f64(tempA, vecB1_1_real);
            vecB1_2_real = vmulq_f64(tempA, vecB1_2_real);
            vecB1_3_real = vmulq_f64(tempA, vecB1_3_real);

            // Calculate t2
            tempA = vdupq_n_f64(vgetq_lane_f64(vecA_imag, 1));

            // t2

            vecB1_0_imag = vmulq_f64(tempA, vecB1_0_imag);
            vecB1_1_imag = vmulq_f64(tempA, vecB1_1_imag);
            vecB1_2_imag = vmulq_f64(tempA, vecB1_2_imag);
            vecB1_3_imag = vmulq_f64(tempA, vecB1_3_imag);

            // Cross Term =  Cross Term - t1
            vecC0_imag = vsubq_f64(vecC0_imag, vecB1_0_real);
            vecC1_imag = vsubq_f64(vecC1_imag, vecB1_1_real);
            vecC2_imag = vsubq_f64(vecC2_imag, vecB1_2_real);
            vecC3_imag = vsubq_f64(vecC3_imag, vecB1_3_real);

            // Finally Imaginary Part = Cross Term - t2
            vecC0_imag = vsubq_f64(vecC0_imag, vecB1_0_imag);
            vecC1_imag = vsubq_f64(vecC1_imag, vecB1_1_imag);
            vecC2_imag = vsubq_f64(vecC2_imag, vecB1_2_imag);
            vecC3_imag = vsubq_f64(vecC3_imag, vecB1_3_imag);

            // Cumulative Real Part Calculation:
            vecC0_real = vaddq_f64(vecB1_0_real, vecC0_real);
            vecC1_real = vaddq_f64(vecB1_1_real, vecC1_real);
            vecC2_real = vaddq_f64(vecB1_2_real, vecC2_real);
            vecC3_real = vaddq_f64(vecB1_3_real, vecC3_real);

            // Real Part =  t1 - t2
            vecC0_real = vsubq_f64(vecC0_real, vecB1_0_imag);
            vecC1_real = vsubq_f64(vecC1_real, vecB1_1_imag);
            vecC2_real = vsubq_f64(vecC2_real, vecB1_2_imag);
            vecC3_real = vsubq_f64(vecC3_real, vecB1_3_imag);
        }

        vst1q_f64(C_real + idx2 * processedIdx2Elements + 0, vecC0_real);
        vst1q_f64(C_real + idx2 * processedIdx2Elements + 2, vecC1_real);
        vst1q_f64(C_real + idx2 * processedIdx2Elements + 4, vecC2_real);
        vst1q_f64(C_real + idx2 * processedIdx2Elements + 6, vecC3_real);

        vst1q_f64(C_imag + idx2 * processedIdx2Elements + 0, vecC0_imag);
        vst1q_f64(C_imag + idx2 * processedIdx2Elements + 2, vecC1_imag);
        vst1q_f64(C_imag + idx2 * processedIdx2Elements + 4, vecC2_imag);
        vst1q_f64(C_imag + idx2 * processedIdx2Elements + 6, vecC3_imag);
    }
    return 0;
}

int32_t Ne10Util::NormalizeComplexVectorFloat1x16(float *A_real, float *A_imag)
{
    float32x4_t sum_vec = vmovq_n_f32(0);

    float32x4_t vecA0_real = vld1q_f32(A_real);
    float32x4_t vecA1_real = vld1q_f32(A_real + 4);
    float32x4_t vecA2_real = vld1q_f32(A_real + 8);
    float32x4_t vecA3_real = vld1q_f32(A_real + 12);

    float32x4_t vecA0_imag = vld1q_f32(A_imag);
    float32x4_t vecA1_imag = vld1q_f32(A_imag + 4);
    float32x4_t vecA2_imag = vld1q_f32(A_imag + 8);
    float32x4_t vecA3_imag = vld1q_f32(A_imag + 12);

    sum_vec = vfmaq_f32(sum_vec, vecA0_real, vecA0_real);
    sum_vec = vfmaq_f32(sum_vec, vecA1_real, vecA1_real);
    sum_vec = vfmaq_f32(sum_vec, vecA2_real, vecA2_real);
    sum_vec = vfmaq_f32(sum_vec, vecA3_real, vecA3_real);

    sum_vec = vfmaq_f32(sum_vec, vecA0_imag, vecA0_imag);
    sum_vec = vfmaq_f32(sum_vec, vecA1_imag, vecA1_imag);
    sum_vec = vfmaq_f32(sum_vec, vecA2_imag, vecA2_imag);
    sum_vec = vfmaq_f32(sum_vec, vecA3_imag, vecA3_imag);

    float32x4_t norm_vec = vdupq_n_f32(sqrtf(vaddvq_f32(sum_vec)));

    vecA0_real = vdivq_f32(vecA0_real, norm_vec);
    vecA1_real = vdivq_f32(vecA1_real, norm_vec);
    vecA2_real = vdivq_f32(vecA2_real, norm_vec);
    vecA3_real = vdivq_f32(vecA3_real, norm_vec);

    vecA0_imag = vdivq_f32(vecA0_imag, norm_vec);
    vecA1_imag = vdivq_f32(vecA1_imag, norm_vec);
    vecA2_imag = vdivq_f32(vecA2_imag, norm_vec);
    vecA3_imag = vdivq_f32(vecA3_imag, norm_vec);

    vst1q_f32(A_real, vecA0_real);
    vst1q_f32(A_real + 4, vecA1_real);
    vst1q_f32(A_real + 8, vecA2_real);
    vst1q_f32(A_real + 12, vecA3_real);

    vst1q_f32(A_imag, vecA0_imag);
    vst1q_f32(A_imag + 4, vecA1_imag);
    vst1q_f32(A_imag + 8, vecA2_imag);
    vst1q_f32(A_imag + 12, vecA3_imag);

    return 0;
}

int32_t Ne10Util::NormalizeComplexVectorDouble1x16(double *A_real, double *A_imag)
{
    float64x2_t sum_vec = vmovq_n_f64(0);

    float64x2_t vecA0_real = vld1q_f64(A_real);
    float64x2_t vecA1_real = vld1q_f64(A_real + 2);
    float64x2_t vecA2_real = vld1q_f64(A_real + 4);
    float64x2_t vecA3_real = vld1q_f64(A_real + 6);

    float64x2_t vecA4_real = vld1q_f64(A_real + 8);
    float64x2_t vecA5_real = vld1q_f64(A_real + 10);
    float64x2_t vecA6_real = vld1q_f64(A_real + 12);
    float64x2_t vecA7_real = vld1q_f64(A_real + 14);

    float64x2_t vecA0_imag = vld1q_f64(A_imag);
    float64x2_t vecA1_imag = vld1q_f64(A_imag + 2);
    float64x2_t vecA2_imag = vld1q_f64(A_imag + 4);
    float64x2_t vecA3_imag = vld1q_f64(A_imag + 6);

    float64x2_t vecA4_imag = vld1q_f64(A_imag + 8);
    float64x2_t vecA5_imag = vld1q_f64(A_imag + 10);
    float64x2_t vecA6_imag = vld1q_f64(A_imag + 12);
    float64x2_t vecA7_imag = vld1q_f64(A_imag + 14);

    sum_vec = vfmaq_f64(sum_vec, vecA0_real, vecA0_real);
    sum_vec = vfmaq_f64(sum_vec, vecA1_real, vecA1_real);
    sum_vec = vfmaq_f64(sum_vec, vecA2_real, vecA2_real);
    sum_vec = vfmaq_f64(sum_vec, vecA3_real, vecA3_real);

    sum_vec = vfmaq_f64(sum_vec, vecA4_real, vecA4_real);
    sum_vec = vfmaq_f64(sum_vec, vecA5_real, vecA5_real);
    sum_vec = vfmaq_f64(sum_vec, vecA6_real, vecA6_real);
    sum_vec = vfmaq_f64(sum_vec, vecA7_real, vecA7_real);

    sum_vec = vfmaq_f64(sum_vec, vecA0_imag, vecA0_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA1_imag, vecA1_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA2_imag, vecA2_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA3_imag, vecA3_imag);

    sum_vec = vfmaq_f64(sum_vec, vecA4_imag, vecA4_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA5_imag, vecA5_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA6_imag, vecA6_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA7_imag, vecA7_imag);

    float64x2_t norm_vec = vdupq_n_f64(sqrt(vaddvq_f64(sum_vec)));

    vecA0_real = vdivq_f64(vecA0_real, norm_vec);
    vecA1_real = vdivq_f64(vecA1_real, norm_vec);
    vecA2_real = vdivq_f64(vecA2_real, norm_vec);
    vecA3_real = vdivq_f64(vecA3_real, norm_vec);

    vecA4_real = vdivq_f64(vecA4_real, norm_vec);
    vecA5_real = vdivq_f64(vecA5_real, norm_vec);
    vecA6_real = vdivq_f64(vecA6_real, norm_vec);
    vecA7_real = vdivq_f64(vecA7_real, norm_vec);

    vecA0_imag = vdivq_f64(vecA0_imag, norm_vec);
    vecA1_imag = vdivq_f64(vecA1_imag, norm_vec);
    vecA2_imag = vdivq_f64(vecA2_imag, norm_vec);
    vecA3_imag = vdivq_f64(vecA3_imag, norm_vec);

    vecA4_imag = vdivq_f64(vecA4_imag, norm_vec);
    vecA5_imag = vdivq_f64(vecA5_imag, norm_vec);
    vecA6_imag = vdivq_f64(vecA6_imag, norm_vec);
    vecA7_imag = vdivq_f64(vecA7_imag, norm_vec);

    vst1q_f64(A_real, vecA0_real);
    vst1q_f64(A_real + 2, vecA1_real);
    vst1q_f64(A_real + 4, vecA2_real);
    vst1q_f64(A_real + 6, vecA3_real);

    vst1q_f64(A_real + 8, vecA4_real);
    vst1q_f64(A_real + 10, vecA5_real);
    vst1q_f64(A_real + 12, vecA6_real);
    vst1q_f64(A_real + 14, vecA7_real);

    vst1q_f64(A_imag, vecA0_imag);
    vst1q_f64(A_imag + 2, vecA1_imag);
    vst1q_f64(A_imag + 4, vecA2_imag);
    vst1q_f64(A_imag + 6, vecA3_imag);

    vst1q_f64(A_imag + 8, vecA4_imag);
    vst1q_f64(A_imag + 10, vecA5_imag);
    vst1q_f64(A_imag + 12, vecA6_imag);
    vst1q_f64(A_imag + 14, vecA7_imag);

    return 0;
}

int32_t Ne10Util::NormComplexVectorDouble1x16(const double *A_real, const double *A_imag, double *C)
{
    float64x2_t sum_vec = vmovq_n_f64(0);

    float64x2_t vecA0_real = vld1q_f64(A_real);
    float64x2_t vecA1_real = vld1q_f64(A_real + 2);
    float64x2_t vecA2_real = vld1q_f64(A_real + 4);
    float64x2_t vecA3_real = vld1q_f64(A_real + 6);

    float64x2_t vecA4_real = vld1q_f64(A_real + 8);
    float64x2_t vecA5_real = vld1q_f64(A_real + 10);
    float64x2_t vecA6_real = vld1q_f64(A_real + 12);
    float64x2_t vecA7_real = vld1q_f64(A_real + 14);

    float64x2_t vecA0_imag = vld1q_f64(A_imag);
    float64x2_t vecA1_imag = vld1q_f64(A_imag + 2);
    float64x2_t vecA2_imag = vld1q_f64(A_imag + 4);
    float64x2_t vecA3_imag = vld1q_f64(A_imag + 6);

    float64x2_t vecA4_imag = vld1q_f64(A_imag + 8);
    float64x2_t vecA5_imag = vld1q_f64(A_imag + 10);
    float64x2_t vecA6_imag = vld1q_f64(A_imag + 12);
    float64x2_t vecA7_imag = vld1q_f64(A_imag + 14);

    sum_vec = vfmaq_f64(sum_vec, vecA0_real, vecA0_real);
    sum_vec = vfmaq_f64(sum_vec, vecA1_real, vecA1_real);
    sum_vec = vfmaq_f64(sum_vec, vecA2_real, vecA2_real);
    sum_vec = vfmaq_f64(sum_vec, vecA3_real, vecA3_real);

    sum_vec = vfmaq_f64(sum_vec, vecA4_real, vecA4_real);
    sum_vec = vfmaq_f64(sum_vec, vecA5_real, vecA5_real);
    sum_vec = vfmaq_f64(sum_vec, vecA6_real, vecA6_real);
    sum_vec = vfmaq_f64(sum_vec, vecA7_real, vecA7_real);

    sum_vec = vfmaq_f64(sum_vec, vecA0_imag, vecA0_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA1_imag, vecA1_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA2_imag, vecA2_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA3_imag, vecA3_imag);

    sum_vec = vfmaq_f64(sum_vec, vecA4_imag, vecA4_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA5_imag, vecA5_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA6_imag, vecA6_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA7_imag, vecA7_imag);

    *C = sqrt(vaddvq_f64(sum_vec));
}

int32_t Ne10Util::NormSquareComplexVectorDouble1x16(const double *A_real, const double *A_imag, double *C)
{
    float64x2_t sum_vec = vmovq_n_f64(0);

    float64x2_t vecA0_real = vld1q_f64(A_real);
    float64x2_t vecA1_real = vld1q_f64(A_real + 2);
    float64x2_t vecA2_real = vld1q_f64(A_real + 4);
    float64x2_t vecA3_real = vld1q_f64(A_real + 6);

    float64x2_t vecA4_real = vld1q_f64(A_real + 8);
    float64x2_t vecA5_real = vld1q_f64(A_real + 10);
    float64x2_t vecA6_real = vld1q_f64(A_real + 12);
    float64x2_t vecA7_real = vld1q_f64(A_real + 14);

    float64x2_t vecA0_imag = vld1q_f64(A_imag);
    float64x2_t vecA1_imag = vld1q_f64(A_imag + 2);
    float64x2_t vecA2_imag = vld1q_f64(A_imag + 4);
    float64x2_t vecA3_imag = vld1q_f64(A_imag + 6);

    float64x2_t vecA4_imag = vld1q_f64(A_imag + 8);
    float64x2_t vecA5_imag = vld1q_f64(A_imag + 10);
    float64x2_t vecA6_imag = vld1q_f64(A_imag + 12);
    float64x2_t vecA7_imag = vld1q_f64(A_imag + 14);

    sum_vec = vfmaq_f64(sum_vec, vecA0_real, vecA0_real);
    sum_vec = vfmaq_f64(sum_vec, vecA1_real, vecA1_real);
    sum_vec = vfmaq_f64(sum_vec, vecA2_real, vecA2_real);
    sum_vec = vfmaq_f64(sum_vec, vecA3_real, vecA3_real);

    sum_vec = vfmaq_f64(sum_vec, vecA4_real, vecA4_real);
    sum_vec = vfmaq_f64(sum_vec, vecA5_real, vecA5_real);
    sum_vec = vfmaq_f64(sum_vec, vecA6_real, vecA6_real);
    sum_vec = vfmaq_f64(sum_vec, vecA7_real, vecA7_real);

    sum_vec = vfmaq_f64(sum_vec, vecA0_imag, vecA0_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA1_imag, vecA1_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA2_imag, vecA2_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA3_imag, vecA3_imag);

    sum_vec = vfmaq_f64(sum_vec, vecA4_imag, vecA4_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA5_imag, vecA5_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA6_imag, vecA6_imag);
    sum_vec = vfmaq_f64(sum_vec, vecA7_imag, vecA7_imag);

    *C = vaddvq_f64(sum_vec);
}

int32_t Ne10Util::NormalizeComplexVectorFloat(float *A_real, float *A_imag, const size_t &vectorSize)
{
}

int32_t Ne10Util::NormalizeComplexVectorDouble(double *A_real, double *A_imag, const size_t &vectorSize)
{
}

int32_t Ne10Util::MatrixMultDouble(const double *A, const size_t &n, const size_t &m1, const double *B, const size_t &m2,
                                   const size_t &k, double *C)
{
    if (m1 != m2)
    {
        return -1;
    }
    constexpr uint32 blockSize = 2;
    //  n, m, k must be multiples of 2.

    omp_set_num_threads(4); // 4 different cores!
    pinThreadToCore();

#pragma omp parallel
    {
// Define variables in the inner-scope so that they become private to each thread!
#pragma omp for
        // A -> (nxm)
        // B -> (mxk)
        // C -> (nxk)
        for (size_t i_idx = 0; i_idx < n; i_idx += blockSize) // Loop over rows of C (and A)
        {
            // rows of A
            float64x2_t A0;
            float64x2_t A1;

            // rows of B
            float64x2_t B0;
            float64x2_t B1;

            // rows of C
            float64x2_t C0;
            float64x2_t C1;
            for (size_t k_idx = 0; k_idx < k; k_idx += blockSize) // Loop over columns of C (and B)
            {
                // Zero accumulators for C values
                C0 = vmovq_n_f64(0);
                C1 = vmovq_n_f64(0);

                for (size_t j_idx = 0; j_idx < m1; j_idx += blockSize) // Multiply and accumulate over block of A and B
                {
                    const size_t A_idx = i_idx * m1 + j_idx;
                    const size_t B_idx = j_idx * k + k_idx;

                    A0 = vld1q_f64(A + A_idx);
                    A1 = vld1q_f64(A + A_idx + m1);

                    B0 = vld1q_f64(B + B_idx);
                    B1 = vld1q_f64(B + B_idx + k);

                    C0 = vfmaq_laneq_f64(C0, B0, A0, 0);
                    C0 = vfmaq_laneq_f64(C0, B1, A0, 1);

                    C1 = vfmaq_laneq_f64(C1, B0, A1, 0);
                    C1 = vfmaq_laneq_f64(C1, B1, A1, 1);
                }
                const size_t C_idx = k * i_idx + k_idx; // C{i,k}
                vst1q_f64(C + C_idx, C0);
                vst1q_f64(C + C_idx + k, C1); // You need to write by rows, hole row must be traversed!!
            }
        }
    }
    return 0;
}

int32_t Ne10Util::MatrixMultComplexDouble(const double *A_real, const double *A_imag, const size_t &n,
                                          const size_t &m, const double *B_real, const double *B_imag, const size_t &m2, const size_t &k,
                                          double *C_real, double *C_imag)
{

    // Important! There are 32 128-bit NEON registers for ARM v8-A AArch64. We exactly use 32 registers! Since we do not
    // have more registers, we need to overwrite some of the used register values. This hardens to readibility.
    // Check the paper "Acceleration of complex matrix multiplication using arbitrary precision floating-point arithmetic"
    if (m != m2)
    {
        return -1;
    }
    constexpr uint32 blockSize = 2;

    omp_set_num_threads(4); // 4 different cores!
    pinThreadToCore();

#pragma omp parallel
    {
// Define variables in the inner-scope so that they become private to each thread!
#pragma omp for
        // A -> (nxm)
        // B -> (mxk)
        // C -> (nxk)
        for (size_t i_idx = 0; i_idx < n; i_idx += blockSize)
        {
            float64x2_t A0_real, A0_imag;
            float64x2_t A1_real, A1_imag;

            float64x2_t B0_real, B0_imag;
            float64x2_t B1_real, B1_imag;

            float64x2_t C0_real, C0_imag;
            float64x2_t C1_real, C1_imag;
            for (size_t k_idx = 0; k_idx < k; k_idx += blockSize)
            {

                C0_real = vmovq_n_f64(0);
                C0_imag = vmovq_n_f64(0);

                C1_real = vmovq_n_f64(0);
                C1_imag = vmovq_n_f64(0);

                // Needed for 3M method (Karatsuba). We are doing complex multiplications, so we need cross multiplications!
                float64x2_t temp0_A;
                float64x2_t temp0_B;

                float64x2_t temp1_A;
                float64x2_t temp1_B;

                for (size_t j_idx = 0; j_idx < m; j_idx += blockSize)
                {
                    const size_t A_idx = i_idx * m + j_idx;
                    const size_t B_idx = j_idx * k + k_idx;

                    A0_real = vld1q_f64(A_real + A_idx); // Travel Rows of A_{i,k}
                    A1_real = vld1q_f64(A_real + A_idx + m);

                    A0_imag = vld1q_f64(A_imag + A_idx);
                    A1_imag = vld1q_f64(A_imag + A_idx + m);

                    B0_real = vld1q_f64(B_real + B_idx);
                    B1_real = vld1q_f64(B_real + B_idx + k);

                    B0_imag = vld1q_f64(B_imag + B_idx);
                    B1_imag = vld1q_f64(B_imag + B_idx + k);

                    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    ///
                    /// CALCULATE SUMMATION AND CROSS MULTIPLICATION
                    // First calculate the cross value because there are not enough registers!
                    // Summation part
                    temp0_A = vaddq_f64(A0_imag, A0_real);
                    temp1_A = vaddq_f64(A1_imag, A1_real);

                    temp0_B = vaddq_f64(B0_imag, B0_real);
                    temp1_B = vaddq_f64(B1_imag, B1_real);

                    // Cross multiplication, since we do not have enough registers, write on A{index}_real registers.
                    A0_real = vmovq_n_f64(0);
                    A0_real = vfmaq_laneq_f64(A0_real, temp0_B, temp0_A, 0);
                    A0_real = vfmaq_laneq_f64(A0_real, temp1_B, temp0_A, 1);

                    A1_real = vmovq_n_f64(0);
                    A1_real = vfmaq_laneq_f64(A1_real, temp0_B, temp1_A, 0);
                    A1_real = vfmaq_laneq_f64(A1_real, temp1_B, temp1_A, 1);

                    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    ///
                    /// CALCULATE T2
                    // Now use  temp{index}_B to calculate pure  imaginary multiplication of A and B
                    // Pure Imaginary part as a matrix multiplication (T2 = Im(A) Im(B))
                    temp0_B = vmovq_n_f64(0);
                    temp0_B = vfmaq_laneq_f64(temp0_B, B0_imag, A0_imag, 0);
                    temp0_B = vfmaq_laneq_f64(temp0_B, B1_imag, A0_imag, 1);

                    temp1_B = vmovq_n_f64(0);
                    temp1_B = vfmaq_laneq_f64(temp1_B, B0_imag, A1_imag, 0);
                    temp1_B = vfmaq_laneq_f64(temp1_B, B1_imag, A1_imag, 1);

                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    ///
                    // A{index}_real = Cross Value - T2.
                    A0_real = vsubq_f64(A0_real, temp0_B);
                    A1_real = vsubq_f64(A1_real, temp1_B);

                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    /// CALCULATE T1
                    // We load Re(A) values again because we used it in the cross-calculation part.

                    temp0_A = vld1q_f64(A_real + A_idx);
                    temp1_A = vld1q_f64(A_real + A_idx + m);

                    // ATTENTION!!!! We use B{index}_imag for calculation on purpose!
                    B0_imag = vmovq_n_f64(0);
                    B1_imag = vmovq_n_f64(0);

                    // Pure Real part as a matrix multiplication (T1 = Re(A) Re(B))
                    B0_imag = vfmaq_laneq_f64(B0_imag, B0_real, temp0_A, 0);
                    B0_imag = vfmaq_laneq_f64(B0_imag, B1_real, temp0_A, 1);

                    B1_imag = vfmaq_laneq_f64(B1_imag, B0_real, temp1_A, 0);
                    B1_imag = vfmaq_laneq_f64(B1_imag, B1_real, temp1_A, 1);

                    // T1 = B{index}_imag = Re(A) * Re(B). It is on purpose!! There is no available temporary register!

                    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    /// CALCULATE Im(AB)
                    // (Cross value - T2) - T1
                    A0_real = vsubq_f64(A0_real, B0_imag);
                    A1_real = vsubq_f64(A1_real, B1_imag);

                    // This addition is required because we are doing block-matrix multiplication!
                    C0_imag = vaddq_f64(C0_imag, A0_real);
                    C1_imag = vaddq_f64(C1_imag, A1_real);

                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    /// CALCULATE Re(AB)
                    // T1 - T2
                    B0_imag = vsubq_f64(B0_imag, temp0_B);
                    B1_imag = vsubq_f64(B1_imag, temp1_B);

                    C0_real = vaddq_f64(C0_real, B0_imag);
                    C1_real = vaddq_f64(C1_real, B1_imag);
                }
                const size_t C_idx = k * i_idx + k_idx; // C{i,k}
                vst1q_f64(C_real + C_idx, C0_real);
                vst1q_f64(C_real + C_idx + k, C1_real);

                vst1q_f64(C_imag + C_idx, C0_imag);
                vst1q_f64(C_imag + C_idx + k, C1_imag);
            }
        }
    }
    return 0;
}

int32_t Ne10Util::VectorDotProductFloat(const float *A, const float *B, const size_t &vectorSize, float *C)
{
    if (0 != vectorSize % 4)
    {
        return -1;
    }
    constexpr uint32 registerElementSize = 4;

    float32x4_t vecC = vmovq_n_f32(0);

    for (size_t i_idx = 0; i_idx < vectorSize; i_idx += registerElementSize)
    {
        float32x4_t vecA = vld1q_f32(A + i_idx);
        float32x4_t vecB = vld1q_f32(B + i_idx);
        vecC = vfmaq_f32(vecC, vecA, vecB);
    }
    *C = vaddvq_f32(vecC);
    return 0;
}

int32_t Ne10Util::VectorDotProductDouble(const double *A, const double *B, const size_t &vectorSize, double *C)
{
    if (0 != vectorSize % 2)
    {
        return -1;
    }
    constexpr uint32 registerElementSize = 2;

    float64x2_t vecC = vmovq_n_f64(0);

    for (size_t i_idx = 0; i_idx < vectorSize; i_idx += registerElementSize)
    {
        float64x2_t vecA = vld1q_f64(A + i_idx);
        float64x2_t vecB = vld1q_f64(B + i_idx);
        vecC = vfmaq_f64(vecC, vecA, vecB);
    }
    *C = vaddvq_f64(vecC);
    return 0;
}

int32_t Ne10Util::VectorDotProductComplexFloat(const float *A_real, const float *A_imag, const float *B_real,
                                               const float *B_imag, const size_t &vectorSize, float *C_real, float *C_imag)
{
    if (0 != vectorSize % 4)
    {
        return -1;
    }
    constexpr uint32 registerElementSize = 4;

    float32x4_t vecC_real = vmovq_n_f32(0);
    float32x4_t vecC_imag = vmovq_n_f32(0);

    for (size_t i_idx = 0; i_idx < vectorSize; i_idx += registerElementSize)
    {
        float32x4_t vecA_real = vld1q_f32(A_real + i_idx);
        float32x4_t vecA_imag = vld1q_f32(A_imag + i_idx);

        float32x4_t vecB_real = vld1q_f32(B_real + i_idx);
        float32x4_t vecB_imag = vld1q_f32(B_imag + i_idx);

        float32x4_t vecC_realTemp;
        float32x4_t vecC_imagTemp;

        float32x4_t temp_A, temp_B;

        temp_A = vaddq_f32(vecA_real, vecA_imag);

        temp_B = vaddq_f32(vecB_real, vecB_imag);

        // Cross Term (CR)
        temp_A = vmulq_f32(temp_A, temp_B);

        // t1
        temp_B = vmulq_f32(vecA_real, vecB_real);

        // t2
        vecC_imagTemp = vmulq_f32(vecA_imag, vecB_imag);

        // CR - t2
        temp_A = vsubq_f32(temp_A, vecC_imagTemp);

        // Re(ab): t1 - t2
        vecC_realTemp = vsubq_f32(temp_B, vecC_imagTemp);

        // Im(ab): (CR - t2) - t1
        vecC_imagTemp = vsubq_f32(temp_A, temp_B);

        // Sum the results in register for acceleration!
        vecC_real = vaddq_f32(vecC_real, vecC_realTemp);
        vecC_imag = vaddq_f32(vecC_imag, vecC_imagTemp);
    }
    *C_real = vaddvq_f32(vecC_real);
    *C_imag = vaddvq_f32(vecC_imag);

    return 0;
}

int32_t Ne10Util::VectorDotProductComplexDouble(const double *A_real, const double *A_imag, const double *B_real,
                                                const double *B_imag, const size_t &vectorSize, double *C_real, double *C_imag)
{
    if (0 != vectorSize % 4)
    {
        return -1;
    }
    constexpr uint32 registerElementSize = 2;

    float64x2_t vecC_real = vmovq_n_f64(0);
    float64x2_t vecC_imag = vmovq_n_f64(0);

    for (size_t i_idx = 0; i_idx < vectorSize; i_idx += registerElementSize)
    {
        float64x2_t vecA_real = vld1q_f64(A_real + i_idx);
        float64x2_t vecA_imag = vld1q_f64(A_imag + i_idx);

        float64x2_t vecB_real = vld1q_f64(B_real + i_idx);
        float64x2_t vecB_imag = vld1q_f64(B_imag + i_idx);

        float64x2_t vecC_realTemp;
        float64x2_t vecC_imagTemp;

        float64x2_t temp_A, temp_B;

        temp_A = vaddq_f64(vecA_real, vecA_imag);

        temp_B = vaddq_f64(vecB_real, vecB_imag);

        // Cross Term (CR)
        temp_A = vmulq_f64(temp_A, temp_B);

        // t1
        temp_B = vmulq_f64(vecA_real, vecB_real);

        // t2
        vecC_imagTemp = vmulq_f64(vecA_imag, vecB_imag);

        // CR - t2
        temp_A = vsubq_f64(temp_A, vecC_imagTemp);

        // Re(ab): t1 - t2
        vecC_realTemp = vsubq_f64(temp_B, vecC_imagTemp);

        // Im(ab): (CR - t2) - t1
        vecC_imagTemp = vsubq_f64(temp_A, temp_B);

        // Sum the results in register for acceleration!
        vecC_real = vaddq_f64(vecC_real, vecC_realTemp);
        vecC_imag = vaddq_f64(vecC_imag, vecC_imagTemp);
    }
    *C_real = vaddvq_f64(vecC_real);
    *C_imag = vaddvq_f64(vecC_imag);

    return 0;
}

int32_t Ne10Util::VectorDotProductFloat1x16(const float *A, const float *B, float *C)
{

    float32x4_t vecA0 = vld1q_f32(A);
    float32x4_t vecA1 = vld1q_f32(A + 4);
    float32x4_t vecA2 = vld1q_f32(A + 8);
    float32x4_t vecA3 = vld1q_f32(A + 12);

    float32x4_t vecB0 = vld1q_f32(B);
    float32x4_t vecB1 = vld1q_f32(B + 4);
    float32x4_t vecB2 = vld1q_f32(B + 8);
    float32x4_t vecB3 = vld1q_f32(B + 12);

    float32x4_t vecC0;
    float32x4_t vecC1;
    float32x4_t vecC2;
    float32x4_t vecC3;

    vecC0 = vmulq_f32(vecA0, vecB0);
    vecC1 = vmulq_f32(vecA1, vecB1);
    vecC2 = vmulq_f32(vecA2, vecB2);
    vecC3 = vmulq_f32(vecA3, vecB3);

    vecC0 = vaddq_f32(vecC0, vecC1);
    vecC2 = vaddq_f32(vecC2, vecC3);

    vecC0 = vaddq_f32(vecC0, vecC2);
    *C = vaddvq_f32(vecC0);
    return 0;
}

int32_t Ne10Util::VectorDotProductDouble1x16(const double *A, const double *B, double *C)
{
    float64x2_t vecA0 = vld1q_f64(A);
    float64x2_t vecA1 = vld1q_f64(A + 2);
    float64x2_t vecA2 = vld1q_f64(A + 4);
    float64x2_t vecA3 = vld1q_f64(A + 6);
    float64x2_t vecA4 = vld1q_f64(A + 8);
    float64x2_t vecA5 = vld1q_f64(A + 10);
    float64x2_t vecA6 = vld1q_f64(A + 12);
    float64x2_t vecA7 = vld1q_f64(A + 14);

    float64x2_t vecB0 = vld1q_f64(B);
    float64x2_t vecB1 = vld1q_f64(B + 2);
    float64x2_t vecB2 = vld1q_f64(B + 4);
    float64x2_t vecB3 = vld1q_f64(B + 6);
    float64x2_t vecB4 = vld1q_f64(B + 8);
    float64x2_t vecB5 = vld1q_f64(B + 10);
    float64x2_t vecB6 = vld1q_f64(B + 12);
    float64x2_t vecB7 = vld1q_f64(B + 14);

    float64x2_t vecC0;
    float64x2_t vecC1;
    float64x2_t vecC2;
    float64x2_t vecC3;
    float64x2_t vecC4;
    float64x2_t vecC5;
    float64x2_t vecC6;
    float64x2_t vecC7;

    vecC0 = vmulq_f64(vecA0, vecB0);
    vecC1 = vmulq_f64(vecA1, vecB1);
    vecC2 = vmulq_f64(vecA2, vecB2);
    vecC3 = vmulq_f64(vecA3, vecB3);
    vecC4 = vmulq_f64(vecA4, vecB4);
    vecC5 = vmulq_f64(vecA5, vecB5);
    vecC6 = vmulq_f64(vecA6, vecB6);
    vecC7 = vmulq_f64(vecA7, vecB7);

    vecC0 = vaddq_f64(vecC0, vecC1);
    vecC2 = vaddq_f64(vecC2, vecC3);
    vecC4 = vaddq_f64(vecC4, vecC5);
    vecC6 = vaddq_f64(vecC6, vecC7);

    vecC0 = vaddq_f64(vecC0, vecC2);
    vecC4 = vaddq_f64(vecC4, vecC6);

    vecC0 = vaddq_f64(vecC0, vecC4);

    *C = vaddvq_f64(vecC0);
    return 0;
}

int32_t Ne10Util::VectorDotProductComplexFloat1x16(const float *A_real, const float *A_imag, const float *B_real, const float *B_imag, float *C_real, float *C_imag)
{
    float32x4_t vecA0_real = vld1q_f32(A_real);
    float32x4_t vecA1_real = vld1q_f32(A_real + 4);
    float32x4_t vecA2_real = vld1q_f32(A_real + 8);
    float32x4_t vecA3_real = vld1q_f32(A_real + 12);

    float32x4_t vecA0_imag = vld1q_f32(A_imag);
    float32x4_t vecA1_imag = vld1q_f32(A_imag + 4);
    float32x4_t vecA2_imag = vld1q_f32(A_imag + 8);
    float32x4_t vecA3_imag = vld1q_f32(A_imag + 12);

    float32x4_t vecB0_real = vld1q_f32(B_real);
    float32x4_t vecB1_real = vld1q_f32(B_real + 4);
    float32x4_t vecB2_real = vld1q_f32(B_real + 8);
    float32x4_t vecB3_real = vld1q_f32(B_real + 12);

    float32x4_t vecB0_imag = vld1q_f32(B_imag);
    float32x4_t vecB1_imag = vld1q_f32(B_imag + 4);
    float32x4_t vecB2_imag = vld1q_f32(B_imag + 8);
    float32x4_t vecB3_imag = vld1q_f32(B_imag + 12);

    float32x4_t vecC0_real, vecC0_imag;
    float32x4_t vecC1_real, vecC1_imag;
    float32x4_t vecC2_real, vecC2_imag;
    float32x4_t vecC3_real, vecC3_imag;

    float32x4_t temp0_A, temp0_B;
    float32x4_t temp1_A, temp1_B;
    float32x4_t temp2_A, temp2_B;
    float32x4_t temp3_A, temp3_B;

    temp0_A = vaddq_f32(vecA0_real, vecA0_imag);
    temp1_A = vaddq_f32(vecA1_real, vecA1_imag);
    temp2_A = vaddq_f32(vecA2_real, vecA2_imag);
    temp3_A = vaddq_f32(vecA3_real, vecA3_imag);

    temp0_B = vaddq_f32(vecB0_real, vecB0_imag);
    temp1_B = vaddq_f32(vecB1_real, vecB1_imag);
    temp2_B = vaddq_f32(vecB2_real, vecB2_imag);
    temp3_B = vaddq_f32(vecB3_real, vecB3_imag);

    // Cross Term (CR)
    temp0_A = vmulq_f32(temp0_A, temp0_B);
    temp1_A = vmulq_f32(temp1_A, temp1_B);
    temp2_A = vmulq_f32(temp2_A, temp2_B);
    temp3_A = vmulq_f32(temp3_A, temp3_B);

    // t1
    temp0_B = vmulq_f32(vecA0_real, vecB0_real);
    temp1_B = vmulq_f32(vecA1_real, vecB1_real);
    temp2_B = vmulq_f32(vecA2_real, vecB2_real);
    temp3_B = vmulq_f32(vecA3_real, vecB3_real);

    // t2
    vecC0_imag = vmulq_f32(vecA0_imag, vecB0_imag);
    vecC1_imag = vmulq_f32(vecA1_imag, vecB1_imag);
    vecC2_imag = vmulq_f32(vecA2_imag, vecB2_imag);
    vecC3_imag = vmulq_f32(vecA3_imag, vecB3_imag);

    // CR - t2
    temp0_A = vsubq_f32(temp0_A, vecC0_imag);
    temp1_A = vsubq_f32(temp1_A, vecC1_imag);
    temp2_A = vsubq_f32(temp2_A, vecC2_imag);
    temp3_A = vsubq_f32(temp3_A, vecC3_imag);

    // Re(ab): t1 - t2
    vecC0_real = vsubq_f32(temp0_B, vecC0_imag);
    vecC1_real = vsubq_f32(temp1_B, vecC1_imag);
    vecC2_real = vsubq_f32(temp2_B, vecC2_imag);
    vecC3_real = vsubq_f32(temp3_B, vecC3_imag);

    // Im(ab): (CR - t2) - t1
    vecC0_imag = vsubq_f32(temp0_A, temp0_B);
    vecC1_imag = vsubq_f32(temp1_A, temp1_B);
    vecC2_imag = vsubq_f32(temp2_A, temp2_B);
    vecC3_imag = vsubq_f32(temp3_A, temp3_B);

    // Now sum the results
    vecC0_real = vaddq_f32(vecC0_real, vecC1_real);
    vecC2_real = vaddq_f32(vecC2_real, vecC3_real);

    vecC0_real = vaddq_f32(vecC0_real, vecC2_real);
    *C_real = vaddvq_f32(vecC0_real);

    vecC0_imag = vaddq_f32(vecC0_imag, vecC1_imag);
    vecC2_imag = vaddq_f32(vecC2_imag, vecC3_imag);

    vecC0_imag = vaddq_f32(vecC0_imag, vecC2_imag);
    *C_imag = vaddvq_f32(vecC0_imag);

    return 0;
}

int32_t Ne10Util::VectorDotProductComplexDouble1x16(const double *A_real, const double *A_imag, const double *B_real, const double *B_imag, double *C_real, double *C_imag)
{
    constexpr uint32 registerBlockSize = 8; // Process 8 elements consecutively.
    constexpr uint32 vectorSize = 16;
    *C_real = 0;
    *C_imag = 0;
    for (uint32 idx = 0; idx < vectorSize; idx += registerBlockSize)
    {
        float64x2_t vecA0_real = vld1q_f64(A_real + idx);
        float64x2_t vecA1_real = vld1q_f64(A_real + idx + 2);
        float64x2_t vecA2_real = vld1q_f64(A_real + idx + 4);
        float64x2_t vecA3_real = vld1q_f64(A_real + idx + 6);

        float64x2_t vecA0_imag = vld1q_f64(A_imag + idx);
        float64x2_t vecA1_imag = vld1q_f64(A_imag + idx + 2);
        float64x2_t vecA2_imag = vld1q_f64(A_imag + idx + 4);
        float64x2_t vecA3_imag = vld1q_f64(A_imag + idx + 6);

        float64x2_t vecB0_real = vld1q_f64(B_real + idx);
        float64x2_t vecB1_real = vld1q_f64(B_real + idx + 2);
        float64x2_t vecB2_real = vld1q_f64(B_real + idx + 4);
        float64x2_t vecB3_real = vld1q_f64(B_real + idx + 6);

        float64x2_t vecB0_imag = vld1q_f64(B_imag + idx);
        float64x2_t vecB1_imag = vld1q_f64(B_imag + idx + 2);
        float64x2_t vecB2_imag = vld1q_f64(B_imag + idx + 4);
        float64x2_t vecB3_imag = vld1q_f64(B_imag + idx + 6);

        float64x2_t vecC0_real, vecC0_imag;
        float64x2_t vecC1_real, vecC1_imag;
        float64x2_t vecC2_real, vecC2_imag;
        float64x2_t vecC3_real, vecC3_imag;

        float64x2_t temp0_A, temp0_B;
        float64x2_t temp1_A, temp1_B;
        float64x2_t temp2_A, temp2_B;
        float64x2_t temp3_A, temp3_B;

        temp0_A = vaddq_f64(vecA0_real, vecA0_imag);
        temp1_A = vaddq_f64(vecA1_real, vecA1_imag);
        temp2_A = vaddq_f64(vecA2_real, vecA2_imag);
        temp3_A = vaddq_f64(vecA3_real, vecA3_imag);

        temp0_B = vaddq_f64(vecB0_real, vecB0_imag);
        temp1_B = vaddq_f64(vecB1_real, vecB1_imag);
        temp2_B = vaddq_f64(vecB2_real, vecB2_imag);
        temp3_B = vaddq_f64(vecB3_real, vecB3_imag);

        // Cross Term (CR)
        temp0_A = vmulq_f64(temp0_A, temp0_B);
        temp1_A = vmulq_f64(temp1_A, temp1_B);
        temp2_A = vmulq_f64(temp2_A, temp2_B);
        temp3_A = vmulq_f64(temp3_A, temp3_B);

        // t1
        temp0_B = vmulq_f64(vecA0_real, vecB0_real);
        temp1_B = vmulq_f64(vecA1_real, vecB1_real);
        temp2_B = vmulq_f64(vecA2_real, vecB2_real);
        temp3_B = vmulq_f64(vecA3_real, vecB3_real);

        // t2
        vecC0_imag = vmulq_f64(vecA0_imag, vecB0_imag);
        vecC1_imag = vmulq_f64(vecA1_imag, vecB1_imag);
        vecC2_imag = vmulq_f64(vecA2_imag, vecB2_imag);
        vecC3_imag = vmulq_f64(vecA3_imag, vecB3_imag);

        // CR - t2
        temp0_A = vsubq_f64(temp0_A, vecC0_imag);
        temp1_A = vsubq_f64(temp1_A, vecC1_imag);
        temp2_A = vsubq_f64(temp2_A, vecC2_imag);
        temp3_A = vsubq_f64(temp3_A, vecC3_imag);

        // Re(ab): t1 - t2
        vecC0_real = vsubq_f64(temp0_B, vecC0_imag);
        vecC1_real = vsubq_f64(temp1_B, vecC1_imag);
        vecC2_real = vsubq_f64(temp2_B, vecC2_imag);
        vecC3_real = vsubq_f64(temp3_B, vecC3_imag);

        // Im(ab): (CR - t2) - t1
        vecC0_imag = vsubq_f64(temp0_A, temp0_B);
        vecC1_imag = vsubq_f64(temp1_A, temp1_B);
        vecC2_imag = vsubq_f64(temp2_A, temp2_B);
        vecC3_imag = vsubq_f64(temp3_A, temp3_B);

        // Calculate the sums for first 8 elements then sum it with last 8 elements.
        vecC0_real = vaddq_f64(vecC0_real, vecC1_real);
        vecC2_real = vaddq_f64(vecC2_real, vecC3_real);

        vecC0_real = vaddq_f64(vecC0_real, vecC2_real);
        *C_real += vaddvq_f64(vecC0_real);

        vecC0_imag = vaddq_f64(vecC0_imag, vecC1_imag);
        vecC2_imag = vaddq_f64(vecC2_imag, vecC3_imag);

        vecC0_imag = vaddq_f64(vecC0_imag, vecC2_imag);
        *C_imag += vaddvq_f64(vecC0_imag);
    }
    return 0;
}

int32_t Ne10Util::VectorComplexConjugateFloat(const float *A_imag, float *A_conjugatedImag, const size_t &vectorSize)
{
    if (0 != vectorSize % 4)
    {
        return -1;
    }
    constexpr uint32 registerElementSize = 4;
    for (size_t i = 0; i < vectorSize; i += registerElementSize)
    {
        float32x4_t imagVec = vld1q_f32(A_imag + i);
        imagVec = vnegq_f32(imagVec);
        vst1q_f32(A_conjugatedImag + i, imagVec);
    }
    return 0;
}

int32_t Ne10Util::VectorComplexConjugateDouble(const double *A_imag, double *A_conjugatedImag, const size_t &vectorSize)
{
    if (0 != vectorSize % 2)
    {
        return -1;
    }
    constexpr uint32 registerElementSize = 2;
    for (size_t i = 0; i < vectorSize; i += registerElementSize)
    {
        float64x2_t imagVec = vld1q_f64(A_imag + i);
        imagVec = vnegq_f64(imagVec);
        vst1q_f64(A_conjugatedImag + i, imagVec);
    }
    return 0;
}

int32_t Ne10Util::VectorComplexConjugateFloat1x16(const float *A_imag, float *A_conjugatedImag)
{
    float32x4_t vecA0_imag = vld1q_f32(A_imag);
    float32x4_t vecA1_imag = vld1q_f32(A_imag + 4);
    float32x4_t vecA2_imag = vld1q_f32(A_imag + 8);
    float32x4_t vecA3_imag = vld1q_f32(A_imag + 12);

    vecA0_imag = vnegq_f32(vecA0_imag);
    vecA1_imag = vnegq_f32(vecA1_imag);
    vecA2_imag = vnegq_f32(vecA2_imag);
    vecA3_imag = vnegq_f32(vecA3_imag);

    vst1q_f32(A_conjugatedImag, vecA0_imag);
    vst1q_f32(A_conjugatedImag + 4, vecA1_imag);
    vst1q_f32(A_conjugatedImag + 8, vecA2_imag);
    vst1q_f32(A_conjugatedImag + 12, vecA3_imag);

    return 0;
}

int32_t Ne10Util::VectorComplexConjugateDouble1x16(const double *A_imag, double *A_conjugatedImag)
{
    float64x2_t vecA0_imag = vld1q_f64(A_imag);
    float64x2_t vecA1_imag = vld1q_f64(A_imag + 2);
    float64x2_t vecA2_imag = vld1q_f64(A_imag + 4);
    float64x2_t vecA3_imag = vld1q_f64(A_imag + 6);

    float64x2_t vecA4_imag = vld1q_f64(A_imag + 8);
    float64x2_t vecA5_imag = vld1q_f64(A_imag + 10);
    float64x2_t vecA6_imag = vld1q_f64(A_imag + 12);
    float64x2_t vecA7_imag = vld1q_f64(A_imag + 14);

    vecA0_imag = vnegq_f64(vecA0_imag);
    vecA1_imag = vnegq_f64(vecA1_imag);
    vecA2_imag = vnegq_f64(vecA2_imag);
    vecA3_imag = vnegq_f64(vecA3_imag);

    vecA4_imag = vnegq_f64(vecA4_imag);
    vecA5_imag = vnegq_f64(vecA5_imag);
    vecA6_imag = vnegq_f64(vecA6_imag);
    vecA7_imag = vnegq_f64(vecA7_imag);

    vst1q_f64(A_conjugatedImag, vecA0_imag);
    vst1q_f64(A_conjugatedImag + 2, vecA1_imag);
    vst1q_f64(A_conjugatedImag + 4, vecA2_imag);
    vst1q_f64(A_conjugatedImag + 6, vecA3_imag);

    vst1q_f64(A_conjugatedImag + 8, vecA4_imag);
    vst1q_f64(A_conjugatedImag + 10, vecA5_imag);
    vst1q_f64(A_conjugatedImag + 12, vecA6_imag);
    vst1q_f64(A_conjugatedImag + 14, vecA7_imag);

    return 0;
}

int32_t Ne10Util::MatrixTransposeFloat(const float *A, const uint32 &n, const uint32 &m, float *A_transposed)
{
    if (0 != n % 4 || 0 != m % 4)
    {
        return -1;
    }
    constexpr size_t registerElementSize = 4;
    for (size_t i = 0; i < n; i += registerElementSize)
    {
        for (size_t j = 0; j < m; j += registerElementSize)
        {
            float32x4_t row0 = vld1q_f32(A + i * m + j);
            float32x4_t row1 = vld1q_f32(A + (i + 1) * m + j);
            float32x4_t row2 = vld1q_f32(A + (i + 2) * m + j);
            float32x4_t row3 = vld1q_f32(A + (i + 3) * m + j);

            float32x4x2_t row01 = vtrnq_f32(row0, row1);
            float32x4x2_t row23 = vtrnq_f32(row2, row3);

            vst1q_f32(A_transposed + j * n + i, vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
            vst1q_f32(A_transposed + (j + 1) * n + i, vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
            vst1q_f32(A_transposed + (j + 2) * n + i, vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0])));
            vst1q_f32(A_transposed + (j + 3) * n + i, vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1])));
        }
    }

    return 0;
}

int32_t Ne10Util::MatrixTransposeDouble(const double *A, const uint32 &n, const uint32 &m, double *A_transposed)
{
    if (0 != n % 2 || 0 != m % 2)
    {
        return -1;
    }
    constexpr size_t registerElementSize = 2;
    for (size_t i = 0; i < n; i += registerElementSize)
    {
        for (size_t j = 0; j < m; j += registerElementSize)
        {
            float64x2_t row0 = vld1q_f64(A + i * m + j);
            float64x2_t row1 = vld1q_f64(A + (i + 1) * m + j);

            float64x2_t temp0 = vcombine_f64(vget_low_f64(row0), vget_low_f64(row1));
            float64x2_t temp1 = vcombine_f64(vget_high_f64(row0), vget_high_f64(row1));

            vst1q_f64(A_transposed + j * n + i, temp0);
            vst1q_f64(A_transposed + (j + 1) * n + i, temp1);
        }
    }

    return 0;
}

int32_t Ne10Util::MatrixTransposeDouble16x16(const double *A, double *A_transposed)
{
    constexpr size_t blockSize = 4;
    constexpr size_t matrixSize = 16;

    for (size_t i = 0; i < matrixSize; i += blockSize)
    {
        for (size_t j = 0; j < matrixSize; j += blockSize)
        {
            float64x2_t row00 = vld1q_f64(A + i * matrixSize + j);
            float64x2_t row01 = vld1q_f64(A + i * matrixSize + j + 2);

            float64x2_t row10 = vld1q_f64(A + (i + 1) * matrixSize + j);
            float64x2_t row11 = vld1q_f64(A + (i + 1) * matrixSize + j + 2);

            float64x2_t row20 = vld1q_f64(A + (i + 2) * matrixSize + j);
            float64x2_t row21 = vld1q_f64(A + (i + 2) * matrixSize + j + 2);

            float64x2_t row30 = vld1q_f64(A + (i + 3) * matrixSize + j);
            float64x2_t row31 = vld1q_f64(A + (i + 3) * matrixSize + j + 2);

            float64x2_t temp00 = vcombine_f64(vget_low_f64(row00), vget_low_f64(row10));
            float64x2_t temp01 = vcombine_f64(vget_high_f64(row00), vget_high_f64(row10));

            float64x2_t temp10 = vcombine_f64(vget_low_f64(row01), vget_low_f64(row11));
            float64x2_t temp11 = vcombine_f64(vget_high_f64(row01), vget_high_f64(row11));

            float64x2_t temp20 = vcombine_f64(vget_low_f64(row20), vget_low_f64(row30));
            float64x2_t temp21 = vcombine_f64(vget_high_f64(row20), vget_high_f64(row30));

            float64x2_t temp30 = vcombine_f64(vget_low_f64(row21), vget_low_f64(row31));
            float64x2_t temp31 = vcombine_f64(vget_high_f64(row21), vget_high_f64(row31));

            vst1q_f64(A_transposed + j * matrixSize + i, temp00);
            vst1q_f64(A_transposed + (j + 1) * matrixSize + i, temp01);
            vst1q_f64(A_transposed + (j + 2) * matrixSize + i, temp10);
            vst1q_f64(A_transposed + (j + 3) * matrixSize + i, temp11);

            vst1q_f64(A_transposed + j * matrixSize + i + 2, temp20);
            vst1q_f64(A_transposed + (j + 1) * matrixSize + i + 2, temp21);
            vst1q_f64(A_transposed + (j + 2) * matrixSize + i + 2, temp30);
            vst1q_f64(A_transposed + (j + 3) * matrixSize + i + 2, temp31);
        }
    }
    return 0;
}

void Ne10Util::pinThreadToCore()
{
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "cores", 1);
}

void Ne10Util::testComplexMultiplications()
{
    constexpr size_t M = 24;
    constexpr size_t N = 32;
    constexpr size_t K = 20;

    float A_real[M * N]{0.814724, 0.678735, 0.709365, 0.814285, 0.568824, 0.106653, 0.401808, 0.575209, 0.486792, 0.225922, 0.085516, 0.098712, 0.805489, 0.972975, 0.372410, 0.032601, 0.824376, 0.068806, 0.637709, 0.322472, 0.647618, 0.318074, 0.192028, 0.683416, 0.768854, 0.850713, 0.083483, 0.123084, 0.105709, 0.467068, 0.178117, 0.441722, 0.905792, 0.757740, 0.754687, 0.243525, 0.469391, 0.961898, 0.075967, 0.059780, 0.435859, 0.170708, 0.262482, 0.261871, 0.576722, 0.648991, 0.198118, 0.561200, 0.982663, 0.319600, 0.957694, 0.784739, 0.679017, 0.119215, 0.138874, 0.704047, 0.167254, 0.560560, 0.625960, 0.205494, 0.142041, 0.648198, 0.359635, 0.013283, 0.126987, 0.743132, 0.276025, 0.929264, 0.011902, 0.004634, 0.239916, 0.234780, 0.446784, 0.227664, 0.801015, 0.335357, 0.182922, 0.800331, 0.489688, 0.881867, 0.730249, 0.530864, 0.240707, 0.471357, 0.635787, 0.939829, 0.696266, 0.442305, 0.861980, 0.929609, 0.660945, 0.146515, 0.166460, 0.025228, 0.056705, 0.897191, 0.913376, 0.392227, 0.679703, 0.349984, 0.337123, 0.774910, 0.123319, 0.353159, 0.306349, 0.435699, 0.029220, 0.679728, 0.239932, 0.453798, 0.339493, 0.669175, 0.343877, 0.654446, 0.676122, 0.035763, 0.945174, 0.645552, 0.093820, 0.019578, 0.989872, 0.696667, 0.729752, 0.189072, 0.620959, 0.842207, 0.521886, 0.196658, 0.632359, 0.655478, 0.655098, 0.196595, 0.162182, 0.817303, 0.183908, 0.821194, 0.508509, 0.311102, 0.928854, 0.136553, 0.886512, 0.432392, 0.951630, 0.190433, 0.584069, 0.407619, 0.289065, 0.175874, 0.208935, 0.479463, 0.525404, 0.330858, 0.514423, 0.582791, 0.890752, 0.042652, 0.573710, 0.559033, 0.335849, 0.093371, 0.097540, 0.171187, 0.162612, 0.251084, 0.794285, 0.868695, 0.239953, 0.015403, 0.510772, 0.923380, 0.730331, 0.721227, 0.028674, 0.825314, 0.920332, 0.368917, 0.107769, 0.819981, 0.671808, 0.721758, 0.709282, 0.639317, 0.530344, 0.424309, 0.884281, 0.815397, 0.982303, 0.635198, 0.052078, 0.854100, 0.175669, 0.307367, 0.278498, 0.706046, 0.118998, 0.616045, 0.311215, 0.084436, 0.417267, 0.043024, 0.817628, 0.430207, 0.488609, 0.106762, 0.489901, 0.083470, 0.052677, 0.460726, 0.906308, 0.718359, 0.695140, 0.473486, 0.236231, 0.544716, 0.861140, 0.270270, 0.588026, 0.879014, 0.769029, 0.281867, 0.931201, 0.347879, 0.208947, 0.456058, 0.546882, 0.031833, 0.498364, 0.473289, 0.528533, 0.399783, 0.049654, 0.168990, 0.794831, 0.184816, 0.578525, 0.653757, 0.167927, 0.133171, 0.737858, 0.981638, 0.879654, 0.968649, 0.067993, 0.152721, 0.119396, 0.647311, 0.484853, 0.197054, 0.154752, 0.988912, 0.581446, 0.538597, 0.728662, 0.446027, 0.905154, 0.101669, 0.957507, 0.276923, 0.959744, 0.351660, 0.165649, 0.259870, 0.902716, 0.649115, 0.644318, 0.904881, 0.237284, 0.494174, 0.978681, 0.173389, 0.269119, 0.156405, 0.817761, 0.531334, 0.254790, 0.341125, 0.607304, 0.543886, 0.393456, 0.821721, 0.199863, 0.000522, 0.928313, 0.695163, 0.737842, 0.054239, 0.675391, 0.995390, 0.964889, 0.046171, 0.340386, 0.830829, 0.601982, 0.800068, 0.944787, 0.731722, 0.378609, 0.979748, 0.458849, 0.779052, 0.712694, 0.390938, 0.422836, 0.855523, 0.260728, 0.325146, 0.224040, 0.607389, 0.450138, 0.721047, 0.671431, 0.429921, 0.406955, 0.865439, 0.580090, 0.499116, 0.063405, 0.177108, 0.468468, 0.332093, 0.157613, 0.097132, 0.585268, 0.585264, 0.262971, 0.431414, 0.490864, 0.647746, 0.811580, 0.438870, 0.963089, 0.715037, 0.500472, 0.831380, 0.547871, 0.644765, 0.594356, 0.105629, 0.667833, 0.191745, 0.458725, 0.522495, 0.741258, 0.887771, 0.748706, 0.612566, 0.016983, 0.535801, 0.860441, 0.662808, 0.912132, 0.297347, 0.970593, 0.823458, 0.223812, 0.549724, 0.654079, 0.910648, 0.489253, 0.450924, 0.532826, 0.111119, 0.546806, 0.903721, 0.471088, 0.803364, 0.942737, 0.376272, 0.022513, 0.610959, 0.844392, 0.738427, 0.661945, 0.993705, 0.520052, 0.391183, 0.825584, 0.989950, 0.120860, 0.445183, 0.934405, 0.330829, 0.104012, 0.062045, 0.957167, 0.694829, 0.751267, 0.917194, 0.689215, 0.181847, 0.337719, 0.547009, 0.350727, 0.258065, 0.521136, 0.890923, 0.059619, 0.060471, 0.417744, 0.190924, 0.425259, 0.778802, 0.344462, 0.242850, 0.770286, 0.218677, 0.347713, 0.769114, 0.789963, 0.527680, 0.862711, 0.123932, 0.984398, 0.898486, 0.745546, 0.298244, 0.485376, 0.317099, 0.255095, 0.285839, 0.748152, 0.263803, 0.900054, 0.296321, 0.939002, 0.408720, 0.231594, 0.334163, 0.681972, 0.399258, 0.983052, 0.428253, 0.312719, 0.423453, 0.780520, 0.917424, 0.350218, 0.105798, 0.149997, 0.396792, 0.318524, 0.479523, 0.484297, 0.490357, 0.858939, 0.118155, 0.736267, 0.046351, 0.800280, 0.950222, 0.505957, 0.757200, 0.450542, 0.145539, 0.369247, 0.744693, 0.875943, 0.594896, 0.488898, 0.698746, 0.042431, 0.526876, 0.301455, 0.482022, 0.161485, 0.090823, 0.675332, 0.269062, 0.662010, 0.109697, 0.586092, 0.808514, 0.534064, 0.801348, 0.844856, 0.852998, 0.785559, 0.988418, 0.561861, 0.505428, 0.141886, 0.034446, 0.699077, 0.753729, 0.083821, 0.136069, 0.111203, 0.188955, 0.550156, 0.262212, 0.624060, 0.197810, 0.071445, 0.416799, 0.701099, 0.120612, 0.178766, 0.266471, 0.006715, 0.765500, 0.416159, 0.063591, 0.262145, 0.755077, 0.089951, 0.227843, 0.209405, 0.873927, 0.513377, 0.539982, 0.184194, 0.761426, 0.421761, 0.438744, 0.890903, 0.380446, 0.228977, 0.869292, 0.780252, 0.686775, 0.622475, 0.602843, 0.679136, 0.030541, 0.521650, 0.656860, 0.666339, 0.589507, 0.422886, 0.153657, 0.602170, 0.188662, 0.841929, 0.404580, 0.044454, 0.377396, 0.111706, 0.498094, 0.552291, 0.270294, 0.177602, 0.706917, 0.597211, 0.631070, 0.915736, 0.381558, 0.959291, 0.567822, 0.913337, 0.579705, 0.389739, 0.183511, 0.587045, 0.711216, 0.395515, 0.744074, 0.096730, 0.627973, 0.539126, 0.226188, 0.094229, 0.281005, 0.386771, 0.287498, 0.832917, 0.448373, 0.754933, 0.216019, 0.136293, 0.900852, 0.629883, 0.208461, 0.398589, 0.999492, 0.299937, 0.089892, 0.792207, 0.765517, 0.547216, 0.075854, 0.152378, 0.549860, 0.241691, 0.368485, 0.207742, 0.221747, 0.367437, 0.500022, 0.818149, 0.291984, 0.698106, 0.384619, 0.598524, 0.440085, 0.915991, 0.091113, 0.256441, 0.365816, 0.242785, 0.790407, 0.678652, 0.574661, 0.031991, 0.564980, 0.133931, 0.287849, 0.134123, 0.080862, 0.959492, 0.795200, 0.138624, 0.053950, 0.825817, 0.144955, 0.403912, 0.625619, 0.301246, 0.117418, 0.987982, 0.479922, 0.817547, 0.431651, 0.666528, 0.582986, 0.470924, 0.527143, 0.001151, 0.576209, 0.613461, 0.763505, 0.442402, 0.949304, 0.495177, 0.845178, 0.614713, 0.640312, 0.030890, 0.414523, 0.212602, 0.777241, 0.655741, 0.186873, 0.149294, 0.530798, 0.538342, 0.853031, 0.096455, 0.780227, 0.470923, 0.296676, 0.037739, 0.904722, 0.722440, 0.015487, 0.178132, 0.251806, 0.695949, 0.457424, 0.462449, 0.683363, 0.582249, 0.627896, 0.687796, 0.327565, 0.189710, 0.738640, 0.362411, 0.417029, 0.939142, 0.464840, 0.894942, 0.905135, 0.035712, 0.489764, 0.257508, 0.779167, 0.996135, 0.622055, 0.131973, 0.081126, 0.230488, 0.318778, 0.885168, 0.609867, 0.149865, 0.984064, 0.128014, 0.290441, 0.699888, 0.875372, 0.424349, 0.546593, 0.540739, 0.771980, 0.359228, 0.671264, 0.495006, 0.585987, 0.049533, 0.205976, 0.301306, 0.763957, 0.071453, 0.533772, 0.849129, 0.445586, 0.840717, 0.934011, 0.078176, 0.350952, 0.942051, 0.929386, 0.844309, 0.424167, 0.913287, 0.617666, 0.659605, 0.167168, 0.999080, 0.617091, 0.638531, 0.518052, 0.460916, 0.425729, 0.869941, 0.932854, 0.736340, 0.438645, 0.147608, 0.246735, 0.489570, 0.947933, 0.295534, 0.818204, 0.242487, 0.109154, 0.933993, 0.646313, 0.254282, 0.129906, 0.442678, 0.513250, 0.956135, 0.775713, 0.194764, 0.507858, 0.796184, 0.859442, 0.518595, 0.106216, 0.171121, 0.265281, 0.033604, 0.943623, 0.770160, 0.644443, 0.264779, 0.972741, 0.394707, 0.833501, 0.054974, 0.666416, 0.192510, 0.082071, 0.332936, 0.100222, 0.053754, 0.825809};
    float A_imag[M * N]{0.338098, 0.890476, 0.366437, 0.112284, 0.059403, 0.892922, 0.161134, 0.935731, 0.809204, 0.671202, 0.025135, 0.910570, 0.534138, 0.411594, 0.922332, 0.557295, 0.343288, 0.954174, 0.503781, 0.331665, 0.922745, 0.300819, 0.668464, 0.450394, 0.884405, 0.084247, 0.613475, 0.694803, 0.669043, 0.848709, 0.239291, 0.074090, 0.293973, 0.798960, 0.369199, 0.784428, 0.315811, 0.703223, 0.758112, 0.457886, 0.748619, 0.715213, 0.421112, 0.800559, 0.885359, 0.602638, 0.770954, 0.772495, 0.936027, 0.031923, 0.489594, 0.152234, 0.800372, 0.939410, 0.206776, 0.205672, 0.720856, 0.163898, 0.818641, 0.426456, 0.500211, 0.916821, 0.578923, 0.393883, 0.746313, 0.734341, 0.685028, 0.291570, 0.772722, 0.555738, 0.871111, 0.240478, 0.120187, 0.642061, 0.184100, 0.745847, 0.899005, 0.750520, 0.042660, 0.311940, 0.124774, 0.356869, 0.877049, 0.348008, 0.285947, 0.980904, 0.653851, 0.899651, 0.018613, 0.324220, 0.886235, 0.836270, 0.217994, 0.986968, 0.866887, 0.003394, 0.010337, 0.051332, 0.597942, 0.603533, 0.696433, 0.184434, 0.350777, 0.763898, 0.525045, 0.419048, 0.725775, 0.813113, 0.625938, 0.583533, 0.378186, 0.178982, 0.730585, 0.662654, 0.353142, 0.121658, 0.543663, 0.286620, 0.072052, 0.762586, 0.674776, 0.301727, 0.931112, 0.731387, 0.571616, 0.505133, 0.406777, 0.220677, 0.048447, 0.072885, 0.789364, 0.964423, 0.125332, 0.212031, 0.685536, 0.759327, 0.325834, 0.390762, 0.370363, 0.383306, 0.137869, 0.551793, 0.704340, 0.338956, 0.646477, 0.281502, 0.449444, 0.884153, 0.984776, 0.800820, 0.406727, 0.882486, 0.438509, 0.011681, 0.190785, 0.360031, 0.122189, 0.271422, 0.112615, 0.001301, 0.667916, 0.088527, 0.367653, 0.432485, 0.130151, 0.077347, 0.294149, 0.740648, 0.546449, 0.816140, 0.841560, 0.617279, 0.217802, 0.583571, 0.729513, 0.210146, 0.833152, 0.230383, 0.963530, 0.094278, 0.715678, 0.896111, 0.666932, 0.284950, 0.437820, 0.539905, 0.258582, 0.454212, 0.671166, 0.100751, 0.443846, 0.189180, 0.603468, 0.798351, 0.206028, 0.694752, 0.092352, 0.913800, 0.530629, 0.743688, 0.398881, 0.317428, 0.734230, 0.575495, 0.182141, 0.511820, 0.224277, 0.510153, 0.398282, 0.711129, 0.042298, 0.930041, 0.838970, 0.597527, 0.933726, 0.673226, 0.117037, 0.095373, 0.897866, 0.386390, 0.599586, 0.507849, 0.300184, 0.142484, 0.526102, 0.943008, 0.086667, 0.758099, 0.007820, 0.706715, 0.832423, 0.105920, 0.415093, 0.814540, 0.571026, 0.530052, 0.041820, 0.082593, 0.269055, 0.906364, 0.749822, 0.624573, 0.972958, 0.399020, 0.433261, 0.884017, 0.810950, 0.664280, 0.814682, 0.146515, 0.593362, 0.775555, 0.055976, 0.585609, 0.401387, 0.268076, 0.729709, 0.683716, 0.771934, 0.432642, 0.423109, 0.557789, 0.597490, 0.681560, 0.180738, 0.789074, 0.176855, 0.275070, 0.106942, 0.719570, 0.673031, 0.628924, 0.835221, 0.590609, 0.189207, 0.047401, 0.470625, 0.943732, 0.484548, 0.122815, 0.324855, 0.631141, 0.503840, 0.734271, 0.056343, 0.762887, 0.833364, 0.174892, 0.707253, 0.132083, 0.205675, 0.655498, 0.655573, 0.313429, 0.335311, 0.463261, 0.255387, 0.852264, 0.957384, 0.248629, 0.616443, 0.996156, 0.477492, 0.101534, 0.322460, 0.660438, 0.667120, 0.342374, 0.560713, 0.549158, 0.756749, 0.407318, 0.246228, 0.859320, 0.612810, 0.430278, 0.152501, 0.082963, 0.403629, 0.138649, 0.781377, 0.722725, 0.388272, 0.109755, 0.722923, 0.166204, 0.299225, 0.212163, 0.020536, 0.505637, 0.265322, 0.451639, 0.939661, 0.354534, 0.623716, 0.390855, 0.552262, 0.047555, 0.586440, 0.735966, 0.269092, 0.728387, 0.417047, 0.275287, 0.342713, 0.974222, 0.819422, 0.693753, 0.019621, 0.661596, 0.390176, 0.598886, 0.287977, 0.110353, 0.551779, 0.933760, 0.531209, 0.622497, 0.452593, 0.098519, 0.923676, 0.635661, 0.924581, 0.227713, 0.354456, 0.971259, 0.236445, 0.054617, 0.979129, 0.348785, 0.675112, 0.794682, 0.749018, 0.576758, 0.971786, 0.716670, 0.375692, 0.570838, 0.531889, 0.945213, 0.435176, 0.516979, 0.360449, 0.901058, 0.692532, 0.117493, 0.228953, 0.187461, 0.108818, 0.987935, 0.422646, 0.823574, 0.653700, 0.950894, 0.223770, 0.804450, 0.410629, 0.346449, 0.177124, 0.501283, 0.549309, 0.451341, 0.361022, 0.544906, 0.503888, 0.025857, 0.987975, 0.283384, 0.546554, 0.996850, 0.202075, 0.784233, 0.832221, 0.171048, 0.140255, 0.939380, 0.556670, 0.640718, 0.641941, 0.266179, 0.631766, 0.170432, 0.359606, 0.175010, 0.932614, 0.443964, 0.373564, 0.986104, 0.984349, 0.886544, 0.829643, 0.431721, 0.330424, 0.240905, 0.620278, 0.686223, 0.646810, 0.446531, 0.864148, 0.896199, 0.561920, 0.553542, 0.453893, 0.705572, 0.617390, 0.938558, 0.260130, 0.221184, 0.396521, 0.328814, 0.484480, 0.797830, 0.126500, 0.257792, 0.558319, 0.163570, 0.163512, 0.060019, 0.087500, 0.029992, 0.945579, 0.454695, 0.766922, 0.997560, 0.619472, 0.715045, 0.811151, 0.893633, 0.307746, 0.646302, 0.388884, 0.826579, 0.395822, 0.515458, 0.427911, 0.109334, 0.520129, 0.590483, 0.086815, 0.482671, 0.061591, 0.653812, 0.151846, 0.487604, 0.134303, 0.396799, 0.742545, 0.665987, 0.921097, 0.866750, 0.640117, 0.535664, 0.676645, 0.413427, 0.934478, 0.811603, 0.360637, 0.856182, 0.019257, 0.054792, 0.138725, 0.521203, 0.454742, 0.390027, 0.398131, 0.330682, 0.966053, 0.389931, 0.863868, 0.440635, 0.429397, 0.376011, 0.780176, 0.749131, 0.781932, 0.768958, 0.098594, 0.073995, 0.424335, 0.894389, 0.794658, 0.631189, 0.180617, 0.087077, 0.988302, 0.217732, 0.107889, 0.485652, 0.756510, 0.281508, 0.083874, 0.303661, 0.475573, 0.372313, 0.246687, 0.497903, 0.515367, 0.430002, 0.620055, 0.590905, 0.097698, 0.941919, 0.257283, 0.523780, 0.337584, 0.583186, 0.100606, 0.396007, 0.142027, 0.684096, 0.429356, 0.516558, 0.577394, 0.355074, 0.045051, 0.802091, 0.766831, 0.125655, 0.182228, 0.894448, 0.413901, 0.731051, 0.974802, 0.046192, 0.362459, 0.937135, 0.784423, 0.694805, 0.657531, 0.491806, 0.695390, 0.459380, 0.908052, 0.655914, 0.297555, 0.264873, 0.607866, 0.740032, 0.294066, 0.272939, 0.168251, 0.402388, 0.124873, 0.702702, 0.440036, 0.997003, 0.723173, 0.989145, 0.336699, 0.308915, 0.099095, 0.137547, 0.492345, 0.137763, 0.651350, 0.195477, 0.788113, 0.829533, 0.882838, 0.834369, 0.950915, 0.071037, 0.720165, 0.050340, 0.108017, 0.451946, 0.424858, 0.068357, 0.741254, 0.234827, 0.237373, 0.037235, 0.196249, 0.982835, 0.024434, 0.153590, 0.257614, 0.224171, 0.347438, 0.066946, 0.662382, 0.726104, 0.489764, 0.390005, 0.694743, 0.836723, 0.231238, 0.720166, 0.780296, 0.849085, 0.913712, 0.609630, 0.722349, 0.887739, 0.346895, 0.228688, 0.516997, 0.839697, 0.119207, 0.436327, 0.104813, 0.734958, 0.530872, 0.673295, 0.317480, 0.402184, 0.290185, 0.953457, 0.751946, 0.652451, 0.660617, 0.939398, 0.244165, 0.782872, 0.193245, 0.927356, 0.972734, 0.138602, 0.403491, 0.721753, 0.668512, 0.372534, 0.558285, 0.574737, 0.400080, 0.064634, 0.516990, 0.834189, 0.143156, 0.532624, 0.495067, 0.173853, 0.127888, 0.970599, 0.091499, 0.429564, 0.316429, 0.620672, 0.317521, 0.540884, 0.228669, 0.604991, 0.383869, 0.018178, 0.295507, 0.693788, 0.895892, 0.917494, 0.327755, 0.588209, 0.122021, 0.877799, 0.133504, 0.593185, 0.598868, 0.326042, 0.831871, 0.436185, 0.556695, 0.015645, 0.559371, 0.553887, 0.706407, 0.026107, 0.549540, 0.866930, 0.405315, 0.451739, 0.217563, 0.154370, 0.653690, 0.679734, 0.064187, 0.387245, 0.627347, 0.683839, 0.680178, 0.009802, 0.099090, 0.713574, 0.837803, 0.366157, 0.268439, 0.582433, 0.021556, 0.872553, 0.148877, 0.456425, 0.134338, 0.826630, 0.156495, 0.863711, 0.004580, 0.680066, 0.243573, 0.954678, 0.485229, 0.086235, 0.104846, 0.609857, 0.251042, 0.381345, 0.956936, 0.036563, 0.767330, 0.142187, 0.021650, 0.783736, 0.527847, 0.843213, 0.044166, 0.618337, 0.739072, 0.806760, 0.257846, 0.070684, 0.559841, 0.933502, 0.899713, 0.713796, 0.060467, 0.394535, 0.562056, 0.078069, 0.766682, 0.367190, 0.785070, 0.430597};
    float B_real[N * K]{0.403881, -0.883349, 0.262251, -0.370262, 0.309746, 0.974988, -0.854353, 0.093187, -2.228510, 2.246065, -0.364190, -0.580010, -0.161949, -0.983348, 1.634499, -0.674018, 0.288342, 0.141571, -1.754302, 0.576244, 1.192469, 0.082538, -0.897980, -0.221145, -0.541987, 0.289719, 0.357319, 0.935251, -0.151522, 0.739281, -0.826189, -0.561367, -0.488902, 0.384829, -0.623494, -1.095178, 1.391855, 1.928134, -0.363809, 1.305784, -1.684759, -0.603875, -2.157303, -0.535213, 0.122283, 1.165408, 2.748467, 0.663518, 1.162744, -2.067645, 0.276011, -1.555896, -0.222782, 0.325698, -1.350100, -0.267617, -1.345502, 0.670638, -0.627094, -0.729306, 0.413149, -0.368728, 0.093917, -0.261972, 0.962279, -0.908706, -1.512977, -0.350233, -0.181936, -1.154551, 0.459797, 1.100208, 0.272066, 1.296310, -1.162241, 0.186550, 0.000747, -0.110111, 0.440150, -0.864575, 0.501745, -0.838170, -0.951181, -1.095415, -0.253975, 0.240575, 0.433979, 1.619877, -0.233954, -0.107192, -0.217507, 0.175058, -1.168531, 1.099229, -0.944279, 0.950945, 0.053495, 0.370962, -1.502641, -0.094921, 0.083064, -0.282475, 1.172590, 0.549233, 1.891756, -0.654046, -0.229768, -0.050833, -1.046734, -1.712879, 0.796123, 1.003611, -1.224234, 0.653223, -0.671211, -0.790519, -2.341905, 1.096595, -0.208235, 1.383214, 0.157798, 3.569868, 1.735096, -2.174621, -1.220254, 1.274955, -0.827086, -0.812697, 1.583319, -0.459895, -1.518023, 1.511008, -2.099479, -0.505073, 0.576681, -0.489478, 1.248095, -0.398163, -1.505125, 1.303681, -0.527943, 3.407532, -0.371696, 0.144615, -0.376237, -0.551191, -0.831983, -0.438420, -0.249413, 1.091000, -1.074917, -1.137239, -0.390243, -0.475994, -2.085770, 2.974474, 2.809232, -0.085463, 1.809741, -0.125812, 0.723061, 1.146918, 1.192864, 0.669033, 0.052429, -0.087409, 0.497900, 0.858610, 1.294019, -0.163880, -3.072166, 0.643018, 0.664282, -2.051561, 0.235964, -0.622585, -0.232185, 2.373287, -0.116973, -0.596874, -0.849942, 0.787258, 0.953648, -0.848829, -1.953934, -0.347820, 2.315626, 0.195216, 0.361980, -0.067240, 0.521407, -0.012760, -0.702264, -0.448305, -0.778421, 1.920303, 0.287032, -0.473921, 1.226209, -1.520646, -0.796384, -1.278137, -1.415045, -0.057090, 0.565421, 0.993963, -0.793826, 0.888862, -0.263053, -1.001592, -0.991024, 0.914260, 0.501328, -1.551197, 1.099562, 0.961149, -0.464604, 0.946341, 0.494943, 0.671842, 0.725338, -0.584593, -0.033356, 0.291786, 0.077794, 0.428436, 0.540960, 0.069222, 1.082144, -0.555661, -0.253126, 1.107695, 0.540314, 0.929826, -0.855564, -0.557803, 0.384318, 0.818162, 0.908595, 0.022844, 1.686544, -0.619406, 0.261118, -0.614619, 0.080701, -1.004185, -0.559061, 2.486821, 0.982434, -1.351556, 1.009265, 0.820471, 0.990751, 0.901940, 0.007558, -0.106555, -0.379828, 1.588971, -1.154219, -1.177789, -0.386394, 0.535301, 0.522995, -0.013643, 0.794708, -0.632688, 1.976566, -1.665643, -0.111444, 0.364211, 0.051012, -0.817612, 0.989378, 0.138287, -0.937586, -0.215161, -0.101778, 0.526012, 1.257556, -0.334578, -0.505078, 0.482987, 0.676787, 0.412113, 1.043689, -1.045710, 0.544660, -0.415928, 1.295951, -0.452208, -0.440433, -0.126485, -0.688841, -0.378408, -0.681560, 0.473490, 1.618151, -2.465198, 1.452532, -1.829455, 0.408074, 0.108475, 0.674396, -0.937150, 0.802185, 0.505707, -0.137906, -0.084225, 1.781295, 0.783366, -0.848469, 0.264143, -0.856838, 0.143064, -0.260139, 1.365641, -0.847230, -0.852528, -2.076811, 0.429990, 1.072356, -0.340285, 0.678399, 0.835157, -0.468824, 0.331954, 0.619876, 0.089252, -0.065615, 1.237237, -0.240371, 3.158528, 0.048390, 1.605039, -0.228795, -1.637804, -0.575883, 0.511735, -0.176488, -0.173993, 0.967983, -0.925848, -0.312580, 1.094538, -0.046592, -1.635413, -0.005583, 1.456147, -0.255681, 1.077198, 0.602929, 1.226617, -0.664853, 1.349116, -0.524813, 2.023729, -0.075913, 0.257775, -0.652082, 1.631242, 0.270146, 0.005322, 0.379244, 0.321181, 0.353453, -1.906840, 1.107199, 0.219454, 0.404638, -0.034841, -1.516254, 2.320635, 1.452742, -0.448442, 1.128319, 0.777789, -0.132472, 1.962447, 0.156534, -1.359428, -0.634407, 1.139149, -2.252898, 0.015113, -1.256801, -0.040399, -0.185590, -0.114877, 0.351757, -0.503987, -0.068349, 0.414498, 1.379846, 0.168411, 0.550139, -0.548902, 1.439310, 1.406315, -0.850226, 0.970507, 0.629763, 0.427361, -1.168351, -1.164493, -1.036031, 0.697193, -1.121418, 0.068602, 0.807511, -1.224019, 0.782393, 0.211832, 0.095136, -1.120497, 1.855147, -0.126011, 2.031263, 0.496834, 0.279046, 0.143640, -0.079234, 0.181032, 1.201966, 0.494841, -0.426963, 0.168803, 0.246450, 0.751495, -0.255590, 0.118509, -1.420726, 0.613167, -0.427127, 0.400731, -0.277299, 0.299580, -0.978202, 0.082831, 0.597094, 0.082604, 1.376949, 0.657480, -0.660739, -1.215849, 0.077669, -1.543796, 1.561037, -0.689427, 0.715113, 0.959474, 0.669365, -0.527777, 0.510808, 0.739063, 1.066631, 0.296171, -0.641024, -1.548478, 0.244717, 0.716664, -1.408212, 0.584246, 0.327559, 1.590788, 1.790930, -1.551802, -1.196622, 0.450814, 1.048597, -0.340930, 0.683237, 1.241601, -0.656274, 0.900045, -2.099239, 1.200783, 0.520476, 1.863160, -0.118347, 1.193484, 0.141232, -1.616451, 1.508127, 0.526475, 0.797263, 0.867078, -0.242345, -1.565025, 0.177549, -0.173318, -0.882136, -0.157631, -0.125017, -1.531218, 0.638484, 1.090173, -0.380700, 0.134027, -1.289408, -1.071062, 1.289719, 0.018774, -0.888909, 0.264819, -0.442501, -0.145373, 1.004828, -0.078796, 0.153298, 0.616301, -1.504509, -1.373602, -0.530467, 0.504621, 0.371465, -0.358703, 1.992852, -1.546003, 0.057564, 1.318902, -0.494936, -0.425961, -0.959285, -0.393598, -0.629091, -0.386242, -1.920126, -0.941792, -1.254324, 0.863919, 0.433139, 0.870838, 0.105586, -0.864215, -0.374179, -0.129928, 1.635870, 0.433282, -0.520933, -1.210003, -0.624825, -2.034027, -0.492719, -2.176310, 1.533796, 1.316204, 0.625446, -0.652862, -1.172801, -1.416914, 0.808256, -1.568533, 1.128350, -0.376555, 0.695348, 0.733738, 0.559019, 0.102954, -1.279027, -1.074115, -0.933634, -1.328553, 0.441057, -1.195654, 2.733540, -0.796464, 0.752960, 0.282529, -1.461442, 0.199952, 0.578938, -1.844253, 0.742478, 0.788010, 0.877629, 0.120332, 0.563638, -0.570346, -0.056227, -0.672560, -0.278725, -0.319886, -0.204764, 0.365328, 0.167439, 0.135441, 0.213484, -1.125079, -1.244288, 0.195827, 0.762849, 0.288410, 1.143603, 0.298206, 1.033606, 1.136331, -0.338397, 0.493062, -0.209680, 0.736462, -0.200525, 0.825430, -0.977807, 2.412443, -0.478865, 0.417806, -0.770234, -0.988974, -0.155099, -0.151132, -1.182127, -0.950911, -0.914714, -0.163742, 0.419791, -0.686773, 0.763647, -0.707513, 0.192102, -1.069961, 0.136741, -0.229293, -1.572947, 0.740247, -1.412654, 0.819880, -0.007135, -1.515859, -1.614914, 0.892784, 0.583848, -0.910732, 0.179802, 0.606733, 0.601069, 0.471683, 1.127442, -1.348025, 0.992332, 0.334715};
    float B_imag[N * K]{0.411883, -0.808845, 1.772483, -1.586821, 0.920985, -0.847371, 0.564558, 0.437824, -0.056144, 0.836830, -0.813314, 1.859280, 0.549349, -0.197264, 0.161436, 1.021377, 0.275259, 1.306725, -0.466712, 0.760868, 0.154120, 1.161004, -2.508804, -1.019146, -0.906881, 1.337238, -0.801539, -0.143268, -0.841239, 0.474509, 0.793403, 0.927072, 0.467634, 0.649153, -1.061811, -0.139831, 0.353289, -0.284375, -1.435821, 0.237920, 0.554571, 0.592105, -0.456589, -1.385234, 0.144441, -1.616317, -1.852947, -0.341533, -0.154844, 1.252166, -0.381945, -1.226964, 0.191457, -0.831474, 0.450494, -0.081551, -1.200376, -0.791710, -0.977716, -0.208373, -1.172846, 0.242748, 2.430378, 0.954886, 1.396409, -0.491818, 0.053216, 2.057811, 1.188595, -0.892623, -1.371204, -0.327239, -0.229805, 0.895954, -0.272798, 1.258247, 0.166129, 0.240619, 0.605895, 0.024739, 1.007587, 0.240477, -0.471454, -0.601120, -0.189699, -0.969302, -0.436415, -0.374535, -0.414907, -0.895274, 0.010307, 0.891646, -0.579217, -1.813486, -0.101454, 1.468534, 0.776162, -0.094646, -0.113655, -0.028091, 0.113520, -0.821500, -0.560328, -1.171891, -0.843836, -0.474108, 0.280007, -0.190203, -0.658617, 0.313326, 0.204132, 0.288186, 0.480481, 1.566683, -1.429106, 0.032941, -1.381435, 0.793235, 0.764562, 0.140359, 0.730051, 0.993112, -1.226467, -0.577110, 1.943438, -0.666164, 0.084606, -0.938732, 0.983991, 0.667033, -0.411026, 2.265195, -0.386833, 0.846501, -0.764409, 1.692478, 1.128279, 0.959440, 0.669897, 0.237586, -0.983512, 0.346395, 0.792950, -0.836431, -0.246922, 0.478375, 0.729119, -0.813462, 0.101227, 0.828117, 0.663561, -0.047890, 0.421616, 0.110155, 0.410138, 0.475729, 2.465440, 1.087103, 0.375784, -0.671549, 0.052032, -0.261134, -2.109940, 0.852970, 0.441619, -1.041058, 0.900753, -0.583619, -0.362706, 0.006509, 0.225782, -1.551864, 1.087706, -1.161102, -0.789933, 0.368530, -1.557917, 0.647169, 0.766653, -1.045000, 0.877599, -0.184713, -0.799395, 0.477331, -0.803611, -0.820659, -1.325638, 1.709525, -0.738828, -0.235892, -0.212843, 0.444083, -2.249332, -0.397536, 0.161637, 2.192001, -2.066648, 0.632550, 1.523472, 0.965767, 1.014141, -0.101735, 0.566975, 0.302320, 0.287609, 0.101130, -1.524579, -0.390420, -1.479842, 0.652821, -0.044035, -0.911819, 1.804480, 0.254294, 1.977904, 1.535685, -0.072599, 1.903276, 0.576824, -0.219757, -0.084348, -0.887043, -0.001443, 0.415776, 0.467018, 1.083983, 0.502433, -0.690065, 0.153372, 1.964533, 0.458182, 0.049435, -0.632060, 1.207789, 0.795266, 0.907260, 0.759111, -1.160026, 0.425973, 1.414536, -1.853532, 0.741377, 0.623925, 0.042975, 0.004562, -1.854527, 0.306391, -0.448221, -0.317017, 0.884584, -0.441417, 1.078046, 1.316458, -1.033542, 1.037367, 0.402260, -0.081664, -0.347309, -0.042931, -0.924187, -1.096821, 1.392208, 0.126441, -0.948853, 0.326090, 1.799753, -0.939160, 0.714376, 0.318918, 0.084537, -1.054947, 0.308179, 1.551592, 1.295147, 2.360330, -0.538382, -0.158473, -1.695065, -1.242263, -0.594141, 0.218628, 2.473900, 0.680868, 0.541608, 1.204587, -0.878374, 1.237402, -1.315731, -1.456547, -0.574514, -0.155584, 0.299639, -1.468908, 2.768123, 1.175395, -0.115215, 0.006498, 0.919386, -0.504952, 1.421275, 0.794246, 0.503919, -1.287416, -0.821128, -1.404116, 0.647733, 0.033012, 0.627122, 1.321835, 0.499751, 0.129148, -0.197218, 0.176938, -0.495347, 0.397740, 1.822000, -1.064608, 0.021531, 0.325851, 1.407608, 0.463124, -0.822481, 0.218085, -1.071905, 0.205564, 1.049565, 0.123568, -1.331295, 0.487204, -0.484104, 0.509428, -0.146422, 3.466260, 0.468766, 1.951249, -2.021507, -1.643915, -0.299340, -1.111423, -1.029001, -0.612630, 0.200982, -1.566567, -1.074092, -1.991196, 1.323093, 0.273939, -0.130848, -1.566626, 0.238446, -0.030124, -0.103071, -0.214627, -0.657297, -0.904656, -0.636977, 1.581550, -1.721016, 0.468184, 0.206512, 2.244400, -1.007053, 0.783282, 0.869553, 0.028409, -0.517359, 0.754166, -1.826896, 1.057060, 0.778248, -0.457365, -2.798982, 0.486283, -1.716957, -0.682152, -0.188803, 0.292614, 0.183432, 0.322613, 1.341111, 0.072348, -0.613172, -0.310570, 0.981051, 1.930515, 0.024315, 1.524498, 0.535922, -0.587578, 0.924273, 0.596336, 0.393275, 0.330885, 1.470519, 2.018292, 0.148275, -1.187609, -0.259732, 0.100021, 1.332717, 0.865514, -0.658963, 0.655324, -1.758825, 0.941137, -0.162367, -0.261976, 0.692865, 0.302222, 0.588138, -0.113528, 0.990247, 1.267902, 0.694137, 0.454343, -1.783880, 1.075937, -1.354789, 0.301435, -1.284895, -0.415696, -0.833229, -0.537467, -0.148096, -0.006924, -0.605661, -1.730684, -0.687550, 0.219462, 1.377934, 0.807024, -1.297633, 1.090514, -0.510697, -0.042104, 0.529068, -1.297372, -1.157442, 0.023824, 1.618376, -1.114938, 0.378179, 0.328643, 0.251942, -0.338242, 0.271710, 1.013464, 0.317576, -0.878201, 1.851205, -0.089803, -1.521968, -0.946451, 0.113387, 0.458167, 0.729732, -0.121770, -0.761986, -0.022055, 0.661644, 0.685252, -1.115343, 1.054072, -0.610385, 0.528266, 0.358643, -0.927736, 0.162842, -2.861400, -1.897665, -0.006292, 0.621011, -0.438499, -0.229807, -0.264144, -0.940135, 0.740772, -0.250300, -0.008758, 0.227346, 1.037673, 0.667241, -1.979699, 0.760980, 1.056900, -0.352315, 1.243680, 1.159116, 0.535555, -1.778684, -0.091861, -1.507523, 0.343221, -1.461733, -0.291985, -0.259287, -0.840718, -1.648763, 0.929455, -0.225576, 1.822212, 0.795333, -1.867388, 0.254430, 0.014741, 0.132360, -0.589489, 0.128388, -0.342037, -0.922641, -0.921238, -1.679402, -0.058403, -2.882336, -0.386798, -1.213545, -0.349471, -0.181991, -0.090432, -0.965993, -0.528988, 1.037492, -1.832356, -0.152834, 1.614761, 2.673582, 1.491482, 1.039458, -0.597859, -1.997927, -0.926952, 0.788976, 2.534975, -0.047468, -0.553993, 1.624550, 0.029632, 0.615687, -2.641146, 0.095018, -1.627967, -0.020431, 0.848592, -0.557411, -0.815855, -0.007566, -0.805978, -0.116771, 0.418913, -0.357100, -0.961221, -0.654282, 0.438645, -0.462466, -0.943728, 0.373512, 1.952373, -0.376674, -0.486150, -0.256733, 1.617302, 0.618953, 0.405187, 0.677018, -1.997759, 0.710303, -1.112599, -0.648297, -0.689681, -0.336401, 1.784830, 1.244903, 0.437518, -0.576646, -0.059848, -0.250892, 0.459024, -0.698035, 0.195970, 2.310131, 0.264137, 1.803064, -0.702456, 0.849297, -0.748191, -0.462539, 0.956940, -0.037981, 0.671191, 0.250426, -0.200189, -1.292264, -0.837676, -0.845968, -0.574685, -0.925456, -0.399312, 0.304122, 0.959722, 0.190060, -1.127146, 0.052993, 1.499046, 0.469307, -0.199304, -0.527878, -1.363052, -0.199325, 0.150897, 0.286162, 0.940412, -0.614352, -1.307503, -1.817236, 1.441855, 0.098660, -0.225385, 1.101080, 1.380324, -0.173404, -0.559176, -0.177890, 0.137795, -0.239782, -0.372760, -0.345787, 0.680585, 0.881670, -0.991319, -0.686169, 0.349156, 0.241695, 0.794143, -0.521731, 0.235527, 1.684280, -0.280283, -0.446608, -0.941464, -0.014029};
    constexpr float C_Matlab_real[M * K]{1.725343, -3.070041, 4.841147, 3.385986, 2.129013, -2.942496, -0.090210, 1.181185, 5.625918, -3.175873, -4.881923, 4.710938, -3.249266, 1.338714, -2.030641, -0.087490, 4.842548, 9.317006, -3.112684, -5.495728, 0.692453, -5.928462, 1.514125, 3.232953, -0.344580, -1.338534, 2.293735, 2.436943, 0.757613, -2.952683, -2.314841, 7.264267, -6.888259, 2.533568, -3.767397, -5.791336, 5.801674, 10.532570, -5.297222, -2.509722, 3.064064, -2.973053, 0.026715, 4.011523, -1.022232, -0.326096, 4.379109, -0.619032, 4.914609, -2.866331, -3.089820, 4.318562, -3.249038, 5.859066, -1.969624, -2.586792, 4.953597, 2.993748, 0.130277, -4.158931, 1.108420, -5.700291, 1.797230, 2.264147, 0.621401, -1.884641, 3.312388, -0.501914, 2.307743, 1.433767, -2.188582, 3.562575, -7.808491, 3.283104, -2.461411, 0.315015, 1.557612, 5.769051, -2.950955, -2.301161, -0.212379, -1.034444, -2.092533, 3.133317, -2.175459, -1.775808, 2.148525, 1.567247, 1.581348, -2.744428, -2.778487, 4.195642, -4.987796, -3.071138, -6.414315, 1.819329, 4.473161, 6.630657, -1.618710, -4.116945, -2.930794, -2.721797, -2.374219, 1.026745, -0.387453, -2.058348, 5.929617, 0.443695, 3.160957, -5.738529, -2.881520, 9.045552, -5.624848, 1.691924, -5.515030, 3.167933, 5.665378, 4.486824, -2.130590, -4.897702, 4.502970, -3.157281, 1.462013, 2.972112, -0.832812, -6.770843, 1.733181, -0.008624, 5.061499, -1.548339, -5.171155, 3.381244, -5.491816, -1.111783, -0.780354, -2.258374, 3.654226, 5.019430, -3.253652, -4.294156, -0.468854, -3.597103, 3.292023, 3.656516, 2.543710, -2.310194, 2.324560, 0.783744, 2.296040, -1.225123, -4.203248, 6.330481, -6.200803, 4.054802, -0.364472, -3.045575, 5.076285, 1.266206, -4.782442, -6.348528, -0.465254, -2.560115, 1.591120, 3.445258, -0.480630, -0.027959, 3.864660, 1.248620, 1.074492, -3.451792, -0.460044, 5.002251, -4.919296, 3.334818, -3.964623, -0.362331, 8.205985, 5.855267, -2.371862, -5.414925, 1.110319, 1.092188, 0.235493, -0.689014, -2.484786, -0.709883, 2.665848, -1.075142, 6.435032, -2.672241, -1.073252, 5.669599, -7.911855, 3.970428, -6.592313, -0.000594, 5.376729, 2.593008, -3.376438, -1.843007, -1.466746, -2.547876, 3.684971, 2.722812, 2.771659, -0.935027, 3.457750, -0.940521, 6.080547, -4.379386, -3.978113, 3.828025, -4.445227, 4.765139, -4.291907, 0.713978, 5.342226, 5.089212, 2.790723, -5.557862, 1.844431, -1.897249, 1.068326, 0.206674, 2.400525, -2.353480, 4.399715, 0.122522, 5.049309, -1.966418, -3.883892, 6.782477, -8.660614, 3.457956, -4.475590, -0.538153, 8.954199, 7.588216, -2.213394, -0.360189, -1.622379, -2.710414, -0.661421, 3.205892, 4.311216, -0.901347, -2.195862, -1.018784, 2.299173, -2.223482, -2.813096, 3.786922, -4.873249, 4.496156, -2.479142, -0.474971, 8.475364, 6.207793, -4.647673, -3.484856, -0.999757, -1.906733, 3.175222, -0.337406, -0.261623, -1.238616, 1.245987, 2.702439, 7.910771, -4.771172, -4.800002, 5.720996, -4.971623, 2.218397, -4.161130, -0.808310, 8.333129, 8.097564, -3.542341, -7.278308, -1.813540, -3.296999, -1.375659, -0.312934, 0.865819, -0.051038, 0.347365, -1.998929, 1.680405, -0.530776, -3.636014, 1.137554, -4.263155, 0.586243, -3.236909, 2.923467, 11.077492, 8.285983, -0.967929, -6.379468, -3.008385, -7.244409, 1.894750, 0.565125, 3.889033, -2.387096, 1.877393, -1.293076, 5.084761, -4.066620, -0.416584, -0.635619, -1.927716, 1.569925, -2.662616, -3.800123, 5.438930, 2.095725, -0.371279, -5.673856, 0.559543, 0.150394, 1.552793, 2.750509, -1.456267, 1.863766, 1.549044, -2.366998, 3.092582, -3.728560, -2.601430, 3.761530, -8.539107, 3.134897, -2.615493, -0.202233, 6.412955, 4.602421, 0.064648, -0.537642, -1.179543, -1.949474, 1.635451, -1.184322, 2.975744, -2.032183, 4.066701, 1.686167, 2.659038, -3.944737, -3.719932, 4.429415, -5.737844, 5.749256, 1.365755, -1.107849, 4.612300, 6.523096, -2.988190, -6.086818, -0.243705, -1.410625, 3.410912, 2.525687, 1.321252, -3.115435, 3.315306, 1.437234, 5.890870, -4.042526, -3.756174, 7.113839, -4.975492, 3.062731, -4.060520, -1.325855, 4.594176, 7.444722, -4.475095, -3.906745, 0.552346, -2.830545, 0.011146, 2.983315, 0.636406, 0.160185, -2.079248, 1.307073, 5.247842, -0.347653, -3.518265, 4.973145, -8.048869, 3.111054, -3.840385, 5.767413, 11.532631, 8.613050, -3.540017, -0.236295, 3.392330, -1.391840, -0.492351, 4.214426, -0.208085, -4.888734, 1.631574, 0.127238, -0.747772, -3.000384, -2.224096, 5.023028, -6.507735, 3.872024, -5.026761, -1.964268, 7.183703, 4.493723, -1.209684, -2.284357, 0.490332, -8.716968, 3.760465, 2.033027, 2.097394, -4.317233, 2.687036, -0.123917, 3.734491, -2.014552, -3.121657, 6.982234, -2.123223, 4.551475, -5.776593, 0.629090, 4.781216, 7.013049, -1.528300, -2.386309, 0.727914, 1.413161, 1.927753, -0.588081, 2.107285, 0.059812, 2.591524, 2.120417, 4.880289, -6.032528, -4.955882, 6.490793, -5.737371, 2.820748, -2.803482, 0.122543, 6.319094, 5.270431, -1.439677, -4.062969, 2.209441, 2.085912, 0.197835, 2.636766, -1.576466, -3.903946, 1.631459, 3.769650, 2.727646, -1.746627, -5.094441, 4.670144, -8.050918, 4.630468, -3.932751, 2.184680, 7.224099, 9.154377, -2.204642, 1.049240};
    constexpr float C_Matlab_imag[M * K]{4.546255, 3.955828, -0.106507, -5.193781, 8.243894, -5.857795, 2.554178, -0.188127, -0.045024, -0.898069, -5.250295, 1.142323, 7.382656, -4.037973, 2.675969, 6.888286, 4.636273, 3.022852, -2.510386, -2.300655, 6.760216, 2.169187, -0.492695, -7.260972, 3.551619, 0.218304, 1.499193, -2.203723, -0.878449, -0.450204, -6.705580, 5.035710, 8.028537, -1.908335, 2.424070, 2.741559, 0.869498, 4.586564, -3.795766, -1.846138, 2.662902, 4.533958, -6.665818, -5.204789, 4.431529, 1.401128, 2.406291, 3.598736, -3.153089, -1.837435, -3.913533, -0.243841, 8.362064, 1.243157, 3.557800, 7.746222, 0.370911, 3.243357, -6.750835, 2.094842, 4.176066, 1.712827, -6.388790, -4.919434, 3.174598, -1.246809, 0.840654, 0.964891, 0.488199, -0.973372, -3.270754, 1.601957, 5.759644, -3.820711, -2.728065, 8.271609, 3.481760, 2.397388, -0.358189, 0.511972, 0.150484, 10.183362, -4.359678, -4.590628, 2.535245, -1.823438, 1.316872, -3.010417, -0.927434, -2.245900, -3.324017, 3.809166, 7.247316, -0.421451, 0.734856, 7.846113, 2.878157, 5.819727, -1.108239, 1.056800, 4.760719, 5.289271, -6.462887, -2.380014, 4.148947, -0.582256, 5.740253, 0.950844, -3.923906, -0.046435, -8.623131, -1.328851, 4.451294, -2.949833, 3.310627, 9.567548, 2.570350, -0.673990, 3.057147, 0.665303, 4.739591, 5.136585, -8.891981, -2.556341, 2.278495, -0.618778, -0.817917, -0.623929, -3.893927, -0.116439, -1.613162, 0.286169, 7.300495, -4.840649, 0.338765, 7.242052, 2.762276, 3.433016, -2.268483, 6.927586, 4.543047, 3.570695, -1.401792, -3.861560, -0.167846, -0.297508, -0.557827, -3.233421, -2.039252, 1.425932, -4.736522, 4.138396, 3.414306, -4.653190, 4.355020, 8.086162, -0.066641, 4.992288, -3.360022, 4.878926, 0.794993, 2.882307, -0.794162, -6.653434, 2.478059, -1.012274, 2.695738, -2.151819, -5.504435, 2.876867, -4.375269, 4.364145, 4.082696, -1.319264, 1.565130, 11.141020, 4.074761, 0.219035, -3.101120, 2.229454, 2.940812, 2.340307, -3.747347, -5.551177, 4.441437, -5.597196, 2.610508, 2.558808, -2.186594, 3.195455, -5.882977, 3.576526, 3.177690, -1.744151, 3.117521, 14.938452, 3.681867, 3.607587, 2.096567, 2.886682, 5.727488, 2.034703, -3.824099, -2.991192, -0.497167, -1.423462, -0.128212, -1.506816, -3.993981, 3.232896, -2.972912, -3.640008, 5.235793, -2.444921, 6.664027, 5.810013, 4.735773, 1.767949, -0.314175, 2.252966, 5.767947, 2.791545, -6.709605, -4.742052, 3.916986, -2.360974, 4.075042, -3.800966, -4.390232, 1.256798, -3.329919, 3.060207, 7.599659, 0.097362, 3.565496, 6.617229, 3.919846, 2.555646, 0.144300, 5.957457, 6.927675, 4.662179, -2.585729, -6.001112, 3.740694, -0.283679, 4.343483, -1.260878, -5.224376, 0.680586, -2.556749, -0.381701, 4.186053, -3.056417, 2.775990, 8.185751, 7.403339, -0.220153, -0.650766, 5.054359, 5.443241, 3.331662, -4.252190, -3.244057, 6.883638, -4.556931, 6.170059, -2.030312, -0.035689, -1.274383, -3.743715, 3.331812, 6.056939, -1.537907, 3.428417, 8.329747, 5.208080, 7.632849, -1.749460, 2.811580, 5.421788, 5.078254, -3.372948, -2.723731, 4.447223, -3.074020, 0.362774, 2.008222, -0.665824, 0.029844, -3.850226, 3.107416, 4.805104, -1.118247, 0.038985, 9.415476, 4.994800, 3.448342, -4.358851, 3.776377, 1.331193, 4.908063, 3.171324, -2.428108, 4.502794, -3.871070, 0.379810, 1.759195, -2.742802, -0.284583, -5.834544, 1.461477, 0.871866, 0.979096, 0.949125, 9.041734, 5.701900, 1.290052, -0.998005, 0.296701, 4.645417, 5.759304, -4.188174, -6.459362, 1.548762, -1.867281, -1.190994, -0.642140, -2.510065, 2.415462, -4.563711, 0.995555, 6.521453, -1.348224, 1.757329, 8.746908, 3.325054, 7.530377, -3.148000, -0.201597, 7.657966, 3.738856, -0.797494, -4.397359, 6.649451, -6.137268, 0.270652, 1.662709, -1.978749, 4.120427, -6.512446, 3.191167, 2.289341, -0.512054, 1.805143, 10.179021, 0.984443, 4.365908, -2.998621, 3.368771, 4.466451, 3.007323, -1.830011, -3.513340, 0.554348, -3.359269, 0.979034, -3.272708, -2.205098, 2.248072, -5.245187, 2.820491, 4.837623, -4.038139, 0.755202, 5.756732, 3.865610, 5.532012, 0.204600, 1.521043, 4.290835, 1.317499, -4.267999, -4.850701, 5.791151, -2.731943, 2.338436, -0.732724, -8.221112, 2.003023, -2.047087, 3.839234, 8.214162, -0.420570, 6.720400, 8.463292, 3.694948, 2.094783, -5.042537, 5.083814, 1.216770, 2.464627, -1.769359, -3.901256, 0.757106, -0.985188, 3.632967, -2.425901, -0.491832, 1.731895, -2.616129, 3.281452, 6.297959, -3.764078, 0.134771, 8.280124, 4.496537, 5.509523, 0.660351, 2.830139, 4.063547, 3.896696, -3.274622, -2.519308, 2.075157, -1.624200, -0.589119, -0.951015, -2.748491, 2.009874, -4.904922, -1.733914, 3.961345, -0.853998, 1.905962, 6.913365, 3.205243, -3.023951, -0.942284, 2.754812, 2.865476, 4.625718, -3.267286, -5.454763, 2.463499, -0.488756, 0.964475, -1.567351, -3.077367, 6.575524, -5.915706, 1.042379, 5.927831, 3.939695, 2.672795, 10.016160, 6.239850, 4.462352, -2.033878, 6.584848, 7.347995, 2.071678, -0.829518, -4.348825, 3.087595, -3.558415, -0.160151, -2.704361, -4.426807, 6.711827, -3.389338, 6.455933, 2.473648, 1.289732, 1.363989, 8.278472, 4.644720, 2.639358, -1.249509, 7.312321};
    float C_real[M * K]{0};
    float C_imag[M * K]{0};
    float C_MatrixUtil_real[M * K]{0};
    float C_MatrixUtil_imag[M * K]{0};

    // double A_real[M * N] {0.814724,0.678735,0.709365,0.814285,0.568824,0.106653,0.401808,0.575209,0.486792,0.225922,0.085516,0.098712,0.805489,0.972975,0.372410,0.032601,0.824376,0.068806,0.637709,0.322472,0.647618,0.318074,0.192028,0.683416,0.768854,0.850713,0.083483,0.123084,0.105709,0.467068,0.178117,0.441722,0.905792,0.757740,0.754687,0.243525,0.469391,0.961898,0.075967,0.059780,0.435859,0.170708,0.262482,0.261871,0.576722,0.648991,0.198118,0.561200,0.982663,0.319600,0.957694,0.784739,0.679017,0.119215,0.138874,0.704047,0.167254,0.560560,0.625960,0.205494,0.142041,0.648198,0.359635,0.013283,0.126987,0.743132,0.276025,0.929264,0.011902,0.004634,0.239916,0.234780,0.446784,0.227664,0.801015,0.335357,0.182922,0.800331,0.489688,0.881867,0.730249,0.530864,0.240707,0.471357,0.635787,0.939829,0.696266,0.442305,0.861980,0.929609,0.660945,0.146515,0.166460,0.025228,0.056705,0.897191,0.913376,0.392227,0.679703,0.349984,0.337123,0.774910,0.123319,0.353159,0.306349,0.435699,0.029220,0.679728,0.239932,0.453798,0.339493,0.669175,0.343877,0.654446,0.676122,0.035763,0.945174,0.645552,0.093820,0.019578,0.989872,0.696667,0.729752,0.189072,0.620959,0.842207,0.521886,0.196658,0.632359,0.655478,0.655098,0.196595,0.162182,0.817303,0.183908,0.821194,0.508509,0.311102,0.928854,0.136553,0.886512,0.432392,0.951630,0.190433,0.584069,0.407619,0.289065,0.175874,0.208935,0.479463,0.525404,0.330858,0.514423,0.582791,0.890752,0.042652,0.573710,0.559033,0.335849,0.093371,0.097540,0.171187,0.162612,0.251084,0.794285,0.868695,0.239953,0.015403,0.510772,0.923380,0.730331,0.721227,0.028674,0.825314,0.920332,0.368917,0.107769,0.819981,0.671808,0.721758,0.709282,0.639317,0.530344,0.424309,0.884281,0.815397,0.982303,0.635198,0.052078,0.854100,0.175669,0.307367,0.278498,0.706046,0.118998,0.616045,0.311215,0.084436,0.417267,0.043024,0.817628,0.430207,0.488609,0.106762,0.489901,0.083470,0.052677,0.460726,0.906308,0.718359,0.695140,0.473486,0.236231,0.544716,0.861140,0.270270,0.588026,0.879014,0.769029,0.281867,0.931201,0.347879,0.208947,0.456058,0.546882,0.031833,0.498364,0.473289,0.528533,0.399783,0.049654,0.168990,0.794831,0.184816,0.578525,0.653757,0.167927,0.133171,0.737858,0.981638,0.879654,0.968649,0.067993,0.152721,0.119396,0.647311,0.484853,0.197054,0.154752,0.988912,0.581446,0.538597,0.728662,0.446027,0.905154,0.101669,0.957507,0.276923,0.959744,0.351660,0.165649,0.259870,0.902716,0.649115,0.644318,0.904881,0.237284,0.494174,0.978681,0.173389,0.269119,0.156405,0.817761,0.531334,0.254790,0.341125,0.607304,0.543886,0.393456,0.821721,0.199863,0.000522,0.928313,0.695163,0.737842,0.054239,0.675391,0.995390,0.964889,0.046171,0.340386,0.830829,0.601982,0.800068,0.944787,0.731722,0.378609,0.979748,0.458849,0.779052,0.712694,0.390938,0.422836,0.855523,0.260728,0.325146,0.224040,0.607389,0.450138,0.721047,0.671431,0.429921,0.406955,0.865439,0.580090,0.499116,0.063405,0.177108,0.468468,0.332093,0.157613,0.097132,0.585268,0.585264,0.262971,0.431414,0.490864,0.647746,0.811580,0.438870,0.963089,0.715037,0.500472,0.831380,0.547871,0.644765,0.594356,0.105629,0.667833,0.191745,0.458725,0.522495,0.741258,0.887771,0.748706,0.612566,0.016983,0.535801,0.860441,0.662808,0.912132,0.297347,0.970593,0.823458,0.223812,0.549724,0.654079,0.910648,0.489253,0.450924,0.532826,0.111119,0.546806,0.903721,0.471088,0.803364,0.942737,0.376272,0.022513,0.610959,0.844392,0.738427,0.661945,0.993705,0.520052,0.391183,0.825584,0.989950,0.120860,0.445183,0.934405,0.330829,0.104012,0.062045,0.957167,0.694829,0.751267,0.917194,0.689215,0.181847,0.337719,0.547009,0.350727,0.258065,0.521136,0.890923,0.059619,0.060471,0.417744,0.190924,0.425259,0.778802,0.344462,0.242850,0.770286,0.218677,0.347713,0.769114,0.789963,0.527680,0.862711,0.123932,0.984398,0.898486,0.745546,0.298244,0.485376,0.317099,0.255095,0.285839,0.748152,0.263803,0.900054,0.296321,0.939002,0.408720,0.231594,0.334163,0.681972,0.399258,0.983052,0.428253,0.312719,0.423453,0.780520,0.917424,0.350218,0.105798,0.149997,0.396792,0.318524,0.479523,0.484297,0.490357,0.858939,0.118155,0.736267,0.046351,0.800280,0.950222,0.505957,0.757200,0.450542,0.145539,0.369247,0.744693,0.875943,0.594896,0.488898,0.698746,0.042431,0.526876,0.301455,0.482022,0.161485,0.090823,0.675332,0.269062,0.662010,0.109697,0.586092,0.808514,0.534064,0.801348,0.844856,0.852998,0.785559,0.988418,0.561861,0.505428,0.141886,0.034446,0.699077,0.753729,0.083821,0.136069,0.111203,0.188955,0.550156,0.262212,0.624060,0.197810,0.071445,0.416799,0.701099,0.120612,0.178766,0.266471,0.006715,0.765500,0.416159,0.063591,0.262145,0.755077,0.089951,0.227843,0.209405,0.873927,0.513377,0.539982,0.184194,0.761426,0.421761,0.438744,0.890903,0.380446,0.228977,0.869292,0.780252,0.686775,0.622475,0.602843,0.679136,0.030541,0.521650,0.656860,0.666339,0.589507,0.422886,0.153657,0.602170,0.188662,0.841929,0.404580,0.044454,0.377396,0.111706,0.498094,0.552291,0.270294,0.177602,0.706917,0.597211,0.631070,0.915736,0.381558,0.959291,0.567822,0.913337,0.579705,0.389739,0.183511,0.587045,0.711216,0.395515,0.744074,0.096730,0.627973,0.539126,0.226188,0.094229,0.281005,0.386771,0.287498,0.832917,0.448373,0.754933,0.216019,0.136293,0.900852,0.629883,0.208461,0.398589,0.999492,0.299937,0.089892,0.792207,0.765517,0.547216,0.075854,0.152378,0.549860,0.241691,0.368485,0.207742,0.221747,0.367437,0.500022,0.818149,0.291984,0.698106,0.384619,0.598524,0.440085,0.915991,0.091113,0.256441,0.365816,0.242785,0.790407,0.678652,0.574661,0.031991,0.564980,0.133931,0.287849,0.134123,0.080862,0.959492,0.795200,0.138624,0.053950,0.825817,0.144955,0.403912,0.625619,0.301246,0.117418,0.987982,0.479922,0.817547,0.431651,0.666528,0.582986,0.470924,0.527143,0.001151,0.576209,0.613461,0.763505,0.442402,0.949304,0.495177,0.845178,0.614713,0.640312,0.030890,0.414523,0.212602,0.777241,0.655741,0.186873,0.149294,0.530798,0.538342,0.853031,0.096455,0.780227,0.470923,0.296676,0.037739,0.904722,0.722440,0.015487,0.178132,0.251806,0.695949,0.457424,0.462449,0.683363,0.582249,0.627896,0.687796,0.327565,0.189710,0.738640,0.362411,0.417029,0.939142,0.464840,0.894942,0.905135,0.035712,0.489764,0.257508,0.779167,0.996135,0.622055,0.131973,0.081126,0.230488,0.318778,0.885168,0.609867,0.149865,0.984064,0.128014,0.290441,0.699888,0.875372,0.424349,0.546593,0.540739,0.771980,0.359228,0.671264,0.495006,0.585987,0.049533,0.205976,0.301306,0.763957,0.071453,0.533772,0.849129,0.445586,0.840717,0.934011,0.078176,0.350952,0.942051,0.929386,0.844309,0.424167,0.913287,0.617666,0.659605,0.167168,0.999080,0.617091,0.638531,0.518052,0.460916,0.425729,0.869941,0.932854,0.736340,0.438645,0.147608,0.246735,0.489570,0.947933,0.295534,0.818204,0.242487,0.109154,0.933993,0.646313,0.254282,0.129906,0.442678,0.513250,0.956135,0.775713,0.194764,0.507858,0.796184,0.859442,0.518595,0.106216,0.171121,0.265281,0.033604,0.943623,0.770160,0.644443,0.264779,0.972741,0.394707,0.833501,0.054974,0.666416,0.192510,0.082071,0.332936,0.100222,0.053754,0.825809};
    // double A_imag[M * N] {0.338098,0.890476,0.366437,0.112284,0.059403,0.892922,0.161134,0.935731,0.809204,0.671202,0.025135,0.910570,0.534138,0.411594,0.922332,0.557295,0.343288,0.954174,0.503781,0.331665,0.922745,0.300819,0.668464,0.450394,0.884405,0.084247,0.613475,0.694803,0.669043,0.848709,0.239291,0.074090,0.293973,0.798960,0.369199,0.784428,0.315811,0.703223,0.758112,0.457886,0.748619,0.715213,0.421112,0.800559,0.885359,0.602638,0.770954,0.772495,0.936027,0.031923,0.489594,0.152234,0.800372,0.939410,0.206776,0.205672,0.720856,0.163898,0.818641,0.426456,0.500211,0.916821,0.578923,0.393883,0.746313,0.734341,0.685028,0.291570,0.772722,0.555738,0.871111,0.240478,0.120187,0.642061,0.184100,0.745847,0.899005,0.750520,0.042660,0.311940,0.124774,0.356869,0.877049,0.348008,0.285947,0.980904,0.653851,0.899651,0.018613,0.324220,0.886235,0.836270,0.217994,0.986968,0.866887,0.003394,0.010337,0.051332,0.597942,0.603533,0.696433,0.184434,0.350777,0.763898,0.525045,0.419048,0.725775,0.813113,0.625938,0.583533,0.378186,0.178982,0.730585,0.662654,0.353142,0.121658,0.543663,0.286620,0.072052,0.762586,0.674776,0.301727,0.931112,0.731387,0.571616,0.505133,0.406777,0.220677,0.048447,0.072885,0.789364,0.964423,0.125332,0.212031,0.685536,0.759327,0.325834,0.390762,0.370363,0.383306,0.137869,0.551793,0.704340,0.338956,0.646477,0.281502,0.449444,0.884153,0.984776,0.800820,0.406727,0.882486,0.438509,0.011681,0.190785,0.360031,0.122189,0.271422,0.112615,0.001301,0.667916,0.088527,0.367653,0.432485,0.130151,0.077347,0.294149,0.740648,0.546449,0.816140,0.841560,0.617279,0.217802,0.583571,0.729513,0.210146,0.833152,0.230383,0.963530,0.094278,0.715678,0.896111,0.666932,0.284950,0.437820,0.539905,0.258582,0.454212,0.671166,0.100751,0.443846,0.189180,0.603468,0.798351,0.206028,0.694752,0.092352,0.913800,0.530629,0.743688,0.398881,0.317428,0.734230,0.575495,0.182141,0.511820,0.224277,0.510153,0.398282,0.711129,0.042298,0.930041,0.838970,0.597527,0.933726,0.673226,0.117037,0.095373,0.897866,0.386390,0.599586,0.507849,0.300184,0.142484,0.526102,0.943008,0.086667,0.758099,0.007820,0.706715,0.832423,0.105920,0.415093,0.814540,0.571026,0.530052,0.041820,0.082593,0.269055,0.906364,0.749822,0.624573,0.972958,0.399020,0.433261,0.884017,0.810950,0.664280,0.814682,0.146515,0.593362,0.775555,0.055976,0.585609,0.401387,0.268076,0.729709,0.683716,0.771934,0.432642,0.423109,0.557789,0.597490,0.681560,0.180738,0.789074,0.176855,0.275070,0.106942,0.719570,0.673031,0.628924,0.835221,0.590609,0.189207,0.047401,0.470625,0.943732,0.484548,0.122815,0.324855,0.631141,0.503840,0.734271,0.056343,0.762887,0.833364,0.174892,0.707253,0.132083,0.205675,0.655498,0.655573,0.313429,0.335311,0.463261,0.255387,0.852264,0.957384,0.248629,0.616443,0.996156,0.477492,0.101534,0.322460,0.660438,0.667120,0.342374,0.560713,0.549158,0.756749,0.407318,0.246228,0.859320,0.612810,0.430278,0.152501,0.082963,0.403629,0.138649,0.781377,0.722725,0.388272,0.109755,0.722923,0.166204,0.299225,0.212163,0.020536,0.505637,0.265322,0.451639,0.939661,0.354534,0.623716,0.390855,0.552262,0.047555,0.586440,0.735966,0.269092,0.728387,0.417047,0.275287,0.342713,0.974222,0.819422,0.693753,0.019621,0.661596,0.390176,0.598886,0.287977,0.110353,0.551779,0.933760,0.531209,0.622497,0.452593,0.098519,0.923676,0.635661,0.924581,0.227713,0.354456,0.971259,0.236445,0.054617,0.979129,0.348785,0.675112,0.794682,0.749018,0.576758,0.971786,0.716670,0.375692,0.570838,0.531889,0.945213,0.435176,0.516979,0.360449,0.901058,0.692532,0.117493,0.228953,0.187461,0.108818,0.987935,0.422646,0.823574,0.653700,0.950894,0.223770,0.804450,0.410629,0.346449,0.177124,0.501283,0.549309,0.451341,0.361022,0.544906,0.503888,0.025857,0.987975,0.283384,0.546554,0.996850,0.202075,0.784233,0.832221,0.171048,0.140255,0.939380,0.556670,0.640718,0.641941,0.266179,0.631766,0.170432,0.359606,0.175010,0.932614,0.443964,0.373564,0.986104,0.984349,0.886544,0.829643,0.431721,0.330424,0.240905,0.620278,0.686223,0.646810,0.446531,0.864148,0.896199,0.561920,0.553542,0.453893,0.705572,0.617390,0.938558,0.260130,0.221184,0.396521,0.328814,0.484480,0.797830,0.126500,0.257792,0.558319,0.163570,0.163512,0.060019,0.087500,0.029992,0.945579,0.454695,0.766922,0.997560,0.619472,0.715045,0.811151,0.893633,0.307746,0.646302,0.388884,0.826579,0.395822,0.515458,0.427911,0.109334,0.520129,0.590483,0.086815,0.482671,0.061591,0.653812,0.151846,0.487604,0.134303,0.396799,0.742545,0.665987,0.921097,0.866750,0.640117,0.535664,0.676645,0.413427,0.934478,0.811603,0.360637,0.856182,0.019257,0.054792,0.138725,0.521203,0.454742,0.390027,0.398131,0.330682,0.966053,0.389931,0.863868,0.440635,0.429397,0.376011,0.780176,0.749131,0.781932,0.768958,0.098594,0.073995,0.424335,0.894389,0.794658,0.631189,0.180617,0.087077,0.988302,0.217732,0.107889,0.485652,0.756510,0.281508,0.083874,0.303661,0.475573,0.372313,0.246687,0.497903,0.515367,0.430002,0.620055,0.590905,0.097698,0.941919,0.257283,0.523780,0.337584,0.583186,0.100606,0.396007,0.142027,0.684096,0.429356,0.516558,0.577394,0.355074,0.045051,0.802091,0.766831,0.125655,0.182228,0.894448,0.413901,0.731051,0.974802,0.046192,0.362459,0.937135,0.784423,0.694805,0.657531,0.491806,0.695390,0.459380,0.908052,0.655914,0.297555,0.264873,0.607866,0.740032,0.294066,0.272939,0.168251,0.402388,0.124873,0.702702,0.440036,0.997003,0.723173,0.989145,0.336699,0.308915,0.099095,0.137547,0.492345,0.137763,0.651350,0.195477,0.788113,0.829533,0.882838,0.834369,0.950915,0.071037,0.720165,0.050340,0.108017,0.451946,0.424858,0.068357,0.741254,0.234827,0.237373,0.037235,0.196249,0.982835,0.024434,0.153590,0.257614,0.224171,0.347438,0.066946,0.662382,0.726104,0.489764,0.390005,0.694743,0.836723,0.231238,0.720166,0.780296,0.849085,0.913712,0.609630,0.722349,0.887739,0.346895,0.228688,0.516997,0.839697,0.119207,0.436327,0.104813,0.734958,0.530872,0.673295,0.317480,0.402184,0.290185,0.953457,0.751946,0.652451,0.660617,0.939398,0.244165,0.782872,0.193245,0.927356,0.972734,0.138602,0.403491,0.721753,0.668512,0.372534,0.558285,0.574737,0.400080,0.064634,0.516990,0.834189,0.143156,0.532624,0.495067,0.173853,0.127888,0.970599,0.091499,0.429564,0.316429,0.620672,0.317521,0.540884,0.228669,0.604991,0.383869,0.018178,0.295507,0.693788,0.895892,0.917494,0.327755,0.588209,0.122021,0.877799,0.133504,0.593185,0.598868,0.326042,0.831871,0.436185,0.556695,0.015645,0.559371,0.553887,0.706407,0.026107,0.549540,0.866930,0.405315,0.451739,0.217563,0.154370,0.653690,0.679734,0.064187,0.387245,0.627347,0.683839,0.680178,0.009802,0.099090,0.713574,0.837803,0.366157,0.268439,0.582433,0.021556,0.872553,0.148877,0.456425,0.134338,0.826630,0.156495,0.863711,0.004580,0.680066,0.243573,0.954678,0.485229,0.086235,0.104846,0.609857,0.251042,0.381345,0.956936,0.036563,0.767330,0.142187,0.021650,0.783736,0.527847,0.843213,0.044166,0.618337,0.739072,0.806760,0.257846,0.070684,0.559841,0.933502,0.899713,0.713796,0.060467,0.394535,0.562056,0.078069,0.766682,0.367190,0.785070,0.430597};
    // double B_real[N * K] {0.403881,-0.883349,0.262251,-0.370262,0.309746,0.974988,-0.854353,0.093187,-2.228510,2.246065,-0.364190,-0.580010,-0.161949,-0.983348,1.634499,-0.674018,0.288342,0.141571,-1.754302,0.576244,1.192469,0.082538,-0.897980,-0.221145,-0.541987,0.289719,0.357319,0.935251,-0.151522,0.739281,-0.826189,-0.561367,-0.488902,0.384829,-0.623494,-1.095178,1.391855,1.928134,-0.363809,1.305784,-1.684759,-0.603875,-2.157303,-0.535213,0.122283,1.165408,2.748467,0.663518,1.162744,-2.067645,0.276011,-1.555896,-0.222782,0.325698,-1.350100,-0.267617,-1.345502,0.670638,-0.627094,-0.729306,0.413149,-0.368728,0.093917,-0.261972,0.962279,-0.908706,-1.512977,-0.350233,-0.181936,-1.154551,0.459797,1.100208,0.272066,1.296310,-1.162241,0.186550,0.000747,-0.110111,0.440150,-0.864575,0.501745,-0.838170,-0.951181,-1.095415,-0.253975,0.240575,0.433979,1.619877,-0.233954,-0.107192,-0.217507,0.175058,-1.168531,1.099229,-0.944279,0.950945,0.053495,0.370962,-1.502641,-0.094921,0.083064,-0.282475,1.172590,0.549233,1.891756,-0.654046,-0.229768,-0.050833,-1.046734,-1.712879,0.796123,1.003611,-1.224234,0.653223,-0.671211,-0.790519,-2.341905,1.096595,-0.208235,1.383214,0.157798,3.569868,1.735096,-2.174621,-1.220254,1.274955,-0.827086,-0.812697,1.583319,-0.459895,-1.518023,1.511008,-2.099479,-0.505073,0.576681,-0.489478,1.248095,-0.398163,-1.505125,1.303681,-0.527943,3.407532,-0.371696,0.144615,-0.376237,-0.551191,-0.831983,-0.438420,-0.249413,1.091000,-1.074917,-1.137239,-0.390243,-0.475994,-2.085770,2.974474,2.809232,-0.085463,1.809741,-0.125812,0.723061,1.146918,1.192864,0.669033,0.052429,-0.087409,0.497900,0.858610,1.294019,-0.163880,-3.072166,0.643018,0.664282,-2.051561,0.235964,-0.622585,-0.232185,2.373287,-0.116973,-0.596874,-0.849942,0.787258,0.953648,-0.848829,-1.953934,-0.347820,2.315626,0.195216,0.361980,-0.067240,0.521407,-0.012760,-0.702264,-0.448305,-0.778421,1.920303,0.287032,-0.473921,1.226209,-1.520646,-0.796384,-1.278137,-1.415045,-0.057090,0.565421,0.993963,-0.793826,0.888862,-0.263053,-1.001592,-0.991024,0.914260,0.501328,-1.551197,1.099562,0.961149,-0.464604,0.946341,0.494943,0.671842,0.725338,-0.584593,-0.033356,0.291786,0.077794,0.428436,0.540960,0.069222,1.082144,-0.555661,-0.253126,1.107695,0.540314,0.929826,-0.855564,-0.557803,0.384318,0.818162,0.908595,0.022844,1.686544,-0.619406,0.261118,-0.614619,0.080701,-1.004185,-0.559061,2.486821,0.982434,-1.351556,1.009265,0.820471,0.990751,0.901940,0.007558,-0.106555,-0.379828,1.588971,-1.154219,-1.177789,-0.386394,0.535301,0.522995,-0.013643,0.794708,-0.632688,1.976566,-1.665643,-0.111444,0.364211,0.051012,-0.817612,0.989378,0.138287,-0.937586,-0.215161,-0.101778,0.526012,1.257556,-0.334578,-0.505078,0.482987,0.676787,0.412113,1.043689,-1.045710,0.544660,-0.415928,1.295951,-0.452208,-0.440433,-0.126485,-0.688841,-0.378408,-0.681560,0.473490,1.618151,-2.465198,1.452532,-1.829455,0.408074,0.108475,0.674396,-0.937150,0.802185,0.505707,-0.137906,-0.084225,1.781295,0.783366,-0.848469,0.264143,-0.856838,0.143064,-0.260139,1.365641,-0.847230,-0.852528,-2.076811,0.429990,1.072356,-0.340285,0.678399,0.835157,-0.468824,0.331954,0.619876,0.089252,-0.065615,1.237237,-0.240371,3.158528,0.048390,1.605039,-0.228795,-1.637804,-0.575883,0.511735,-0.176488,-0.173993,0.967983,-0.925848,-0.312580,1.094538,-0.046592,-1.635413,-0.005583,1.456147,-0.255681,1.077198,0.602929,1.226617,-0.664853,1.349116,-0.524813,2.023729,-0.075913,0.257775,-0.652082,1.631242,0.270146,0.005322,0.379244,0.321181,0.353453,-1.906840,1.107199,0.219454,0.404638,-0.034841,-1.516254,2.320635,1.452742,-0.448442,1.128319,0.777789,-0.132472,1.962447,0.156534,-1.359428,-0.634407,1.139149,-2.252898,0.015113,-1.256801,-0.040399,-0.185590,-0.114877,0.351757,-0.503987,-0.068349,0.414498,1.379846,0.168411,0.550139,-0.548902,1.439310,1.406315,-0.850226,0.970507,0.629763,0.427361,-1.168351,-1.164493,-1.036031,0.697193,-1.121418,0.068602,0.807511,-1.224019,0.782393,0.211832,0.095136,-1.120497,1.855147,-0.126011,2.031263,0.496834,0.279046,0.143640,-0.079234,0.181032,1.201966,0.494841,-0.426963,0.168803,0.246450,0.751495,-0.255590,0.118509,-1.420726,0.613167,-0.427127,0.400731,-0.277299,0.299580,-0.978202,0.082831,0.597094,0.082604,1.376949,0.657480,-0.660739,-1.215849,0.077669,-1.543796,1.561037,-0.689427,0.715113,0.959474,0.669365,-0.527777,0.510808,0.739063,1.066631,0.296171,-0.641024,-1.548478,0.244717,0.716664,-1.408212,0.584246,0.327559,1.590788,1.790930,-1.551802,-1.196622,0.450814,1.048597,-0.340930,0.683237,1.241601,-0.656274,0.900045,-2.099239,1.200783,0.520476,1.863160,-0.118347,1.193484,0.141232,-1.616451,1.508127,0.526475,0.797263,0.867078,-0.242345,-1.565025,0.177549,-0.173318,-0.882136,-0.157631,-0.125017,-1.531218,0.638484,1.090173,-0.380700,0.134027,-1.289408,-1.071062,1.289719,0.018774,-0.888909,0.264819,-0.442501,-0.145373,1.004828,-0.078796,0.153298,0.616301,-1.504509,-1.373602,-0.530467,0.504621,0.371465,-0.358703,1.992852,-1.546003,0.057564,1.318902,-0.494936,-0.425961,-0.959285,-0.393598,-0.629091,-0.386242,-1.920126,-0.941792,-1.254324,0.863919,0.433139,0.870838,0.105586,-0.864215,-0.374179,-0.129928,1.635870,0.433282,-0.520933,-1.210003,-0.624825,-2.034027,-0.492719,-2.176310,1.533796,1.316204,0.625446,-0.652862,-1.172801,-1.416914,0.808256,-1.568533,1.128350,-0.376555,0.695348,0.733738,0.559019,0.102954,-1.279027,-1.074115,-0.933634,-1.328553,0.441057,-1.195654,2.733540,-0.796464,0.752960,0.282529,-1.461442,0.199952,0.578938,-1.844253,0.742478,0.788010,0.877629,0.120332,0.563638,-0.570346,-0.056227,-0.672560,-0.278725,-0.319886,-0.204764,0.365328,0.167439,0.135441,0.213484,-1.125079,-1.244288,0.195827,0.762849,0.288410,1.143603,0.298206,1.033606,1.136331,-0.338397,0.493062,-0.209680,0.736462,-0.200525,0.825430,-0.977807,2.412443,-0.478865,0.417806,-0.770234,-0.988974,-0.155099,-0.151132,-1.182127,-0.950911,-0.914714,-0.163742,0.419791,-0.686773,0.763647,-0.707513,0.192102,-1.069961,0.136741,-0.229293,-1.572947,0.740247,-1.412654,0.819880,-0.007135,-1.515859,-1.614914,0.892784,0.583848,-0.910732,0.179802,0.606733,0.601069,0.471683,1.127442,-1.348025,0.992332,0.334715};
    // double B_imag[N * K] {0.411883,-0.808845,1.772483,-1.586821,0.920985,-0.847371,0.564558,0.437824,-0.056144,0.836830,-0.813314,1.859280,0.549349,-0.197264,0.161436,1.021377,0.275259,1.306725,-0.466712,0.760868,0.154120,1.161004,-2.508804,-1.019146,-0.906881,1.337238,-0.801539,-0.143268,-0.841239,0.474509,0.793403,0.927072,0.467634,0.649153,-1.061811,-0.139831,0.353289,-0.284375,-1.435821,0.237920,0.554571,0.592105,-0.456589,-1.385234,0.144441,-1.616317,-1.852947,-0.341533,-0.154844,1.252166,-0.381945,-1.226964,0.191457,-0.831474,0.450494,-0.081551,-1.200376,-0.791710,-0.977716,-0.208373,-1.172846,0.242748,2.430378,0.954886,1.396409,-0.491818,0.053216,2.057811,1.188595,-0.892623,-1.371204,-0.327239,-0.229805,0.895954,-0.272798,1.258247,0.166129,0.240619,0.605895,0.024739,1.007587,0.240477,-0.471454,-0.601120,-0.189699,-0.969302,-0.436415,-0.374535,-0.414907,-0.895274,0.010307,0.891646,-0.579217,-1.813486,-0.101454,1.468534,0.776162,-0.094646,-0.113655,-0.028091,0.113520,-0.821500,-0.560328,-1.171891,-0.843836,-0.474108,0.280007,-0.190203,-0.658617,0.313326,0.204132,0.288186,0.480481,1.566683,-1.429106,0.032941,-1.381435,0.793235,0.764562,0.140359,0.730051,0.993112,-1.226467,-0.577110,1.943438,-0.666164,0.084606,-0.938732,0.983991,0.667033,-0.411026,2.265195,-0.386833,0.846501,-0.764409,1.692478,1.128279,0.959440,0.669897,0.237586,-0.983512,0.346395,0.792950,-0.836431,-0.246922,0.478375,0.729119,-0.813462,0.101227,0.828117,0.663561,-0.047890,0.421616,0.110155,0.410138,0.475729,2.465440,1.087103,0.375784,-0.671549,0.052032,-0.261134,-2.109940,0.852970,0.441619,-1.041058,0.900753,-0.583619,-0.362706,0.006509,0.225782,-1.551864,1.087706,-1.161102,-0.789933,0.368530,-1.557917,0.647169,0.766653,-1.045000,0.877599,-0.184713,-0.799395,0.477331,-0.803611,-0.820659,-1.325638,1.709525,-0.738828,-0.235892,-0.212843,0.444083,-2.249332,-0.397536,0.161637,2.192001,-2.066648,0.632550,1.523472,0.965767,1.014141,-0.101735,0.566975,0.302320,0.287609,0.101130,-1.524579,-0.390420,-1.479842,0.652821,-0.044035,-0.911819,1.804480,0.254294,1.977904,1.535685,-0.072599,1.903276,0.576824,-0.219757,-0.084348,-0.887043,-0.001443,0.415776,0.467018,1.083983,0.502433,-0.690065,0.153372,1.964533,0.458182,0.049435,-0.632060,1.207789,0.795266,0.907260,0.759111,-1.160026,0.425973,1.414536,-1.853532,0.741377,0.623925,0.042975,0.004562,-1.854527,0.306391,-0.448221,-0.317017,0.884584,-0.441417,1.078046,1.316458,-1.033542,1.037367,0.402260,-0.081664,-0.347309,-0.042931,-0.924187,-1.096821,1.392208,0.126441,-0.948853,0.326090,1.799753,-0.939160,0.714376,0.318918,0.084537,-1.054947,0.308179,1.551592,1.295147,2.360330,-0.538382,-0.158473,-1.695065,-1.242263,-0.594141,0.218628,2.473900,0.680868,0.541608,1.204587,-0.878374,1.237402,-1.315731,-1.456547,-0.574514,-0.155584,0.299639,-1.468908,2.768123,1.175395,-0.115215,0.006498,0.919386,-0.504952,1.421275,0.794246,0.503919,-1.287416,-0.821128,-1.404116,0.647733,0.033012,0.627122,1.321835,0.499751,0.129148,-0.197218,0.176938,-0.495347,0.397740,1.822000,-1.064608,0.021531,0.325851,1.407608,0.463124,-0.822481,0.218085,-1.071905,0.205564,1.049565,0.123568,-1.331295,0.487204,-0.484104,0.509428,-0.146422,3.466260,0.468766,1.951249,-2.021507,-1.643915,-0.299340,-1.111423,-1.029001,-0.612630,0.200982,-1.566567,-1.074092,-1.991196,1.323093,0.273939,-0.130848,-1.566626,0.238446,-0.030124,-0.103071,-0.214627,-0.657297,-0.904656,-0.636977,1.581550,-1.721016,0.468184,0.206512,2.244400,-1.007053,0.783282,0.869553,0.028409,-0.517359,0.754166,-1.826896,1.057060,0.778248,-0.457365,-2.798982,0.486283,-1.716957,-0.682152,-0.188803,0.292614,0.183432,0.322613,1.341111,0.072348,-0.613172,-0.310570,0.981051,1.930515,0.024315,1.524498,0.535922,-0.587578,0.924273,0.596336,0.393275,0.330885,1.470519,2.018292,0.148275,-1.187609,-0.259732,0.100021,1.332717,0.865514,-0.658963,0.655324,-1.758825,0.941137,-0.162367,-0.261976,0.692865,0.302222,0.588138,-0.113528,0.990247,1.267902,0.694137,0.454343,-1.783880,1.075937,-1.354789,0.301435,-1.284895,-0.415696,-0.833229,-0.537467,-0.148096,-0.006924,-0.605661,-1.730684,-0.687550,0.219462,1.377934,0.807024,-1.297633,1.090514,-0.510697,-0.042104,0.529068,-1.297372,-1.157442,0.023824,1.618376,-1.114938,0.378179,0.328643,0.251942,-0.338242,0.271710,1.013464,0.317576,-0.878201,1.851205,-0.089803,-1.521968,-0.946451,0.113387,0.458167,0.729732,-0.121770,-0.761986,-0.022055,0.661644,0.685252,-1.115343,1.054072,-0.610385,0.528266,0.358643,-0.927736,0.162842,-2.861400,-1.897665,-0.006292,0.621011,-0.438499,-0.229807,-0.264144,-0.940135,0.740772,-0.250300,-0.008758,0.227346,1.037673,0.667241,-1.979699,0.760980,1.056900,-0.352315,1.243680,1.159116,0.535555,-1.778684,-0.091861,-1.507523,0.343221,-1.461733,-0.291985,-0.259287,-0.840718,-1.648763,0.929455,-0.225576,1.822212,0.795333,-1.867388,0.254430,0.014741,0.132360,-0.589489,0.128388,-0.342037,-0.922641,-0.921238,-1.679402,-0.058403,-2.882336,-0.386798,-1.213545,-0.349471,-0.181991,-0.090432,-0.965993,-0.528988,1.037492,-1.832356,-0.152834,1.614761,2.673582,1.491482,1.039458,-0.597859,-1.997927,-0.926952,0.788976,2.534975,-0.047468,-0.553993,1.624550,0.029632,0.615687,-2.641146,0.095018,-1.627967,-0.020431,0.848592,-0.557411,-0.815855,-0.007566,-0.805978,-0.116771,0.418913,-0.357100,-0.961221,-0.654282,0.438645,-0.462466,-0.943728,0.373512,1.952373,-0.376674,-0.486150,-0.256733,1.617302,0.618953,0.405187,0.677018,-1.997759,0.710303,-1.112599,-0.648297,-0.689681,-0.336401,1.784830,1.244903,0.437518,-0.576646,-0.059848,-0.250892,0.459024,-0.698035,0.195970,2.310131,0.264137,1.803064,-0.702456,0.849297,-0.748191,-0.462539,0.956940,-0.037981,0.671191,0.250426,-0.200189,-1.292264,-0.837676,-0.845968,-0.574685,-0.925456,-0.399312,0.304122,0.959722,0.190060,-1.127146,0.052993,1.499046,0.469307,-0.199304,-0.527878,-1.363052,-0.199325,0.150897,0.286162,0.940412,-0.614352,-1.307503,-1.817236,1.441855,0.098660,-0.225385,1.101080,1.380324,-0.173404,-0.559176,-0.177890,0.137795,-0.239782,-0.372760,-0.345787,0.680585,0.881670,-0.991319,-0.686169,0.349156,0.241695,0.794143,-0.521731,0.235527,1.684280,-0.280283,-0.446608,-0.941464,-0.014029};
    // constexpr double C_Matlab_real[M * K] {1.725343,-3.070041,4.841147,3.385986,2.129013,-2.942496,-0.090210,1.181185,5.625918,-3.175873,-4.881923,4.710938,-3.249266,1.338714,-2.030641,-0.087490,4.842548,9.317006,-3.112684,-5.495728,0.692453,-5.928462,1.514125,3.232953,-0.344580,-1.338534,2.293735,2.436943,0.757613,-2.952683,-2.314841,7.264267,-6.888259,2.533568,-3.767397,-5.791336,5.801674,10.532570,-5.297222,-2.509722,3.064064,-2.973053,0.026715,4.011523,-1.022232,-0.326096,4.379109,-0.619032,4.914609,-2.866331,-3.089820,4.318562,-3.249038,5.859066,-1.969624,-2.586792,4.953597,2.993748,0.130277,-4.158931,1.108420,-5.700291,1.797230,2.264147,0.621401,-1.884641,3.312388,-0.501914,2.307743,1.433767,-2.188582,3.562575,-7.808491,3.283104,-2.461411,0.315015,1.557612,5.769051,-2.950955,-2.301161,-0.212379,-1.034444,-2.092533,3.133317,-2.175459,-1.775808,2.148525,1.567247,1.581348,-2.744428,-2.778487,4.195642,-4.987796,-3.071138,-6.414315,1.819329,4.473161,6.630657,-1.618710,-4.116945,-2.930794,-2.721797,-2.374219,1.026745,-0.387453,-2.058348,5.929617,0.443695,3.160957,-5.738529,-2.881520,9.045552,-5.624848,1.691924,-5.515030,3.167933,5.665378,4.486824,-2.130590,-4.897702,4.502970,-3.157281,1.462013,2.972112,-0.832812,-6.770843,1.733181,-0.008624,5.061499,-1.548339,-5.171155,3.381244,-5.491816,-1.111783,-0.780354,-2.258374,3.654226,5.019430,-3.253652,-4.294156,-0.468854,-3.597103,3.292023,3.656516,2.543710,-2.310194,2.324560,0.783744,2.296040,-1.225123,-4.203248,6.330481,-6.200803,4.054802,-0.364472,-3.045575,5.076285,1.266206,-4.782442,-6.348528,-0.465254,-2.560115,1.591120,3.445258,-0.480630,-0.027959,3.864660,1.248620,1.074492,-3.451792,-0.460044,5.002251,-4.919296,3.334818,-3.964623,-0.362331,8.205985,5.855267,-2.371862,-5.414925,1.110319,1.092188,0.235493,-0.689014,-2.484786,-0.709883,2.665848,-1.075142,6.435032,-2.672241,-1.073252,5.669599,-7.911855,3.970428,-6.592313,-0.000594,5.376729,2.593008,-3.376438,-1.843007,-1.466746,-2.547876,3.684971,2.722812,2.771659,-0.935027,3.457750,-0.940521,6.080547,-4.379386,-3.978113,3.828025,-4.445227,4.765139,-4.291907,0.713978,5.342226,5.089212,2.790723,-5.557862,1.844431,-1.897249,1.068326,0.206674,2.400525,-2.353480,4.399715,0.122522,5.049309,-1.966418,-3.883892,6.782477,-8.660614,3.457956,-4.475590,-0.538153,8.954199,7.588216,-2.213394,-0.360189,-1.622379,-2.710414,-0.661421,3.205892,4.311216,-0.901347,-2.195862,-1.018784,2.299173,-2.223482,-2.813096,3.786922,-4.873249,4.496156,-2.479142,-0.474971,8.475364,6.207793,-4.647673,-3.484856,-0.999757,-1.906733,3.175222,-0.337406,-0.261623,-1.238616,1.245987,2.702439,7.910771,-4.771172,-4.800002,5.720996,-4.971623,2.218397,-4.161130,-0.808310,8.333129,8.097564,-3.542341,-7.278308,-1.813540,-3.296999,-1.375659,-0.312934,0.865819,-0.051038,0.347365,-1.998929,1.680405,-0.530776,-3.636014,1.137554,-4.263155,0.586243,-3.236909,2.923467,11.077492,8.285983,-0.967929,-6.379468,-3.008385,-7.244409,1.894750,0.565125,3.889033,-2.387096,1.877393,-1.293076,5.084761,-4.066620,-0.416584,-0.635619,-1.927716,1.569925,-2.662616,-3.800123,5.438930,2.095725,-0.371279,-5.673856,0.559543,0.150394,1.552793,2.750509,-1.456267,1.863766,1.549044,-2.366998,3.092582,-3.728560,-2.601430,3.761530,-8.539107,3.134897,-2.615493,-0.202233,6.412955,4.602421,0.064648,-0.537642,-1.179543,-1.949474,1.635451,-1.184322,2.975744,-2.032183,4.066701,1.686167,2.659038,-3.944737,-3.719932,4.429415,-5.737844,5.749256,1.365755,-1.107849,4.612300,6.523096,-2.988190,-6.086818,-0.243705,-1.410625,3.410912,2.525687,1.321252,-3.115435,3.315306,1.437234,5.890870,-4.042526,-3.756174,7.113839,-4.975492,3.062731,-4.060520,-1.325855,4.594176,7.444722,-4.475095,-3.906745,0.552346,-2.830545,0.011146,2.983315,0.636406,0.160185,-2.079248,1.307073,5.247842,-0.347653,-3.518265,4.973145,-8.048869,3.111054,-3.840385,5.767413,11.532631,8.613050,-3.540017,-0.236295,3.392330,-1.391840,-0.492351,4.214426,-0.208085,-4.888734,1.631574,0.127238,-0.747772,-3.000384,-2.224096,5.023028,-6.507735,3.872024,-5.026761,-1.964268,7.183703,4.493723,-1.209684,-2.284357,0.490332,-8.716968,3.760465,2.033027,2.097394,-4.317233,2.687036,-0.123917,3.734491,-2.014552,-3.121657,6.982234,-2.123223,4.551475,-5.776593,0.629090,4.781216,7.013049,-1.528300,-2.386309,0.727914,1.413161,1.927753,-0.588081,2.107285,0.059812,2.591524,2.120417,4.880289,-6.032528,-4.955882,6.490793,-5.737371,2.820748,-2.803482,0.122543,6.319094,5.270431,-1.439677,-4.062969,2.209441,2.085912,0.197835,2.636766,-1.576466,-3.903946,1.631459,3.769650,2.727646,-1.746627,-5.094441,4.670144,-8.050918,4.630468,-3.932751,2.184680,7.224099,9.154377,-2.204642,1.049240};
    // constexpr double C_Matlab_imag[M * K] {4.546255,3.955828,-0.106507,-5.193781,8.243894,-5.857795,2.554178,-0.188127,-0.045024,-0.898069,-5.250295,1.142323,7.382656,-4.037973,2.675969,6.888286,4.636273,3.022852,-2.510386,-2.300655,6.760216,2.169187,-0.492695,-7.260972,3.551619,0.218304,1.499193,-2.203723,-0.878449,-0.450204,-6.705580,5.035710,8.028537,-1.908335,2.424070,2.741559,0.869498,4.586564,-3.795766,-1.846138,2.662902,4.533958,-6.665818,-5.204789,4.431529,1.401128,2.406291,3.598736,-3.153089,-1.837435,-3.913533,-0.243841,8.362064,1.243157,3.557800,7.746222,0.370911,3.243357,-6.750835,2.094842,4.176066,1.712827,-6.388790,-4.919434,3.174598,-1.246809,0.840654,0.964891,0.488199,-0.973372,-3.270754,1.601957,5.759644,-3.820711,-2.728065,8.271609,3.481760,2.397388,-0.358189,0.511972,0.150484,10.183362,-4.359678,-4.590628,2.535245,-1.823438,1.316872,-3.010417,-0.927434,-2.245900,-3.324017,3.809166,7.247316,-0.421451,0.734856,7.846113,2.878157,5.819727,-1.108239,1.056800,4.760719,5.289271,-6.462887,-2.380014,4.148947,-0.582256,5.740253,0.950844,-3.923906,-0.046435,-8.623131,-1.328851,4.451294,-2.949833,3.310627,9.567548,2.570350,-0.673990,3.057147,0.665303,4.739591,5.136585,-8.891981,-2.556341,2.278495,-0.618778,-0.817917,-0.623929,-3.893927,-0.116439,-1.613162,0.286169,7.300495,-4.840649,0.338765,7.242052,2.762276,3.433016,-2.268483,6.927586,4.543047,3.570695,-1.401792,-3.861560,-0.167846,-0.297508,-0.557827,-3.233421,-2.039252,1.425932,-4.736522,4.138396,3.414306,-4.653190,4.355020,8.086162,-0.066641,4.992288,-3.360022,4.878926,0.794993,2.882307,-0.794162,-6.653434,2.478059,-1.012274,2.695738,-2.151819,-5.504435,2.876867,-4.375269,4.364145,4.082696,-1.319264,1.565130,11.141020,4.074761,0.219035,-3.101120,2.229454,2.940812,2.340307,-3.747347,-5.551177,4.441437,-5.597196,2.610508,2.558808,-2.186594,3.195455,-5.882977,3.576526,3.177690,-1.744151,3.117521,14.938452,3.681867,3.607587,2.096567,2.886682,5.727488,2.034703,-3.824099,-2.991192,-0.497167,-1.423462,-0.128212,-1.506816,-3.993981,3.232896,-2.972912,-3.640008,5.235793,-2.444921,6.664027,5.810013,4.735773,1.767949,-0.314175,2.252966,5.767947,2.791545,-6.709605,-4.742052,3.916986,-2.360974,4.075042,-3.800966,-4.390232,1.256798,-3.329919,3.060207,7.599659,0.097362,3.565496,6.617229,3.919846,2.555646,0.144300,5.957457,6.927675,4.662179,-2.585729,-6.001112,3.740694,-0.283679,4.343483,-1.260878,-5.224376,0.680586,-2.556749,-0.381701,4.186053,-3.056417,2.775990,8.185751,7.403339,-0.220153,-0.650766,5.054359,5.443241,3.331662,-4.252190,-3.244057,6.883638,-4.556931,6.170059,-2.030312,-0.035689,-1.274383,-3.743715,3.331812,6.056939,-1.537907,3.428417,8.329747,5.208080,7.632849,-1.749460,2.811580,5.421788,5.078254,-3.372948,-2.723731,4.447223,-3.074020,0.362774,2.008222,-0.665824,0.029844,-3.850226,3.107416,4.805104,-1.118247,0.038985,9.415476,4.994800,3.448342,-4.358851,3.776377,1.331193,4.908063,3.171324,-2.428108,4.502794,-3.871070,0.379810,1.759195,-2.742802,-0.284583,-5.834544,1.461477,0.871866,0.979096,0.949125,9.041734,5.701900,1.290052,-0.998005,0.296701,4.645417,5.759304,-4.188174,-6.459362,1.548762,-1.867281,-1.190994,-0.642140,-2.510065,2.415462,-4.563711,0.995555,6.521453,-1.348224,1.757329,8.746908,3.325054,7.530377,-3.148000,-0.201597,7.657966,3.738856,-0.797494,-4.397359,6.649451,-6.137268,0.270652,1.662709,-1.978749,4.120427,-6.512446,3.191167,2.289341,-0.512054,1.805143,10.179021,0.984443,4.365908,-2.998621,3.368771,4.466451,3.007323,-1.830011,-3.513340,0.554348,-3.359269,0.979034,-3.272708,-2.205098,2.248072,-5.245187,2.820491,4.837623,-4.038139,0.755202,5.756732,3.865610,5.532012,0.204600,1.521043,4.290835,1.317499,-4.267999,-4.850701,5.791151,-2.731943,2.338436,-0.732724,-8.221112,2.003023,-2.047087,3.839234,8.214162,-0.420570,6.720400,8.463292,3.694948,2.094783,-5.042537,5.083814,1.216770,2.464627,-1.769359,-3.901256,0.757106,-0.985188,3.632967,-2.425901,-0.491832,1.731895,-2.616129,3.281452,6.297959,-3.764078,0.134771,8.280124,4.496537,5.509523,0.660351,2.830139,4.063547,3.896696,-3.274622,-2.519308,2.075157,-1.624200,-0.589119,-0.951015,-2.748491,2.009874,-4.904922,-1.733914,3.961345,-0.853998,1.905962,6.913365,3.205243,-3.023951,-0.942284,2.754812,2.865476,4.625718,-3.267286,-5.454763,2.463499,-0.488756,0.964475,-1.567351,-3.077367,6.575524,-5.915706,1.042379,5.927831,3.939695,2.672795,10.016160,6.239850,4.462352,-2.033878,6.584848,7.347995,2.071678,-0.829518,-4.348825,3.087595,-3.558415,-0.160151,-2.704361,-4.426807,6.711827,-3.389338,6.455933,2.473648,1.289732,1.363989,8.278472,4.644720,2.639358,-1.249509,7.312321};
    // double C_real[M * K];
    // double C_imag[M * K];
    // double C_MatrixUtil_real[M * K];
    // double C_MatrixUtil_imag[M * K];

    // constexpr size_t M = 16;
    //  float A_real[M]{0.814724,0.905792,0.126987,0.913376,0.632359,0.097540,0.278498,0.546882,0.957507,0.964889,0.157613,0.970593,0.957167,0.485376,0.800280,0.141886};
    //  float A_imag[M]{0.421761,0.915736,0.792207,0.959492,0.655741,0.035712,0.849129,0.933993,0.678735,0.757740,0.743132,0.392227,0.655478,0.171187,0.706046,0.031833};
    //  float B_real[M * M]{-1.068870,1.093266,1.544212,1.419310,-0.082494,1.354594,0.701541,-0.831367,-0.293754,-0.130285,-0.303108,-0.162338,-1.506160,-1.066701,2.023691,1.006077,-0.809499,1.109273,0.085931,0.291584,-1.933023,-1.072155,-2.051816,-0.979206,-0.847926,0.183689,0.023046,-0.146055,-0.444628,0.933728,-2.258354,-0.650908,-2.944284,-0.863653,-1.491590,0.197811,-0.438966,0.960954,-0.353850,-1.156402,-1.120128,-0.476153,0.051290,-0.532011,-0.155941,0.350321,2.229446,0.257056,1.438380,0.077359,-0.742302,1.587699,-1.794679,0.124050,-0.823587,-0.533557,2.526000,0.862022,0.826063,1.682104,0.276068,-0.029006,0.337564,-0.944378,0.325191,-1.214117,-1.061582,-0.804466,0.840376,1.436697,-1.577057,-2.002636,1.655498,-1.361694,1.526977,-0.875729,-0.261164,0.182452,1.000061,-1.321789,-0.754928,-1.113501,2.350457,0.696624,-0.888032,-1.960900,0.507975,0.964229,0.307535,0.455030,0.466914,-0.483815,0.443422,-1.565056,-1.664164,0.924826,1.370299,-0.006849,-0.615602,0.835088,0.100093,-0.197698,0.281984,0.520060,-1.257118,-0.848709,-0.209713,-0.712005,0.391894,-0.084539,-0.590035,0.000050,-1.711516,1.532630,0.748077,-0.243715,-0.544529,-1.207845,0.033480,-0.020028,-0.865468,-0.334887,0.625190,-1.174212,-1.250679,1.603946,-0.278064,-0.054919,-0.102242,-0.769666,-0.192419,0.215670,0.303521,2.908008,-1.333678,-0.034771,-0.176534,0.552783,0.183227,-0.192240,-0.947961,0.098348,0.422716,0.911127,-0.241447,0.371379,0.888610,-1.165844,-0.600327,0.825219,1.127492,-0.798164,0.791416,1.039091,-1.029768,-0.274070,-0.741106,0.041374,-1.670201,0.594584,0.319207,-0.225584,-0.764849,-1.147953,0.489965,1.378972,0.350179,1.018685,-1.332004,-1.117639,0.949222,1.530073,-0.507818,-0.734169,0.471634,0.350201,0.312859,1.117356,-1.402269,0.104875,0.739363,-1.058180,-0.299066,-0.133217,-2.329867,1.260659,0.307062,-0.249025,-0.320576,-0.030814,-1.212847,1.250251,-0.864880,-1.089064,-1.422376,0.722254,1.711888,-0.468616,0.022890,-0.714530,-1.449097,0.660143,0.135175,-1.064213,0.012469,0.232347,0.066190,0.929789,-0.030051,0.032557,0.488194,2.585491,-0.194124,-0.272469,-0.261995,1.351386,0.333511,-0.067866,0.515246,1.603457,-3.029177,0.426388,0.652356,0.239763,-0.164879,0.552527,-0.177375,-0.666891,-2.138355,1.098425,-1.750212,-0.224771,0.391354,-0.195221,0.261406,1.234679,-0.457015,-0.372809,0.327060,-0.690361,0.627707,1.100610,-0.196053,0.187331,-0.839589,-0.277872,-0.285651,-0.589029,0.451679,-0.217606,-0.941486,-0.229626,1.242448,-0.236455,1.082634,-0.651554};
    //  float B_imag[M * M]{1.192102,0.857733,-1.014944,2.177779,-0.784146,-0.284141,-1.035985,0.269649,-0.292588,0.600143,-0.425058,-1.521027,0.176947,-0.737060,-0.433609,-0.507323,-1.611830,-0.691159,-0.471070,1.138465,-1.805373,-0.086690,1.877865,0.494287,-0.540786,0.593931,0.589433,-0.723631,-0.307503,-1.749879,-0.168470,0.235810,-0.024462,0.449378,0.137025,-2.496887,1.858593,-1.469395,0.940704,-1.483121,-0.308642,-2.186022,-0.062791,-0.593250,-0.131820,0.910483,-0.218534,0.245805,-1.948847,0.100633,-0.291863,0.441327,-0.604530,0.192182,0.787346,-1.020264,-1.096593,-1.327043,-2.021959,0.401336,0.595358,0.867083,0.541334,0.070045,1.020498,0.826070,0.301819,-1.398138,0.103360,-0.822293,-0.875874,-0.446995,-0.493010,-1.441014,-0.982132,0.942133,1.046833,-0.079893,0.389266,-0.608581,0.861716,0.536157,0.399931,-0.255055,0.563167,-0.094241,0.319949,0.109659,-0.180739,0.401844,0.612511,0.300486,-0.197959,0.898476,0.751229,-1.222593,0.001162,0.897888,-0.929962,0.164404,0.113597,0.336213,-0.558294,1.128736,0.045841,1.470201,-0.054886,-0.373071,0.327678,0.183703,1.778256,0.316500,-0.070837,-0.131938,-0.176830,0.747734,-0.904726,-0.904654,-0.311429,-0.289963,-0.063783,-0.326814,-1.118732,0.815489,-0.238302,0.290790,1.223063,-1.342869,-2.486284,-0.147201,-2.132095,-0.273047,-0.467715,-0.288256,-0.570010,1.261551,0.611335,0.812323,-0.626379,0.798887,0.229597,0.112945,-1.283256,-1.032184,0.581172,1.007773,1.145362,1.576300,-0.124890,0.350063,-1.025734,0.475425,0.109318,0.545540,0.249518,0.120205,0.439998,0.439952,-2.328955,1.331216,-2.192435,-2.123655,-0.629091,-0.480937,1.478958,-1.835859,-0.908746,1.174117,1.814015,-1.051632,-0.993019,0.571248,-0.616866,0.101662,0.901931,-0.418903,-2.319280,-0.504586,-1.203850,0.327512,-0.860816,1.035976,-0.209897,0.126947,0.312024,0.397467,0.974950,0.412796,0.274837,2.787335,-1.835639,-0.140322,0.079934,-1.270594,-0.253945,0.664734,0.784668,2.424461,-1.698864,-0.656816,1.804494,-0.751895,-0.640710,-0.986962,0.601102,-1.166665,0.066757,0.899822,-0.948481,-0.382585,-1.428647,0.085189,0.308623,0.959401,0.607601,-1.481399,-0.723121,1.516267,1.808863,0.759568,0.092308,-1.854299,0.035479,-0.300111,0.411491,0.648679,-0.020858,0.880953,-0.233860,-0.315772,-0.117798,0.155489,0.526547,-0.032567,-1.079866,-0.657201,1.729841,-1.140681,2.227168,1.029366,0.676978,0.825727,-0.560665,0.323213,-1.056973,0.428623,0.699160,0.818551,-0.260251,1.636000,0.199189,-0.603918,-0.608557,-1.093343,-0.069214,-0.345066};
    //
    //  constexpr float C_Matlab_real[M]{3.862976,2.203384,1.897545,0.814627,-2.083515,5.024381,-3.368916,-4.702002,-2.438559,4.602648,6.715126,0.125027,-8.501292,1.193197,-3.044202,1.668985};
    //  constexpr float C_Matlab_imag[M]{-8.910003,1.418880,-8.021302,6.794002,-7.361468,5.788503,-6.943144,-3.570569,-1.249440,0.398305,-0.344431,-0.612275,-0.665065,0.778557,-1.789679,0.532209};
    //  float C_real[M];
    //  float C_imag[M];
    //  float C_MatrixUtil_real[M];
    //  float C_MatrixUtil_imag[M];

    // double A_real[M]{0.814724,0.905792,0.126987,0.913376,0.632359,0.097540,0.278498,0.546882,0.957507,0.964889,0.157613,0.970593,0.957167,0.485376,0.800280,0.141886};
    // double A_imag[M]{0.421761,0.915736,0.792207,0.959492,0.655741,0.035712,0.849129,0.933993,0.678735,0.757740,0.743132,0.392227,0.655478,0.171187,0.706046,0.031833};
    // double B_real[M * M]{-1.068870,1.093266,1.544212,1.419310,-0.082494,1.354594,0.701541,-0.831367,-0.293754,-0.130285,-0.303108,-0.162338,-1.506160,-1.066701,2.023691,1.006077,-0.809499,1.109273,0.085931,0.291584,-1.933023,-1.072155,-2.051816,-0.979206,-0.847926,0.183689,0.023046,-0.146055,-0.444628,0.933728,-2.258354,-0.650908,-2.944284,-0.863653,-1.491590,0.197811,-0.438966,0.960954,-0.353850,-1.156402,-1.120128,-0.476153,0.051290,-0.532011,-0.155941,0.350321,2.229446,0.257056,1.438380,0.077359,-0.742302,1.587699,-1.794679,0.124050,-0.823587,-0.533557,2.526000,0.862022,0.826063,1.682104,0.276068,-0.029006,0.337564,-0.944378,0.325191,-1.214117,-1.061582,-0.804466,0.840376,1.436697,-1.577057,-2.002636,1.655498,-1.361694,1.526977,-0.875729,-0.261164,0.182452,1.000061,-1.321789,-0.754928,-1.113501,2.350457,0.696624,-0.888032,-1.960900,0.507975,0.964229,0.307535,0.455030,0.466914,-0.483815,0.443422,-1.565056,-1.664164,0.924826,1.370299,-0.006849,-0.615602,0.835088,0.100093,-0.197698,0.281984,0.520060,-1.257118,-0.848709,-0.209713,-0.712005,0.391894,-0.084539,-0.590035,0.000050,-1.711516,1.532630,0.748077,-0.243715,-0.544529,-1.207845,0.033480,-0.020028,-0.865468,-0.334887,0.625190,-1.174212,-1.250679,1.603946,-0.278064,-0.054919,-0.102242,-0.769666,-0.192419,0.215670,0.303521,2.908008,-1.333678,-0.034771,-0.176534,0.552783,0.183227,-0.192240,-0.947961,0.098348,0.422716,0.911127,-0.241447,0.371379,0.888610,-1.165844,-0.600327,0.825219,1.127492,-0.798164,0.791416,1.039091,-1.029768,-0.274070,-0.741106,0.041374,-1.670201,0.594584,0.319207,-0.225584,-0.764849,-1.147953,0.489965,1.378972,0.350179,1.018685,-1.332004,-1.117639,0.949222,1.530073,-0.507818,-0.734169,0.471634,0.350201,0.312859,1.117356,-1.402269,0.104875,0.739363,-1.058180,-0.299066,-0.133217,-2.329867,1.260659,0.307062,-0.249025,-0.320576,-0.030814,-1.212847,1.250251,-0.864880,-1.089064,-1.422376,0.722254,1.711888,-0.468616,0.022890,-0.714530,-1.449097,0.660143,0.135175,-1.064213,0.012469,0.232347,0.066190,0.929789,-0.030051,0.032557,0.488194,2.585491,-0.194124,-0.272469,-0.261995,1.351386,0.333511,-0.067866,0.515246,1.603457,-3.029177,0.426388,0.652356,0.239763,-0.164879,0.552527,-0.177375,-0.666891,-2.138355,1.098425,-1.750212,-0.224771,0.391354,-0.195221,0.261406,1.234679,-0.457015,-0.372809,0.327060,-0.690361,0.627707,1.100610,-0.196053,0.187331,-0.839589,-0.277872,-0.285651,-0.589029,0.451679,-0.217606,-0.941486,-0.229626,1.242448,-0.236455,1.082634,-0.651554};
    // double B_imag[M * M]{1.192102,0.857733,-1.014944,2.177779,-0.784146,-0.284141,-1.035985,0.269649,-0.292588,0.600143,-0.425058,-1.521027,0.176947,-0.737060,-0.433609,-0.507323,-1.611830,-0.691159,-0.471070,1.138465,-1.805373,-0.086690,1.877865,0.494287,-0.540786,0.593931,0.589433,-0.723631,-0.307503,-1.749879,-0.168470,0.235810,-0.024462,0.449378,0.137025,-2.496887,1.858593,-1.469395,0.940704,-1.483121,-0.308642,-2.186022,-0.062791,-0.593250,-0.131820,0.910483,-0.218534,0.245805,-1.948847,0.100633,-0.291863,0.441327,-0.604530,0.192182,0.787346,-1.020264,-1.096593,-1.327043,-2.021959,0.401336,0.595358,0.867083,0.541334,0.070045,1.020498,0.826070,0.301819,-1.398138,0.103360,-0.822293,-0.875874,-0.446995,-0.493010,-1.441014,-0.982132,0.942133,1.046833,-0.079893,0.389266,-0.608581,0.861716,0.536157,0.399931,-0.255055,0.563167,-0.094241,0.319949,0.109659,-0.180739,0.401844,0.612511,0.300486,-0.197959,0.898476,0.751229,-1.222593,0.001162,0.897888,-0.929962,0.164404,0.113597,0.336213,-0.558294,1.128736,0.045841,1.470201,-0.054886,-0.373071,0.327678,0.183703,1.778256,0.316500,-0.070837,-0.131938,-0.176830,0.747734,-0.904726,-0.904654,-0.311429,-0.289963,-0.063783,-0.326814,-1.118732,0.815489,-0.238302,0.290790,1.223063,-1.342869,-2.486284,-0.147201,-2.132095,-0.273047,-0.467715,-0.288256,-0.570010,1.261551,0.611335,0.812323,-0.626379,0.798887,0.229597,0.112945,-1.283256,-1.032184,0.581172,1.007773,1.145362,1.576300,-0.124890,0.350063,-1.025734,0.475425,0.109318,0.545540,0.249518,0.120205,0.439998,0.439952,-2.328955,1.331216,-2.192435,-2.123655,-0.629091,-0.480937,1.478958,-1.835859,-0.908746,1.174117,1.814015,-1.051632,-0.993019,0.571248,-0.616866,0.101662,0.901931,-0.418903,-2.319280,-0.504586,-1.203850,0.327512,-0.860816,1.035976,-0.209897,0.126947,0.312024,0.397467,0.974950,0.412796,0.274837,2.787335,-1.835639,-0.140322,0.079934,-1.270594,-0.253945,0.664734,0.784668,2.424461,-1.698864,-0.656816,1.804494,-0.751895,-0.640710,-0.986962,0.601102,-1.166665,0.066757,0.899822,-0.948481,-0.382585,-1.428647,0.085189,0.308623,0.959401,0.607601,-1.481399,-0.723121,1.516267,1.808863,0.759568,0.092308,-1.854299,0.035479,-0.300111,0.411491,0.648679,-0.020858,0.880953,-0.233860,-0.315772,-0.117798,0.155489,0.526547,-0.032567,-1.079866,-0.657201,1.729841,-1.140681,2.227168,1.029366,0.676978,0.825727,-0.560665,0.323213,-1.056973,0.428623,0.699160,0.818551,-0.260251,1.636000,0.199189,-0.603918,-0.608557,-1.093343,-0.069214,-0.345066};
    //
    // constexpr double C_Matlab_real[M]{3.862976,2.203384,1.897545,0.814627,-2.083515,5.024381,-3.368916,-4.702002,-2.438559,4.602648,6.715126,0.125027,-8.501292,1.193197,-3.044202,1.668985};
    // constexpr double C_Matlab_imag[M]{-8.910003,1.418880,-8.021302,6.794002,-7.361468,5.788503,-6.943144,-3.570569,-1.249440,0.398305,-0.344431,-0.612275,-0.665065,0.778557,-1.789679,0.532209};
    // double C_real[M];
    // double C_imag[M];
    // double C_MatrixUtil_real[M];
    // double C_MatrixUtil_imag[M];

    TimerUtil timer;

    timer.tic();
    for (size_t i = 0; i < 1000; ++i)
    {
        // MatrixUtil::MatrixComplexMult(A_real, A_imag, M, N, B_real, B_imag, N, K, C_MatrixUtil_real, C_MatrixUtil_imag);
        MatrixUtil::MatrixComplexMultFloat(A_real, A_imag, M, N, B_real, B_imag, N, K, C_MatrixUtil_real, C_MatrixUtil_imag);
        // MatrixUtil::MatrixComplexMultFloat(A_real, A_imag, 1, M, B_real, B_imag, M, M, C_MatrixUtil_real, C_MatrixUtil_imag);
    }
    timer.toc("Complex MatrixUtil A(24x32) B(32x20) Elapsed Time");

    logPRN(LOG_TYPE_GENERAL, "\nDifference (MATLAB - MatrixUtil) Real Part:\n\n");
    printMatrixDifference(C_Matlab_real, C_MatrixUtil_real, M, K);

    logPRN(LOG_TYPE_GENERAL, "\nDifference (MATLAB - MatrixUtil) Imag Part:\n\n");
    printMatrixDifference(C_Matlab_imag, C_MatrixUtil_imag, M, K);

    timer.tic();
    for (size_t i = 0; i < 1000; ++i)
    {
        // Vector1x16MatrixMultiplicationComplexDouble16x16(A_real, A_imag, B_real, B_imag, C_real, C_imag);
        // Vector1x16MatrixMultiplicationComplexFloat16x16(A_real, A_imag, B_real, B_imag, C_real, C_imag);
        // MatrixMultComplexDouble(A_real, A_imag, M, N, B_real, B_imag, N, K, C_real, C_imag);
        MatrixMultComplexFloat(A_real, A_imag, M, N, B_real, B_imag, N, K, C_real, C_imag);
    }
    timer.toc("Complex Ne10 A(24x32) B(32x20) Elapsed Time");

    logPRN(LOG_TYPE_GENERAL, "\nDifference (MATLAB - Ne10) Real Part:\n\n");
    printMatrixDifference(C_Matlab_real, C_real, M, K);

    logPRN(LOG_TYPE_GENERAL, "\nDifference (MATLAB - Ne10) Imag Part:\n\n");
    printMatrixDifference(C_Matlab_imag, C_imag, M, K);

    logPRN(LOG_TYPE_GENERAL, "\nDifference (MatrixUtil - Ne10) Real Part:\n\n");
    printMatrixDifference(C_MatrixUtil_real, C_real, M, K);

    logPRN(LOG_TYPE_GENERAL, "\nDifference (MatrixUtil - Ne10) Imag Part:\n\n");
    printMatrixDifference(C_MatrixUtil_imag, C_imag, M, K);
}
//
void Ne10Util::testRealMultiplications()
{
    // To Test double, change the functions and inputs!

    constexpr size_t M = 24;
    constexpr size_t N = 32;
    constexpr size_t K = 20;
    constexpr size_t vectorSize = 16;

    // float A[M * N] = {0.814724,0.678735,0.709365,0.814285,0.568824,0.106653,0.401808,0.575209,0.486792,0.225922,0.085516,0.098712,0.805489,0.972975,0.372410,0.032601,0.824376,0.068806,0.637709,0.322472,0.647618,0.318074,0.192028,0.683416,0.768854,0.850713,0.083483,0.123084,0.105709,0.467068,0.178117,0.441722,0.905792,0.757740,0.754687,0.243525,0.469391,0.961898,0.075967,0.059780,0.435859,0.170708,0.262482,0.261871,0.576722,0.648991,0.198118,0.561200,0.982663,0.319600,0.957694,0.784739,0.679017,0.119215,0.138874,0.704047,0.167254,0.560560,0.625960,0.205494,0.142041,0.648198,0.359635,0.013283,0.126987,0.743132,0.276025,0.929264,0.011902,0.004634,0.239916,0.234780,0.446784,0.227664,0.801015,0.335357,0.182922,0.800331,0.489688,0.881867,0.730249,0.530864,0.240707,0.471357,0.635787,0.939829,0.696266,0.442305,0.861980,0.929609,0.660945,0.146515,0.166460,0.025228,0.056705,0.897191,0.913376,0.392227,0.679703,0.349984,0.337123,0.774910,0.123319,0.353159,0.306349,0.435699,0.029220,0.679728,0.239932,0.453798,0.339493,0.669175,0.343877,0.654446,0.676122,0.035763,0.945174,0.645552,0.093820,0.019578,0.989872,0.696667,0.729752,0.189072,0.620959,0.842207,0.521886,0.196658,0.632359,0.655478,0.655098,0.196595,0.162182,0.817303,0.183908,0.821194,0.508509,0.311102,0.928854,0.136553,0.886512,0.432392,0.951630,0.190433,0.584069,0.407619,0.289065,0.175874,0.208935,0.479463,0.525404,0.330858,0.514423,0.582791,0.890752,0.042652,0.573710,0.559033,0.335849,0.093371,0.097540,0.171187,0.162612,0.251084,0.794285,0.868695,0.239953,0.015403,0.510772,0.923380,0.730331,0.721227,0.028674,0.825314,0.920332,0.368917,0.107769,0.819981,0.671808,0.721758,0.709282,0.639317,0.530344,0.424309,0.884281,0.815397,0.982303,0.635198,0.052078,0.854100,0.175669,0.307367,0.278498,0.706046,0.118998,0.616045,0.311215,0.084436,0.417267,0.043024,0.817628,0.430207,0.488609,0.106762,0.489901,0.083470,0.052677,0.460726,0.906308,0.718359,0.695140,0.473486,0.236231,0.544716,0.861140,0.270270,0.588026,0.879014,0.769029,0.281867,0.931201,0.347879,0.208947,0.456058,0.546882,0.031833,0.498364,0.473289,0.528533,0.399783,0.049654,0.168990,0.794831,0.184816,0.578525,0.653757,0.167927,0.133171,0.737858,0.981638,0.879654,0.968649,0.067993,0.152721,0.119396,0.647311,0.484853,0.197054,0.154752,0.988912,0.581446,0.538597,0.728662,0.446027,0.905154,0.101669,0.957507,0.276923,0.959744,0.351660,0.165649,0.259870,0.902716,0.649115,0.644318,0.904881,0.237284,0.494174,0.978681,0.173389,0.269119,0.156405,0.817761,0.531334,0.254790,0.341125,0.607304,0.543886,0.393456,0.821721,0.199863,0.000522,0.928313,0.695163,0.737842,0.054239,0.675391,0.995390,0.964889,0.046171,0.340386,0.830829,0.601982,0.800068,0.944787,0.731722,0.378609,0.979748,0.458849,0.779052,0.712694,0.390938,0.422836,0.855523,0.260728,0.325146,0.224040,0.607389,0.450138,0.721047,0.671431,0.429921,0.406955,0.865439,0.580090,0.499116,0.063405,0.177108,0.468468,0.332093,0.157613,0.097132,0.585268,0.585264,0.262971,0.431414,0.490864,0.647746,0.811580,0.438870,0.963089,0.715037,0.500472,0.831380,0.547871,0.644765,0.594356,0.105629,0.667833,0.191745,0.458725,0.522495,0.741258,0.887771,0.748706,0.612566,0.016983,0.535801,0.860441,0.662808,0.912132,0.297347,0.970593,0.823458,0.223812,0.549724,0.654079,0.910648,0.489253,0.450924,0.532826,0.111119,0.546806,0.903721,0.471088,0.803364,0.942737,0.376272,0.022513,0.610959,0.844392,0.738427,0.661945,0.993705,0.520052,0.391183,0.825584,0.989950,0.120860,0.445183,0.934405,0.330829,0.104012,0.062045,0.957167,0.694829,0.751267,0.917194,0.689215,0.181847,0.337719,0.547009,0.350727,0.258065,0.521136,0.890923,0.059619,0.060471,0.417744,0.190924,0.425259,0.778802,0.344462,0.242850,0.770286,0.218677,0.347713,0.769114,0.789963,0.527680,0.862711,0.123932,0.984398,0.898486,0.745546,0.298244,0.485376,0.317099,0.255095,0.285839,0.748152,0.263803,0.900054,0.296321,0.939002,0.408720,0.231594,0.334163,0.681972,0.399258,0.983052,0.428253,0.312719,0.423453,0.780520,0.917424,0.350218,0.105798,0.149997,0.396792,0.318524,0.479523,0.484297,0.490357,0.858939,0.118155,0.736267,0.046351,0.800280,0.950222,0.505957,0.757200,0.450542,0.145539,0.369247,0.744693,0.875943,0.594896,0.488898,0.698746,0.042431,0.526876,0.301455,0.482022,0.161485,0.090823,0.675332,0.269062,0.662010,0.109697,0.586092,0.808514,0.534064,0.801348,0.844856,0.852998,0.785559,0.988418,0.561861,0.505428,0.141886,0.034446,0.699077,0.753729,0.083821,0.136069,0.111203,0.188955,0.550156,0.262212,0.624060,0.197810,0.071445,0.416799,0.701099,0.120612,0.178766,0.266471,0.006715,0.765500,0.416159,0.063591,0.262145,0.755077,0.089951,0.227843,0.209405,0.873927,0.513377,0.539982,0.184194,0.761426,0.421761,0.438744,0.890903,0.380446,0.228977,0.869292,0.780252,0.686775,0.622475,0.602843,0.679136,0.030541,0.521650,0.656860,0.666339,0.589507,0.422886,0.153657,0.602170,0.188662,0.841929,0.404580,0.044454,0.377396,0.111706,0.498094,0.552291,0.270294,0.177602,0.706917,0.597211,0.631070,0.915736,0.381558,0.959291,0.567822,0.913337,0.579705,0.389739,0.183511,0.587045,0.711216,0.395515,0.744074,0.096730,0.627973,0.539126,0.226188,0.094229,0.281005,0.386771,0.287498,0.832917,0.448373,0.754933,0.216019,0.136293,0.900852,0.629883,0.208461,0.398589,0.999492,0.299937,0.089892,0.792207,0.765517,0.547216,0.075854,0.152378,0.549860,0.241691,0.368485,0.207742,0.221747,0.367437,0.500022,0.818149,0.291984,0.698106,0.384619,0.598524,0.440085,0.915991,0.091113,0.256441,0.365816,0.242785,0.790407,0.678652,0.574661,0.031991,0.564980,0.133931,0.287849,0.134123,0.080862,0.959492,0.795200,0.138624,0.053950,0.825817,0.144955,0.403912,0.625619,0.301246,0.117418,0.987982,0.479922,0.817547,0.431651,0.666528,0.582986,0.470924,0.527143,0.001151,0.576209,0.613461,0.763505,0.442402,0.949304,0.495177,0.845178,0.614713,0.640312,0.030890,0.414523,0.212602,0.777241,0.655741,0.186873,0.149294,0.530798,0.538342,0.853031,0.096455,0.780227,0.470923,0.296676,0.037739,0.904722,0.722440,0.015487,0.178132,0.251806,0.695949,0.457424,0.462449,0.683363,0.582249,0.627896,0.687796,0.327565,0.189710,0.738640,0.362411,0.417029,0.939142,0.464840,0.894942,0.905135,0.035712,0.489764,0.257508,0.779167,0.996135,0.622055,0.131973,0.081126,0.230488,0.318778,0.885168,0.609867,0.149865,0.984064,0.128014,0.290441,0.699888,0.875372,0.424349,0.546593,0.540739,0.771980,0.359228,0.671264,0.495006,0.585987,0.049533,0.205976,0.301306,0.763957,0.071453,0.533772,0.849129,0.445586,0.840717,0.934011,0.078176,0.350952,0.942051,0.929386,0.844309,0.424167,0.913287,0.617666,0.659605,0.167168,0.999080,0.617091,0.638531,0.518052,0.460916,0.425729,0.869941,0.932854,0.736340,0.438645,0.147608,0.246735,0.489570,0.947933,0.295534,0.818204,0.242487,0.109154,0.933993,0.646313,0.254282,0.129906,0.442678,0.513250,0.956135,0.775713,0.194764,0.507858,0.796184,0.859442,0.518595,0.106216,0.171121,0.265281,0.033604,0.943623,0.770160,0.644443,0.264779,0.972741,0.394707,0.833501,0.054974,0.666416,0.192510,0.082071,0.332936,0.100222,0.053754,0.825809};
    // float B[N * K] = {-0.592656,-0.773064,-0.268183,-0.729445,0.547640,-0.774513,0.039740,-0.112437,0.269541,-0.848110,-0.897601,-0.802823,0.112440,-0.618593,-0.374437,-1.075235,0.303564,-0.225794,1.281631,0.901491,-0.469809,0.836634,-0.409873,1.147328,1.565084,-1.393273,-0.450599,-1.556594,-2.564449,-0.764753,-0.792337,-1.265636,-0.308625,0.512016,-1.451741,-0.090967,-0.790258,-0.404922,0.620090,0.394676,0.886377,-1.128330,-0.711323,0.597865,-1.693344,-0.386235,0.109248,1.915102,0.465864,-1.127695,-0.952975,-0.149331,0.456660,0.011354,-0.618682,-0.252772,0.803380,0.527859,-0.286674,0.004854,-1.385220,-1.424470,0.061445,-1.281281,-0.449397,0.525586,-0.250553,0.609846,1.853561,0.078189,0.353905,-1.636447,-0.275101,-0.043989,0.934501,1.194824,-1.319903,-1.006963,0.598016,0.436919,-1.956754,0.717442,-1.846129,-2.203264,-0.084292,1.523269,-0.189902,-0.647912,1.039289,2.106630,1.597026,0.017344,0.443144,2.949093,1.055929,0.606064,-0.273846,1.089027,-0.245533,1.130073,0.420684,-0.777906,-0.398333,-0.571246,-1.991997,1.798494,-1.032914,2.617335,0.910897,-0.715847,0.527470,0.828387,-0.134765,-0.630046,0.160227,0.540514,0.271867,1.784875,-1.780737,0.153771,0.400738,0.315986,-0.543548,0.213996,0.841246,-0.116884,-0.323292,0.550950,-0.239731,-0.280516,0.854202,0.217738,-0.018328,-0.046879,0.287400,-1.444939,1.489554,-0.303755,-2.347239,-0.758627,0.095142,1.406535,-0.911899,0.942377,-0.414659,-0.320196,0.766527,0.294204,0.180998,1.166475,1.341847,-1.909245,0.460789,2.683026,0.632906,-0.967694,1.437142,-0.008697,-1.713595,-0.180163,0.496684,0.401125,0.652699,0.093725,1.912181,0.817516,1.744673,-0.777844,0.244250,1.212821,-2.499533,-0.536822,1.362315,-1.146691,-1.459042,0.202051,-0.027561,0.506635,-0.237127,-0.207790,1.082241,0.929660,-0.734271,-1.122312,-0.390899,0.490159,-1.160520,-1.064930,0.096393,0.485541,-0.167559,-0.302032,0.451875,0.552999,-0.581710,-0.347878,0.923931,1.203252,-0.619626,0.896745,0.970448,-1.605802,0.540633,0.306158,0.409182,0.765251,2.377412,-1.768414,-0.830468,1.026016,0.353015,1.813582,1.648384,-1.076458,-1.830149,1.290088,-0.321280,0.522018,-0.720160,0.412308,-0.568570,0.661536,0.975841,-1.172335,-1.142428,0.778279,1.526078,-0.422920,-0.352252,0.870726,0.717254,0.914852,-2.028362,1.030640,-0.449103,1.341154,0.661125,0.397046,0.040657,0.547520,0.809972,2.138502,-0.156870,-0.960967,-0.624864,-1.480305,0.168508,-1.053102,-0.174775,-0.381758,-1.304852,-0.057081,-0.449257,0.327530,0.949275,-0.580798,1.915294,-0.482811,-0.658981,0.147835,0.173247,0.541139,0.277799,-0.653735,-1.168723,0.540364,-0.301207,0.647755,-0.480653,0.428893,-1.005869,1.309362,0.235993,0.652125,0.717441,0.875136,0.156760,-0.231497,-0.630515,-0.362267,-0.505543,-1.540877,0.639517,-1.229394,0.392575,-0.091539,-0.698654,-0.317628,0.836837,-0.299131,0.790683,-1.044736,-0.835173,-0.278861,2.287829,1.395450,-0.300536,0.613385,0.609625,0.061141,-1.193306,-0.203143,-0.080978,-0.270965,1.301840,-0.760252,0.832771,1.768992,2.538349,-0.899869,-0.116571,-0.348267,-1.275955,0.245192,0.166728,0.320985,-0.500035,1.682851,0.782335,0.216706,0.646971,-0.499965,0.540870,-0.899950,-0.593642,-0.693595,-0.694605,1.510582,-1.323334,0.634745,0.553090,1.412561,0.617035,1.472513,-2.156491,1.623382,0.716471,0.568394,2.436584,-1.398122,-0.353623,0.383024,-1.262565,-0.285686,0.436375,1.281458,-0.461883,0.164010,0.128340,0.067454,-0.960645,1.502383,0.612702,-2.275102,1.689399,1.062433,1.337289,-1.206029,0.302407,0.178870,0.046435,0.412035,1.110424,-0.462422,-0.504362,-0.809738,0.883617,-0.282764,-1.442379,-0.187121,-1.633802,0.730376,0.289381,-1.633291,1.282281,0.214105,2.125680,0.433060,0.058320,0.927584,-0.792948,0.405493,-0.989563,-0.409785,0.102108,-1.236818,0.435944,1.152166,1.302508,0.291727,0.761200,0.490752,0.395316,0.415469,-0.582631,0.876803,0.054046,-0.092121,-0.574134,-0.110178,-1.550514,-0.363781,-1.828836,-0.503539,1.196251,0.214686,0.896747,-1.146508,1.409912,0.987695,1.193307,-0.586126,-0.870563,-0.654769,0.222614,0.194407,0.163036,-0.244055,-0.195212,1.572398,0.171586,-0.599272,1.384499,1.233297,0.120283,2.010772,0.504732,0.673699,-1.662543,0.392935,1.632057,0.744900,-0.497688,-0.296348,0.779451,-0.414892,-0.632707,-0.219189,-0.050531,0.560491,-0.062139,-0.589589,-0.062727,0.610305,-1.036843,0.025554,-0.400897,-0.669113,1.943684,0.194551,-1.532190,-0.828155,-0.106672,-1.496919,0.384707,0.358459,1.611991,-0.879767,-1.755775,-0.420345,1.199028,0.853541,0.448921,0.059072,-0.857103,0.308299,-0.513848,-0.400323,-1.084698,0.279785,-1.336852,0.574521,-0.687829,-0.904834,0.696367,0.036223,-0.075449,-0.320804,-0.257358,-0.153945,0.801704,-1.853008,-0.363258,-1.466947,-0.169874,-0.938247,0.796368,-0.671802,0.226819,0.051220,-1.473846,0.281841,0.331881,-0.404182,-0.112716,-0.364631,-0.473237,-0.784415,0.749542,-0.275199,1.053305,-0.207303,-1.020583,-1.625803,-0.191668,1.674216,-0.671190,0.575629,1.098929,-0.774466,-0.041663,1.139306,2.365225,-0.725798,-0.038824,1.771020,2.184241,-0.364963,-0.570764,0.241120,-0.748877,0.270378,-3.072989,-1.964752,-0.865815,0.124988,1.186659,-0.778094,0.147189,0.786782,-0.615507,-0.425868,-0.482231,-0.866485,0.088089,0.221273,0.809881,0.117271,0.494233,0.754686,-0.936326,-0.652771,0.626279,2.605196,0.180664,0.530101,0.790702,-1.063561,2.295666,1.408907,1.314155,0.636140,0.647448,-0.421847,-0.789656,2.730378,0.716343,0.174340,0.991440,-0.291910,-1.269087,0.477227,-0.286685,0.972375,1.266528,-0.952068,0.287721,0.552978,2.752558,-0.534099,-1.455067,0.793178,-1.034425,-0.942666,1.422961,-0.296165,-1.005571,-0.215656,1.077140,0.458445,0.497981,-0.071320,-0.197343,0.256981,-0.251169,0.854043,0.003226,-0.423429,0.138318,1.927758,-1.742349,-0.898377,1.339555,1.341884,0.006332,0.564296,0.433987,-0.152611,0.776842,1.755289,2.789081,-0.938301,0.405605,-0.974240,-0.204570,0.389146,0.365617,0.361587,-1.907066,-0.176248,0.205305,0.156245,-0.969140,-0.988435,0.686481,1.582621,0.520144,0.033688,-2.259840,0.931491,0.727572,0.161364,-1.419348,-1.146364,-2.201522,-1.156001,3.526678,-0.351889,-0.364993,-0.243750,1.192930,1.597254,0.208716,1.817943,-0.854934,2.729230,-1.092245,0.458282,-0.564377,0.825264};
    // constexpr float C_Matlab[M * K] = {1.345760,-0.445204,-3.818317,-7.071449,-2.962649,-0.439156,2.548880,-0.018196,0.528865,2.397087,-4.254694,0.488440,3.091365,2.156695,1.005222,5.516024,5.879253,-0.701959,-0.409589,3.445267,0.244595,-0.477863,-4.517088,-6.635486,-2.851521,0.492737,1.964499,2.763880,1.105394,1.871776,-5.307483,2.445226,1.663239,-1.505096,-0.468993,6.648451,6.919802,2.788480,0.521769,4.636955,-0.092956,-3.720236,-4.211475,-6.354866,-1.712793,1.300743,7.075634,-0.289556,2.813224,2.043292,-1.720242,3.579487,1.574915,-1.999236,-0.459376,9.922272,2.713548,-0.429888,0.003724,2.956175,-0.599332,-2.962303,-5.740487,-7.343363,-1.859670,2.863567,4.171525,1.690147,4.283063,2.184314,-4.270372,1.322532,-0.029939,-2.193159,2.945593,5.722311,5.822718,1.596868,0.614168,7.093074,2.405849,-1.812575,-4.470653,-5.209097,-2.484073,1.325553,3.417810,-0.295977,1.647757,2.142857,-4.976914,0.628519,2.259905,-2.020041,1.626210,5.167320,6.682544,1.424811,-2.126158,3.767716,-0.586026,-2.969169,-6.138661,-8.997524,-3.581949,7.032939,4.818201,-1.940008,5.303719,6.782180,-2.210786,4.496804,2.905834,-2.815321,2.889565,11.085191,6.758179,2.722786,-2.490343,6.546255,-0.076370,-0.694186,-4.675801,-5.053861,0.466741,0.467796,4.788959,-1.188672,3.876452,2.578478,-6.205358,3.236892,3.016353,-4.215972,-0.104144,7.445860,5.890336,-0.558422,0.739564,3.344817,0.378777,-3.953086,-2.740330,-5.903536,-0.121235,5.332105,4.061218,2.695852,6.254270,3.024727,-2.423898,3.609776,1.336833,-3.600400,1.850226,10.814178,5.441546,2.319990,0.739041,3.964396,2.331643,1.070832,-6.250754,-4.721246,-3.749353,-1.142321,6.862478,-0.640946,2.251930,2.757474,-2.504283,2.274251,-0.898213,-0.848626,0.151687,6.162573,7.378494,1.249832,-2.762426,4.179531,0.199258,-1.148828,-5.477040,-8.088164,-3.427472,4.011926,3.980451,1.508168,6.743257,3.150457,0.554059,0.964285,0.521278,0.021978,2.237925,6.752538,8.295572,2.528622,-5.627674,4.835286,3.755869,-2.777474,-0.167612,-4.334915,-2.647611,2.830276,6.392189,0.211307,4.070575,4.553881,-4.954081,3.576782,1.137465,-2.139548,1.916852,9.261957,6.346883,1.742648,-4.068094,4.076083,-2.841688,-2.194540,-2.831188,-5.703318,-0.128937,3.652199,4.119833,0.235635,5.762571,2.515619,-3.113007,2.891446,0.490035,-3.418658,4.981602,7.448514,5.944544,0.517191,-1.964296,5.723610,-1.022172,-2.773105,-6.970033,-7.459031,-1.247338,1.977320,5.367422,-1.504587,3.979340,4.259752,-4.728564,0.106209,-0.158769,-1.623363,2.539200,7.224172,4.446808,-0.644335,0.439936,7.309030,-0.715988,0.504517,-3.862616,-6.359638,1.610877,0.157545,3.597874,-0.200050,5.023803,2.879389,-3.359409,1.552256,0.283059,-2.434287,4.109738,6.296660,6.576332,1.854382,-2.766123,3.720479,-0.481908,-0.950001,-5.572354,-4.617050,-0.817475,1.559366,7.130622,-3.492194,5.214360,5.475186,-5.865790,-0.752974,1.922469,-0.883356,0.070602,8.873040,5.879740,0.872799,-1.090073,7.010184,-0.195483,-2.766362,-2.666138,-1.723362,-2.384783,1.112878,5.533151,-0.447352,6.266007,3.657306,-1.289249,1.960052,1.886466,-0.968540,0.150260,9.526908,1.288673,0.672858,-0.544772,2.985897,2.711884,-1.583626,-5.100007,-5.263050,-2.798309,2.113805,5.235039,1.458644,1.878938,2.668076,-1.753228,1.096741,2.135887,-0.003620,0.846608,6.661644,6.415336,3.626228,-4.708508,5.390811,-1.326920,-2.319596,-6.162217,-7.157856,-2.618475,5.550494,2.775111,-0.258573,6.125362,4.520704,-3.620576,-0.155859,2.241137,-0.702079,1.929843,6.701456,6.750952,1.965016,-2.610636,7.317395,1.735065,-0.940765,-0.512012,-3.511472,-2.028010,-0.196793,2.113285,-0.560079,-0.177933,0.631992,-4.418488,2.199284,0.876753,-2.268795,1.427826,5.306685,7.042498,1.086380,0.405567,2.816671,-0.055701,-0.389130,-5.851556,-5.862928,-1.295746,1.828769,7.065789,-3.631787,1.847077,4.753578,-0.022325,3.098745,2.079338,1.169604,0.126212,8.982183,5.571905,0.754399,-1.490074,4.943506,0.399031,0.429051,-4.859672,-6.413958,-4.246317,2.060524,6.171733,1.764856,5.250731,3.532447,-0.623754,2.859708,-0.495043,0.598131,2.335803,9.698425,6.635998,1.408039,-3.265816,5.764218,-0.700987,-1.631853,-3.484339,-6.350687,-3.681504,5.678374,4.078689,0.421867,1.960467,5.763502,-1.152368,5.994359,2.881854,0.999346,1.171615,10.311854,3.124194,0.918925,-0.884798,4.855608,-0.948022,-3.295305,-2.612484,-1.973828,-0.049434,2.623005,6.176755,-0.853379,5.749740,5.325172,-1.637518,-1.132431,1.198618,-2.196430,1.210761,7.567923,7.793386,1.082332,-1.941502,4.858668,0.807849,2.063232,-4.127200,-3.994300,-2.402945,2.353823,6.086142,-0.795167,-0.487933,1.885520,-0.318516,4.463936,1.038881,-1.221152,1.981562,4.929876,7.403000,0.681313,-5.269679,4.948593};
    // float C[M * K];
    // float C_MatrixUtil[M * K];

    // float A_vec[vectorSize] = {0.814724,0.905792,0.126987,0.913376,0.632359,0.097540,0.278498,0.546882,0.957507,0.964889,0.157613,0.970593,0.957167,0.485376,0.800280,0.141886};
    // float B_matrix[vectorSize * vectorSize] = {-0.124144,-1.068870,1.093266,1.544212,1.489698,-0.809499,1.109273,0.085931,1.409034,-2.944284,-0.863653,-1.491590,1.417192,1.438380,0.077359,-0.742302,0.671497,0.325191,-1.214117,-1.061582,-1.207487,-0.754928,-1.113501,2.350457,0.717239,1.370299,-0.006849,-0.615602,1.630235,-1.711516,1.532630,0.748077,0.488894,-0.102242,-0.769666,-0.192419,1.034693,-0.241447,0.371379,0.888610,0.726885,0.319207,-0.225584,-0.764849,-0.303441,0.312859,1.117356,-1.402269,0.293871,-0.864880,-1.089064,-1.422376,-0.787283,-0.030051,0.032557,0.488194,0.888396,-0.164879,0.552527,-0.177375,-1.147070,0.627707,1.100610,-0.196053};
    // constexpr float C_vecResult[vectorSize] = {5.853615,-1.948750,2.058402,-1.838518};
    // float C_MatrixUtilvec[vectorSize];
    // float C_vec[vectorSize];
    //
    double A[M * N] = {0.814724, 0.678735, 0.709365, 0.814285, 0.568824, 0.106653, 0.401808, 0.575209, 0.486792, 0.225922, 0.085516, 0.098712, 0.805489, 0.972975, 0.372410, 0.032601, 0.824376, 0.068806, 0.637709, 0.322472, 0.647618, 0.318074, 0.192028, 0.683416, 0.768854, 0.850713, 0.083483, 0.123084, 0.105709, 0.467068, 0.178117, 0.441722, 0.905792, 0.757740, 0.754687, 0.243525, 0.469391, 0.961898, 0.075967, 0.059780, 0.435859, 0.170708, 0.262482, 0.261871, 0.576722, 0.648991, 0.198118, 0.561200, 0.982663, 0.319600, 0.957694, 0.784739, 0.679017, 0.119215, 0.138874, 0.704047, 0.167254, 0.560560, 0.625960, 0.205494, 0.142041, 0.648198, 0.359635, 0.013283, 0.126987, 0.743132, 0.276025, 0.929264, 0.011902, 0.004634, 0.239916, 0.234780, 0.446784, 0.227664, 0.801015, 0.335357, 0.182922, 0.800331, 0.489688, 0.881867, 0.730249, 0.530864, 0.240707, 0.471357, 0.635787, 0.939829, 0.696266, 0.442305, 0.861980, 0.929609, 0.660945, 0.146515, 0.166460, 0.025228, 0.056705, 0.897191, 0.913376, 0.392227, 0.679703, 0.349984, 0.337123, 0.774910, 0.123319, 0.353159, 0.306349, 0.435699, 0.029220, 0.679728, 0.239932, 0.453798, 0.339493, 0.669175, 0.343877, 0.654446, 0.676122, 0.035763, 0.945174, 0.645552, 0.093820, 0.019578, 0.989872, 0.696667, 0.729752, 0.189072, 0.620959, 0.842207, 0.521886, 0.196658, 0.632359, 0.655478, 0.655098, 0.196595, 0.162182, 0.817303, 0.183908, 0.821194, 0.508509, 0.311102, 0.928854, 0.136553, 0.886512, 0.432392, 0.951630, 0.190433, 0.584069, 0.407619, 0.289065, 0.175874, 0.208935, 0.479463, 0.525404, 0.330858, 0.514423, 0.582791, 0.890752, 0.042652, 0.573710, 0.559033, 0.335849, 0.093371, 0.097540, 0.171187, 0.162612, 0.251084, 0.794285, 0.868695, 0.239953, 0.015403, 0.510772, 0.923380, 0.730331, 0.721227, 0.028674, 0.825314, 0.920332, 0.368917, 0.107769, 0.819981, 0.671808, 0.721758, 0.709282, 0.639317, 0.530344, 0.424309, 0.884281, 0.815397, 0.982303, 0.635198, 0.052078, 0.854100, 0.175669, 0.307367, 0.278498, 0.706046, 0.118998, 0.616045, 0.311215, 0.084436, 0.417267, 0.043024, 0.817628, 0.430207, 0.488609, 0.106762, 0.489901, 0.083470, 0.052677, 0.460726, 0.906308, 0.718359, 0.695140, 0.473486, 0.236231, 0.544716, 0.861140, 0.270270, 0.588026, 0.879014, 0.769029, 0.281867, 0.931201, 0.347879, 0.208947, 0.456058, 0.546882, 0.031833, 0.498364, 0.473289, 0.528533, 0.399783, 0.049654, 0.168990, 0.794831, 0.184816, 0.578525, 0.653757, 0.167927, 0.133171, 0.737858, 0.981638, 0.879654, 0.968649, 0.067993, 0.152721, 0.119396, 0.647311, 0.484853, 0.197054, 0.154752, 0.988912, 0.581446, 0.538597, 0.728662, 0.446027, 0.905154, 0.101669, 0.957507, 0.276923, 0.959744, 0.351660, 0.165649, 0.259870, 0.902716, 0.649115, 0.644318, 0.904881, 0.237284, 0.494174, 0.978681, 0.173389, 0.269119, 0.156405, 0.817761, 0.531334, 0.254790, 0.341125, 0.607304, 0.543886, 0.393456, 0.821721, 0.199863, 0.000522, 0.928313, 0.695163, 0.737842, 0.054239, 0.675391, 0.995390, 0.964889, 0.046171, 0.340386, 0.830829, 0.601982, 0.800068, 0.944787, 0.731722, 0.378609, 0.979748, 0.458849, 0.779052, 0.712694, 0.390938, 0.422836, 0.855523, 0.260728, 0.325146, 0.224040, 0.607389, 0.450138, 0.721047, 0.671431, 0.429921, 0.406955, 0.865439, 0.580090, 0.499116, 0.063405, 0.177108, 0.468468, 0.332093, 0.157613, 0.097132, 0.585268, 0.585264, 0.262971, 0.431414, 0.490864, 0.647746, 0.811580, 0.438870, 0.963089, 0.715037, 0.500472, 0.831380, 0.547871, 0.644765, 0.594356, 0.105629, 0.667833, 0.191745, 0.458725, 0.522495, 0.741258, 0.887771, 0.748706, 0.612566, 0.016983, 0.535801, 0.860441, 0.662808, 0.912132, 0.297347, 0.970593, 0.823458, 0.223812, 0.549724, 0.654079, 0.910648, 0.489253, 0.450924, 0.532826, 0.111119, 0.546806, 0.903721, 0.471088, 0.803364, 0.942737, 0.376272, 0.022513, 0.610959, 0.844392, 0.738427, 0.661945, 0.993705, 0.520052, 0.391183, 0.825584, 0.989950, 0.120860, 0.445183, 0.934405, 0.330829, 0.104012, 0.062045, 0.957167, 0.694829, 0.751267, 0.917194, 0.689215, 0.181847, 0.337719, 0.547009, 0.350727, 0.258065, 0.521136, 0.890923, 0.059619, 0.060471, 0.417744, 0.190924, 0.425259, 0.778802, 0.344462, 0.242850, 0.770286, 0.218677, 0.347713, 0.769114, 0.789963, 0.527680, 0.862711, 0.123932, 0.984398, 0.898486, 0.745546, 0.298244, 0.485376, 0.317099, 0.255095, 0.285839, 0.748152, 0.263803, 0.900054, 0.296321, 0.939002, 0.408720, 0.231594, 0.334163, 0.681972, 0.399258, 0.983052, 0.428253, 0.312719, 0.423453, 0.780520, 0.917424, 0.350218, 0.105798, 0.149997, 0.396792, 0.318524, 0.479523, 0.484297, 0.490357, 0.858939, 0.118155, 0.736267, 0.046351, 0.800280, 0.950222, 0.505957, 0.757200, 0.450542, 0.145539, 0.369247, 0.744693, 0.875943, 0.594896, 0.488898, 0.698746, 0.042431, 0.526876, 0.301455, 0.482022, 0.161485, 0.090823, 0.675332, 0.269062, 0.662010, 0.109697, 0.586092, 0.808514, 0.534064, 0.801348, 0.844856, 0.852998, 0.785559, 0.988418, 0.561861, 0.505428, 0.141886, 0.034446, 0.699077, 0.753729, 0.083821, 0.136069, 0.111203, 0.188955, 0.550156, 0.262212, 0.624060, 0.197810, 0.071445, 0.416799, 0.701099, 0.120612, 0.178766, 0.266471, 0.006715, 0.765500, 0.416159, 0.063591, 0.262145, 0.755077, 0.089951, 0.227843, 0.209405, 0.873927, 0.513377, 0.539982, 0.184194, 0.761426, 0.421761, 0.438744, 0.890903, 0.380446, 0.228977, 0.869292, 0.780252, 0.686775, 0.622475, 0.602843, 0.679136, 0.030541, 0.521650, 0.656860, 0.666339, 0.589507, 0.422886, 0.153657, 0.602170, 0.188662, 0.841929, 0.404580, 0.044454, 0.377396, 0.111706, 0.498094, 0.552291, 0.270294, 0.177602, 0.706917, 0.597211, 0.631070, 0.915736, 0.381558, 0.959291, 0.567822, 0.913337, 0.579705, 0.389739, 0.183511, 0.587045, 0.711216, 0.395515, 0.744074, 0.096730, 0.627973, 0.539126, 0.226188, 0.094229, 0.281005, 0.386771, 0.287498, 0.832917, 0.448373, 0.754933, 0.216019, 0.136293, 0.900852, 0.629883, 0.208461, 0.398589, 0.999492, 0.299937, 0.089892, 0.792207, 0.765517, 0.547216, 0.075854, 0.152378, 0.549860, 0.241691, 0.368485, 0.207742, 0.221747, 0.367437, 0.500022, 0.818149, 0.291984, 0.698106, 0.384619, 0.598524, 0.440085, 0.915991, 0.091113, 0.256441, 0.365816, 0.242785, 0.790407, 0.678652, 0.574661, 0.031991, 0.564980, 0.133931, 0.287849, 0.134123, 0.080862, 0.959492, 0.795200, 0.138624, 0.053950, 0.825817, 0.144955, 0.403912, 0.625619, 0.301246, 0.117418, 0.987982, 0.479922, 0.817547, 0.431651, 0.666528, 0.582986, 0.470924, 0.527143, 0.001151, 0.576209, 0.613461, 0.763505, 0.442402, 0.949304, 0.495177, 0.845178, 0.614713, 0.640312, 0.030890, 0.414523, 0.212602, 0.777241, 0.655741, 0.186873, 0.149294, 0.530798, 0.538342, 0.853031, 0.096455, 0.780227, 0.470923, 0.296676, 0.037739, 0.904722, 0.722440, 0.015487, 0.178132, 0.251806, 0.695949, 0.457424, 0.462449, 0.683363, 0.582249, 0.627896, 0.687796, 0.327565, 0.189710, 0.738640, 0.362411, 0.417029, 0.939142, 0.464840, 0.894942, 0.905135, 0.035712, 0.489764, 0.257508, 0.779167, 0.996135, 0.622055, 0.131973, 0.081126, 0.230488, 0.318778, 0.885168, 0.609867, 0.149865, 0.984064, 0.128014, 0.290441, 0.699888, 0.875372, 0.424349, 0.546593, 0.540739, 0.771980, 0.359228, 0.671264, 0.495006, 0.585987, 0.049533, 0.205976, 0.301306, 0.763957, 0.071453, 0.533772, 0.849129, 0.445586, 0.840717, 0.934011, 0.078176, 0.350952, 0.942051, 0.929386, 0.844309, 0.424167, 0.913287, 0.617666, 0.659605, 0.167168, 0.999080, 0.617091, 0.638531, 0.518052, 0.460916, 0.425729, 0.869941, 0.932854, 0.736340, 0.438645, 0.147608, 0.246735, 0.489570, 0.947933, 0.295534, 0.818204, 0.242487, 0.109154, 0.933993, 0.646313, 0.254282, 0.129906, 0.442678, 0.513250, 0.956135, 0.775713, 0.194764, 0.507858, 0.796184, 0.859442, 0.518595, 0.106216, 0.171121, 0.265281, 0.033604, 0.943623, 0.770160, 0.644443, 0.264779, 0.972741, 0.394707, 0.833501, 0.054974, 0.666416, 0.192510, 0.082071, 0.332936, 0.100222, 0.053754, 0.825809};
    double B[N * K] = {-0.592656, -0.773064, -0.268183, -0.729445, 0.547640, -0.774513, 0.039740, -0.112437, 0.269541, -0.848110, -0.897601, -0.802823, 0.112440, -0.618593, -0.374437, -1.075235, 0.303564, -0.225794, 1.281631, 0.901491, -0.469809, 0.836634, -0.409873, 1.147328, 1.565084, -1.393273, -0.450599, -1.556594, -2.564449, -0.764753, -0.792337, -1.265636, -0.308625, 0.512016, -1.451741, -0.090967, -0.790258, -0.404922, 0.620090, 0.394676, 0.886377, -1.128330, -0.711323, 0.597865, -1.693344, -0.386235, 0.109248, 1.915102, 0.465864, -1.127695, -0.952975, -0.149331, 0.456660, 0.011354, -0.618682, -0.252772, 0.803380, 0.527859, -0.286674, 0.004854, -1.385220, -1.424470, 0.061445, -1.281281, -0.449397, 0.525586, -0.250553, 0.609846, 1.853561, 0.078189, 0.353905, -1.636447, -0.275101, -0.043989, 0.934501, 1.194824, -1.319903, -1.006963, 0.598016, 0.436919, -1.956754, 0.717442, -1.846129, -2.203264, -0.084292, 1.523269, -0.189902, -0.647912, 1.039289, 2.106630, 1.597026, 0.017344, 0.443144, 2.949093, 1.055929, 0.606064, -0.273846, 1.089027, -0.245533, 1.130073, 0.420684, -0.777906, -0.398333, -0.571246, -1.991997, 1.798494, -1.032914, 2.617335, 0.910897, -0.715847, 0.527470, 0.828387, -0.134765, -0.630046, 0.160227, 0.540514, 0.271867, 1.784875, -1.780737, 0.153771, 0.400738, 0.315986, -0.543548, 0.213996, 0.841246, -0.116884, -0.323292, 0.550950, -0.239731, -0.280516, 0.854202, 0.217738, -0.018328, -0.046879, 0.287400, -1.444939, 1.489554, -0.303755, -2.347239, -0.758627, 0.095142, 1.406535, -0.911899, 0.942377, -0.414659, -0.320196, 0.766527, 0.294204, 0.180998, 1.166475, 1.341847, -1.909245, 0.460789, 2.683026, 0.632906, -0.967694, 1.437142, -0.008697, -1.713595, -0.180163, 0.496684, 0.401125, 0.652699, 0.093725, 1.912181, 0.817516, 1.744673, -0.777844, 0.244250, 1.212821, -2.499533, -0.536822, 1.362315, -1.146691, -1.459042, 0.202051, -0.027561, 0.506635, -0.237127, -0.207790, 1.082241, 0.929660, -0.734271, -1.122312, -0.390899, 0.490159, -1.160520, -1.064930, 0.096393, 0.485541, -0.167559, -0.302032, 0.451875, 0.552999, -0.581710, -0.347878, 0.923931, 1.203252, -0.619626, 0.896745, 0.970448, -1.605802, 0.540633, 0.306158, 0.409182, 0.765251, 2.377412, -1.768414, -0.830468, 1.026016, 0.353015, 1.813582, 1.648384, -1.076458, -1.830149, 1.290088, -0.321280, 0.522018, -0.720160, 0.412308, -0.568570, 0.661536, 0.975841, -1.172335, -1.142428, 0.778279, 1.526078, -0.422920, -0.352252, 0.870726, 0.717254, 0.914852, -2.028362, 1.030640, -0.449103, 1.341154, 0.661125, 0.397046, 0.040657, 0.547520, 0.809972, 2.138502, -0.156870, -0.960967, -0.624864, -1.480305, 0.168508, -1.053102, -0.174775, -0.381758, -1.304852, -0.057081, -0.449257, 0.327530, 0.949275, -0.580798, 1.915294, -0.482811, -0.658981, 0.147835, 0.173247, 0.541139, 0.277799, -0.653735, -1.168723, 0.540364, -0.301207, 0.647755, -0.480653, 0.428893, -1.005869, 1.309362, 0.235993, 0.652125, 0.717441, 0.875136, 0.156760, -0.231497, -0.630515, -0.362267, -0.505543, -1.540877, 0.639517, -1.229394, 0.392575, -0.091539, -0.698654, -0.317628, 0.836837, -0.299131, 0.790683, -1.044736, -0.835173, -0.278861, 2.287829, 1.395450, -0.300536, 0.613385, 0.609625, 0.061141, -1.193306, -0.203143, -0.080978, -0.270965, 1.301840, -0.760252, 0.832771, 1.768992, 2.538349, -0.899869, -0.116571, -0.348267, -1.275955, 0.245192, 0.166728, 0.320985, -0.500035, 1.682851, 0.782335, 0.216706, 0.646971, -0.499965, 0.540870, -0.899950, -0.593642, -0.693595, -0.694605, 1.510582, -1.323334, 0.634745, 0.553090, 1.412561, 0.617035, 1.472513, -2.156491, 1.623382, 0.716471, 0.568394, 2.436584, -1.398122, -0.353623, 0.383024, -1.262565, -0.285686, 0.436375, 1.281458, -0.461883, 0.164010, 0.128340, 0.067454, -0.960645, 1.502383, 0.612702, -2.275102, 1.689399, 1.062433, 1.337289, -1.206029, 0.302407, 0.178870, 0.046435, 0.412035, 1.110424, -0.462422, -0.504362, -0.809738, 0.883617, -0.282764, -1.442379, -0.187121, -1.633802, 0.730376, 0.289381, -1.633291, 1.282281, 0.214105, 2.125680, 0.433060, 0.058320, 0.927584, -0.792948, 0.405493, -0.989563, -0.409785, 0.102108, -1.236818, 0.435944, 1.152166, 1.302508, 0.291727, 0.761200, 0.490752, 0.395316, 0.415469, -0.582631, 0.876803, 0.054046, -0.092121, -0.574134, -0.110178, -1.550514, -0.363781, -1.828836, -0.503539, 1.196251, 0.214686, 0.896747, -1.146508, 1.409912, 0.987695, 1.193307, -0.586126, -0.870563, -0.654769, 0.222614, 0.194407, 0.163036, -0.244055, -0.195212, 1.572398, 0.171586, -0.599272, 1.384499, 1.233297, 0.120283, 2.010772, 0.504732, 0.673699, -1.662543, 0.392935, 1.632057, 0.744900, -0.497688, -0.296348, 0.779451, -0.414892, -0.632707, -0.219189, -0.050531, 0.560491, -0.062139, -0.589589, -0.062727, 0.610305, -1.036843, 0.025554, -0.400897, -0.669113, 1.943684, 0.194551, -1.532190, -0.828155, -0.106672, -1.496919, 0.384707, 0.358459, 1.611991, -0.879767, -1.755775, -0.420345, 1.199028, 0.853541, 0.448921, 0.059072, -0.857103, 0.308299, -0.513848, -0.400323, -1.084698, 0.279785, -1.336852, 0.574521, -0.687829, -0.904834, 0.696367, 0.036223, -0.075449, -0.320804, -0.257358, -0.153945, 0.801704, -1.853008, -0.363258, -1.466947, -0.169874, -0.938247, 0.796368, -0.671802, 0.226819, 0.051220, -1.473846, 0.281841, 0.331881, -0.404182, -0.112716, -0.364631, -0.473237, -0.784415, 0.749542, -0.275199, 1.053305, -0.207303, -1.020583, -1.625803, -0.191668, 1.674216, -0.671190, 0.575629, 1.098929, -0.774466, -0.041663, 1.139306, 2.365225, -0.725798, -0.038824, 1.771020, 2.184241, -0.364963, -0.570764, 0.241120, -0.748877, 0.270378, -3.072989, -1.964752, -0.865815, 0.124988, 1.186659, -0.778094, 0.147189, 0.786782, -0.615507, -0.425868, -0.482231, -0.866485, 0.088089, 0.221273, 0.809881, 0.117271, 0.494233, 0.754686, -0.936326, -0.652771, 0.626279, 2.605196, 0.180664, 0.530101, 0.790702, -1.063561, 2.295666, 1.408907, 1.314155, 0.636140, 0.647448, -0.421847, -0.789656, 2.730378, 0.716343, 0.174340, 0.991440, -0.291910, -1.269087, 0.477227, -0.286685, 0.972375, 1.266528, -0.952068, 0.287721, 0.552978, 2.752558, -0.534099, -1.455067, 0.793178, -1.034425, -0.942666, 1.422961, -0.296165, -1.005571, -0.215656, 1.077140, 0.458445, 0.497981, -0.071320, -0.197343, 0.256981, -0.251169, 0.854043, 0.003226, -0.423429, 0.138318, 1.927758, -1.742349, -0.898377, 1.339555, 1.341884, 0.006332, 0.564296, 0.433987, -0.152611, 0.776842, 1.755289, 2.789081, -0.938301, 0.405605, -0.974240, -0.204570, 0.389146, 0.365617, 0.361587, -1.907066, -0.176248, 0.205305, 0.156245, -0.969140, -0.988435, 0.686481, 1.582621, 0.520144, 0.033688, -2.259840, 0.931491, 0.727572, 0.161364, -1.419348, -1.146364, -2.201522, -1.156001, 3.526678, -0.351889, -0.364993, -0.243750, 1.192930, 1.597254, 0.208716, 1.817943, -0.854934, 2.729230, -1.092245, 0.458282, -0.564377, 0.825264};
    constexpr double C_Matlab[M * K] = {1.345760, -0.445204, -3.818317, -7.071449, -2.962649, -0.439156, 2.548880, -0.018196, 0.528865, 2.397087, -4.254694, 0.488440, 3.091365, 2.156695, 1.005222, 5.516024, 5.879253, -0.701959, -0.409589, 3.445267, 0.244595, -0.477863, -4.517088, -6.635486, -2.851521, 0.492737, 1.964499, 2.763880, 1.105394, 1.871776, -5.307483, 2.445226, 1.663239, -1.505096, -0.468993, 6.648451, 6.919802, 2.788480, 0.521769, 4.636955, -0.092956, -3.720236, -4.211475, -6.354866, -1.712793, 1.300743, 7.075634, -0.289556, 2.813224, 2.043292, -1.720242, 3.579487, 1.574915, -1.999236, -0.459376, 9.922272, 2.713548, -0.429888, 0.003724, 2.956175, -0.599332, -2.962303, -5.740487, -7.343363, -1.859670, 2.863567, 4.171525, 1.690147, 4.283063, 2.184314, -4.270372, 1.322532, -0.029939, -2.193159, 2.945593, 5.722311, 5.822718, 1.596868, 0.614168, 7.093074, 2.405849, -1.812575, -4.470653, -5.209097, -2.484073, 1.325553, 3.417810, -0.295977, 1.647757, 2.142857, -4.976914, 0.628519, 2.259905, -2.020041, 1.626210, 5.167320, 6.682544, 1.424811, -2.126158, 3.767716, -0.586026, -2.969169, -6.138661, -8.997524, -3.581949, 7.032939, 4.818201, -1.940008, 5.303719, 6.782180, -2.210786, 4.496804, 2.905834, -2.815321, 2.889565, 11.085191, 6.758179, 2.722786, -2.490343, 6.546255, -0.076370, -0.694186, -4.675801, -5.053861, 0.466741, 0.467796, 4.788959, -1.188672, 3.876452, 2.578478, -6.205358, 3.236892, 3.016353, -4.215972, -0.104144, 7.445860, 5.890336, -0.558422, 0.739564, 3.344817, 0.378777, -3.953086, -2.740330, -5.903536, -0.121235, 5.332105, 4.061218, 2.695852, 6.254270, 3.024727, -2.423898, 3.609776, 1.336833, -3.600400, 1.850226, 10.814178, 5.441546, 2.319990, 0.739041, 3.964396, 2.331643, 1.070832, -6.250754, -4.721246, -3.749353, -1.142321, 6.862478, -0.640946, 2.251930, 2.757474, -2.504283, 2.274251, -0.898213, -0.848626, 0.151687, 6.162573, 7.378494, 1.249832, -2.762426, 4.179531, 0.199258, -1.148828, -5.477040, -8.088164, -3.427472, 4.011926, 3.980451, 1.508168, 6.743257, 3.150457, 0.554059, 0.964285, 0.521278, 0.021978, 2.237925, 6.752538, 8.295572, 2.528622, -5.627674, 4.835286, 3.755869, -2.777474, -0.167612, -4.334915, -2.647611, 2.830276, 6.392189, 0.211307, 4.070575, 4.553881, -4.954081, 3.576782, 1.137465, -2.139548, 1.916852, 9.261957, 6.346883, 1.742648, -4.068094, 4.076083, -2.841688, -2.194540, -2.831188, -5.703318, -0.128937, 3.652199, 4.119833, 0.235635, 5.762571, 2.515619, -3.113007, 2.891446, 0.490035, -3.418658, 4.981602, 7.448514, 5.944544, 0.517191, -1.964296, 5.723610, -1.022172, -2.773105, -6.970033, -7.459031, -1.247338, 1.977320, 5.367422, -1.504587, 3.979340, 4.259752, -4.728564, 0.106209, -0.158769, -1.623363, 2.539200, 7.224172, 4.446808, -0.644335, 0.439936, 7.309030, -0.715988, 0.504517, -3.862616, -6.359638, 1.610877, 0.157545, 3.597874, -0.200050, 5.023803, 2.879389, -3.359409, 1.552256, 0.283059, -2.434287, 4.109738, 6.296660, 6.576332, 1.854382, -2.766123, 3.720479, -0.481908, -0.950001, -5.572354, -4.617050, -0.817475, 1.559366, 7.130622, -3.492194, 5.214360, 5.475186, -5.865790, -0.752974, 1.922469, -0.883356, 0.070602, 8.873040, 5.879740, 0.872799, -1.090073, 7.010184, -0.195483, -2.766362, -2.666138, -1.723362, -2.384783, 1.112878, 5.533151, -0.447352, 6.266007, 3.657306, -1.289249, 1.960052, 1.886466, -0.968540, 0.150260, 9.526908, 1.288673, 0.672858, -0.544772, 2.985897, 2.711884, -1.583626, -5.100007, -5.263050, -2.798309, 2.113805, 5.235039, 1.458644, 1.878938, 2.668076, -1.753228, 1.096741, 2.135887, -0.003620, 0.846608, 6.661644, 6.415336, 3.626228, -4.708508, 5.390811, -1.326920, -2.319596, -6.162217, -7.157856, -2.618475, 5.550494, 2.775111, -0.258573, 6.125362, 4.520704, -3.620576, -0.155859, 2.241137, -0.702079, 1.929843, 6.701456, 6.750952, 1.965016, -2.610636, 7.317395, 1.735065, -0.940765, -0.512012, -3.511472, -2.028010, -0.196793, 2.113285, -0.560079, -0.177933, 0.631992, -4.418488, 2.199284, 0.876753, -2.268795, 1.427826, 5.306685, 7.042498, 1.086380, 0.405567, 2.816671, -0.055701, -0.389130, -5.851556, -5.862928, -1.295746, 1.828769, 7.065789, -3.631787, 1.847077, 4.753578, -0.022325, 3.098745, 2.079338, 1.169604, 0.126212, 8.982183, 5.571905, 0.754399, -1.490074, 4.943506, 0.399031, 0.429051, -4.859672, -6.413958, -4.246317, 2.060524, 6.171733, 1.764856, 5.250731, 3.532447, -0.623754, 2.859708, -0.495043, 0.598131, 2.335803, 9.698425, 6.635998, 1.408039, -3.265816, 5.764218, -0.700987, -1.631853, -3.484339, -6.350687, -3.681504, 5.678374, 4.078689, 0.421867, 1.960467, 5.763502, -1.152368, 5.994359, 2.881854, 0.999346, 1.171615, 10.311854, 3.124194, 0.918925, -0.884798, 4.855608, -0.948022, -3.295305, -2.612484, -1.973828, -0.049434, 2.623005, 6.176755, -0.853379, 5.749740, 5.325172, -1.637518, -1.132431, 1.198618, -2.196430, 1.210761, 7.567923, 7.793386, 1.082332, -1.941502, 4.858668, 0.807849, 2.063232, -4.127200, -3.994300, -2.402945, 2.353823, 6.086142, -0.795167, -0.487933, 1.885520, -0.318516, 4.463936, 1.038881, -1.221152, 1.981562, 4.929876, 7.403000, 0.681313, -5.269679, 4.948593};
    double C[M * K] = {0};
    double C_MatrixUtil[M * K] = {0};
    // // double A_MatrixUtilTransposed[M * N]{0};
    // // double B_MatrixUtilTransposed[N * K]{0};
    // // double A_Ne10Transposed[M * N]{0};
    // // double B_Ne10Transposed[N * K]{0};
    // //
    // // double A_vec[vectorSize] = {0.814724,0.905792,0.126987,0.913376,0.632359,0.097540,0.278498,0.546882,0.957507,0.964889,0.157613,0.970593,0.957167,0.485376,0.800280,0.141886};
    // // double B_matrix[vectorSize * 4] = {-0.124144,-1.068870,1.093266,1.544212,1.489698,-0.809499,1.109273,0.085931,1.409034,-2.944284,-0.863653,-1.491590,1.417192,1.438380,0.077359,-0.742302,0.671497,0.325191,-1.214117,-1.061582,-1.207487,-0.754928,-1.113501,2.350457,0.717239,1.370299,-0.006849,-0.615602,1.630235,-1.711516,1.532630,0.748077,0.488894,-0.102242,-0.769666,-0.192419,1.034693,-0.241447,0.371379,0.888610,0.726885,0.319207,-0.225584,-0.764849,-0.303441,0.312859,1.117356,-1.402269,0.293871,-0.864880,-1.089064,-1.422376,-0.787283,-0.030051,0.032557,0.488194,0.888396,-0.164879,0.552527,-0.177375,-1.147070,0.627707,1.100610,-0.196053};
    // // constexpr double C_vecResult[4] = {5.853615,-1.948750,2.058402,-1.838518};
    // // double C_MatrixUtilvec[vectorSize];
    // // double C_vec[4];
    //
    // // double A_16x16[vectorSize * vectorSize] = {0.537667,-0.124144,-1.068870,1.093266,1.544212,1.419310,-0.082494,1.354594,0.701541,-0.831367,-0.293754,-0.130285,-0.303108,-0.162338,-1.506160,-1.066701,1.833885,1.489698,-0.809499,1.109273,0.085931,0.291584,-1.933023,-1.072155,-2.051816,-0.979206,-0.847926,0.183689,0.023046,-0.146055,-0.444628,0.933728,-2.258847,1.409034,-2.944284,-0.863653,-1.491590,0.197811,-0.438966,0.960954,-0.353850,-1.156402,-1.120128,-0.476153,0.051290,-0.532011,-0.155941,0.350321,0.862173,1.417192,1.438380,0.077359,-0.742302,1.587699,-1.794679,0.124050,-0.823587,-0.533557,2.526000,0.862022,0.826063,1.682104,0.276068,-0.029006,0.318765,0.671497,0.325191,-1.214117,-1.061582,-0.804466,0.840376,1.436697,-1.577057,-2.002636,1.655498,-1.361694,1.526977,-0.875729,-0.261164,0.182452,-1.307688,-1.207487,-0.754928,-1.113501,2.350457,0.696624,-0.888032,-1.960900,0.507975,0.964229,0.307535,0.455030,0.466914,-0.483815,0.443422,-1.565056,-0.433592,0.717239,1.370299,-0.006849,-0.615602,0.835088,0.100093,-0.197698,0.281984,0.520060,-1.257118,-0.848709,-0.209713,-0.712005,0.391894,-0.084539,0.342624,1.630235,-1.711516,1.532630,0.748077,-0.243715,-0.544529,-1.207845,0.033480,-0.020028,-0.865468,-0.334887,0.625190,-1.174212,-1.250679,1.603946,3.578397,0.488894,-0.102242,-0.769666,-0.192419,0.215670,0.303521,2.908008,-1.333678,-0.034771,-0.176534,0.552783,0.183227,-0.192240,-0.947961,0.098348,2.769437,1.034693,-0.241447,0.371379,0.888610,-1.165844,-0.600327,0.825219,1.127492,-0.798164,0.791416,1.039091,-1.029768,-0.274070,-0.741106,0.041374,-1.349887,0.726885,0.319207,-0.225584,-0.764849,-1.147953,0.489965,1.378972,0.350179,1.018685,-1.332004,-1.117639,0.949222,1.530073,-0.507818,-0.734169,3.034923,-0.303441,0.312859,1.117356,-1.402269,0.104875,0.739363,-1.058180,-0.299066,-0.133217,-2.329867,1.260659,0.307062,-0.249025,-0.320576,-0.030814,0.725404,0.293871,-0.864880,-1.089064,-1.422376,0.722254,1.711888,-0.468616,0.022890,-0.714530,-1.449097,0.660143,0.135175,-1.064213,0.012469,0.232347,-0.063055,-0.787283,-0.030051,0.032557,0.488194,2.585491,-0.194124,-0.272469,-0.261995,1.351386,0.333511,-0.067866,0.515246,1.603457,-3.029177,0.426388,0.714743,0.888396,-0.164879,0.552527,-0.177375,-0.666891,-2.138355,1.098425,-1.750212,-0.224771,0.391354,-0.195221,0.261406,1.234679,-0.457015,-0.372809,-0.204966,-1.147070,0.627707,1.100610,-0.196053,0.187331,-0.839589,-0.277872,-0.285651,-0.589029,0.451679,-0.217606,-0.941486,-0.229626,1.242448,-0.236455};
    // // double A_MatrixUtil_16x16[vectorSize * vectorSize]{0};
    // // double A_Ne10_16x16[vectorSize * vectorSize]{0};
    // // float A[M * N] = {0.814724,0.678735,0.709365,0.814285,0.568824,0.106653,0.401808,0.575209,0.486792,0.225922,0.085516,0.098712,0.805489,0.972975,0.372410,0.032601,0.824376,0.068806,0.637709,0.322472,0.647618,0.318074,0.192028,0.683416,0.768854,0.850713,0.083483,0.123084,0.105709,0.467068,0.178117,0.441722,0.905792,0.757740,0.754687,0.243525,0.469391,0.961898,0.075967,0.059780,0.435859,0.170708,0.262482,0.261871,0.576722,0.648991,0.198118,0.561200,0.982663,0.319600,0.957694,0.784739,0.679017,0.119215,0.138874,0.704047,0.167254,0.560560,0.625960,0.205494,0.142041,0.648198,0.359635,0.013283,0.126987,0.743132,0.276025,0.929264,0.011902,0.004634,0.239916,0.234780,0.446784,0.227664,0.801015,0.335357,0.182922,0.800331,0.489688,0.881867,0.730249,0.530864,0.240707,0.471357,0.635787,0.939829,0.696266,0.442305,0.861980,0.929609,0.660945,0.146515,0.166460,0.025228,0.056705,0.897191,0.913376,0.392227,0.679703,0.349984,0.337123,0.774910,0.123319,0.353159,0.306349,0.435699,0.029220,0.679728,0.239932,0.453798,0.339493,0.669175,0.343877,0.654446,0.676122,0.035763,0.945174,0.645552,0.093820,0.019578,0.989872,0.696667,0.729752,0.189072,0.620959,0.842207,0.521886,0.196658,0.632359,0.655478,0.655098,0.196595,0.162182,0.817303,0.183908,0.821194,0.508509,0.311102,0.928854,0.136553,0.886512,0.432392,0.951630,0.190433,0.584069,0.407619,0.289065,0.175874,0.208935,0.479463,0.525404,0.330858,0.514423,0.582791,0.890752,0.042652,0.573710,0.559033,0.335849,0.093371,0.097540,0.171187,0.162612,0.251084,0.794285,0.868695,0.239953,0.015403,0.510772,0.923380,0.730331,0.721227,0.028674,0.825314,0.920332,0.368917,0.107769,0.819981,0.671808,0.721758,0.709282,0.639317,0.530344,0.424309,0.884281,0.815397,0.982303,0.635198,0.052078,0.854100,0.175669,0.307367,0.278498,0.706046,0.118998,0.616045,0.311215,0.084436,0.417267,0.043024,0.817628,0.430207,0.488609,0.106762,0.489901,0.083470,0.052677,0.460726,0.906308,0.718359,0.695140,0.473486,0.236231,0.544716,0.861140,0.270270,0.588026,0.879014,0.769029,0.281867,0.931201,0.347879,0.208947,0.456058,0.546882,0.031833,0.498364,0.473289,0.528533,0.399783,0.049654,0.168990,0.794831,0.184816,0.578525,0.653757,0.167927,0.133171,0.737858,0.981638,0.879654,0.968649,0.067993,0.152721,0.119396,0.647311,0.484853,0.197054,0.154752,0.988912,0.581446,0.538597,0.728662,0.446027,0.905154,0.101669,0.957507,0.276923,0.959744,0.351660,0.165649,0.259870,0.902716,0.649115,0.644318,0.904881,0.237284,0.494174,0.978681,0.173389,0.269119,0.156405,0.817761,0.531334,0.254790,0.341125,0.607304,0.543886,0.393456,0.821721,0.199863,0.000522,0.928313,0.695163,0.737842,0.054239,0.675391,0.995390,0.964889,0.046171,0.340386,0.830829,0.601982,0.800068,0.944787,0.731722,0.378609,0.979748,0.458849,0.779052,0.712694,0.390938,0.422836,0.855523,0.260728,0.325146,0.224040,0.607389,0.450138,0.721047,0.671431,0.429921,0.406955,0.865439,0.580090,0.499116,0.063405,0.177108,0.468468,0.332093,0.157613,0.097132,0.585268,0.585264,0.262971,0.431414,0.490864,0.647746,0.811580,0.438870,0.963089,0.715037,0.500472,0.831380,0.547871,0.644765,0.594356,0.105629,0.667833,0.191745,0.458725,0.522495,0.741258,0.887771,0.748706,0.612566,0.016983,0.535801,0.860441,0.662808,0.912132,0.297347,0.970593,0.823458,0.223812,0.549724,0.654079,0.910648,0.489253,0.450924,0.532826,0.111119,0.546806,0.903721,0.471088,0.803364,0.942737,0.376272,0.022513,0.610959,0.844392,0.738427,0.661945,0.993705,0.520052,0.391183,0.825584,0.989950,0.120860,0.445183,0.934405,0.330829,0.104012,0.062045,0.957167,0.694829,0.751267,0.917194,0.689215,0.181847,0.337719,0.547009,0.350727,0.258065,0.521136,0.890923,0.059619,0.060471,0.417744,0.190924,0.425259,0.778802,0.344462,0.242850,0.770286,0.218677,0.347713,0.769114,0.789963,0.527680,0.862711,0.123932,0.984398,0.898486,0.745546,0.298244,0.485376,0.317099,0.255095,0.285839,0.748152,0.263803,0.900054,0.296321,0.939002,0.408720,0.231594,0.334163,0.681972,0.399258,0.983052,0.428253,0.312719,0.423453,0.780520,0.917424,0.350218,0.105798,0.149997,0.396792,0.318524,0.479523,0.484297,0.490357,0.858939,0.118155,0.736267,0.046351,0.800280,0.950222,0.505957,0.757200,0.450542,0.145539,0.369247,0.744693,0.875943,0.594896,0.488898,0.698746,0.042431,0.526876,0.301455,0.482022,0.161485,0.090823,0.675332,0.269062,0.662010,0.109697,0.586092,0.808514,0.534064,0.801348,0.844856,0.852998,0.785559,0.988418,0.561861,0.505428,0.141886,0.034446,0.699077,0.753729,0.083821,0.136069,0.111203,0.188955,0.550156,0.262212,0.624060,0.197810,0.071445,0.416799,0.701099,0.120612,0.178766,0.266471,0.006715,0.765500,0.416159,0.063591,0.262145,0.755077,0.089951,0.227843,0.209405,0.873927,0.513377,0.539982,0.184194,0.761426,0.421761,0.438744,0.890903,0.380446,0.228977,0.869292,0.780252,0.686775,0.622475,0.602843,0.679136,0.030541,0.521650,0.656860,0.666339,0.589507,0.422886,0.153657,0.602170,0.188662,0.841929,0.404580,0.044454,0.377396,0.111706,0.498094,0.552291,0.270294,0.177602,0.706917,0.597211,0.631070,0.915736,0.381558,0.959291,0.567822,0.913337,0.579705,0.389739,0.183511,0.587045,0.711216,0.395515,0.744074,0.096730,0.627973,0.539126,0.226188,0.094229,0.281005,0.386771,0.287498,0.832917,0.448373,0.754933,0.216019,0.136293,0.900852,0.629883,0.208461,0.398589,0.999492,0.299937,0.089892,0.792207,0.765517,0.547216,0.075854,0.152378,0.549860,0.241691,0.368485,0.207742,0.221747,0.367437,0.500022,0.818149,0.291984,0.698106,0.384619,0.598524,0.440085,0.915991,0.091113,0.256441,0.365816,0.242785,0.790407,0.678652,0.574661,0.031991,0.564980,0.133931,0.287849,0.134123,0.080862,0.959492,0.795200,0.138624,0.053950,0.825817,0.144955,0.403912,0.625619,0.301246,0.117418,0.987982,0.479922,0.817547,0.431651,0.666528,0.582986,0.470924,0.527143,0.001151,0.576209,0.613461,0.763505,0.442402,0.949304,0.495177,0.845178,0.614713,0.640312,0.030890,0.414523,0.212602,0.777241,0.655741,0.186873,0.149294,0.530798,0.538342,0.853031,0.096455,0.780227,0.470923,0.296676,0.037739,0.904722,0.722440,0.015487,0.178132,0.251806,0.695949,0.457424,0.462449,0.683363,0.582249,0.627896,0.687796,0.327565,0.189710,0.738640,0.362411,0.417029,0.939142,0.464840,0.894942,0.905135,0.035712,0.489764,0.257508,0.779167,0.996135,0.622055,0.131973,0.081126,0.230488,0.318778,0.885168,0.609867,0.149865,0.984064,0.128014,0.290441,0.699888,0.875372,0.424349,0.546593,0.540739,0.771980,0.359228,0.671264,0.495006,0.585987,0.049533,0.205976,0.301306,0.763957,0.071453,0.533772,0.849129,0.445586,0.840717,0.934011,0.078176,0.350952,0.942051,0.929386,0.844309,0.424167,0.913287,0.617666,0.659605,0.167168,0.999080,0.617091,0.638531,0.518052,0.460916,0.425729,0.869941,0.932854,0.736340,0.438645,0.147608,0.246735,0.489570,0.947933,0.295534,0.818204,0.242487,0.109154,0.933993,0.646313,0.254282,0.129906,0.442678,0.513250,0.956135,0.775713,0.194764,0.507858,0.796184,0.859442,0.518595,0.106216,0.171121,0.265281,0.033604,0.943623,0.770160,0.644443,0.264779,0.972741,0.394707,0.833501,0.054974,0.666416,0.192510,0.082071,0.332936,0.100222,0.053754,0.825809};
    // // float B[N * K] = {-0.592656,-0.773064,-0.268183,-0.729445,0.547640,-0.774513,0.039740,-0.112437,0.269541,-0.848110,-0.897601,-0.802823,0.112440,-0.618593,-0.374437,-1.075235,0.303564,-0.225794,1.281631,0.901491,-0.469809,0.836634,-0.409873,1.147328,1.565084,-1.393273,-0.450599,-1.556594,-2.564449,-0.764753,-0.792337,-1.265636,-0.308625,0.512016,-1.451741,-0.090967,-0.790258,-0.404922,0.620090,0.394676,0.886377,-1.128330,-0.711323,0.597865,-1.693344,-0.386235,0.109248,1.915102,0.465864,-1.127695,-0.952975,-0.149331,0.456660,0.011354,-0.618682,-0.252772,0.803380,0.527859,-0.286674,0.004854,-1.385220,-1.424470,0.061445,-1.281281,-0.449397,0.525586,-0.250553,0.609846,1.853561,0.078189,0.353905,-1.636447,-0.275101,-0.043989,0.934501,1.194824,-1.319903,-1.006963,0.598016,0.436919,-1.956754,0.717442,-1.846129,-2.203264,-0.084292,1.523269,-0.189902,-0.647912,1.039289,2.106630,1.597026,0.017344,0.443144,2.949093,1.055929,0.606064,-0.273846,1.089027,-0.245533,1.130073,0.420684,-0.777906,-0.398333,-0.571246,-1.991997,1.798494,-1.032914,2.617335,0.910897,-0.715847,0.527470,0.828387,-0.134765,-0.630046,0.160227,0.540514,0.271867,1.784875,-1.780737,0.153771,0.400738,0.315986,-0.543548,0.213996,0.841246,-0.116884,-0.323292,0.550950,-0.239731,-0.280516,0.854202,0.217738,-0.018328,-0.046879,0.287400,-1.444939,1.489554,-0.303755,-2.347239,-0.758627,0.095142,1.406535,-0.911899,0.942377,-0.414659,-0.320196,0.766527,0.294204,0.180998,1.166475,1.341847,-1.909245,0.460789,2.683026,0.632906,-0.967694,1.437142,-0.008697,-1.713595,-0.180163,0.496684,0.401125,0.652699,0.093725,1.912181,0.817516,1.744673,-0.777844,0.244250,1.212821,-2.499533,-0.536822,1.362315,-1.146691,-1.459042,0.202051,-0.027561,0.506635,-0.237127,-0.207790,1.082241,0.929660,-0.734271,-1.122312,-0.390899,0.490159,-1.160520,-1.064930,0.096393,0.485541,-0.167559,-0.302032,0.451875,0.552999,-0.581710,-0.347878,0.923931,1.203252,-0.619626,0.896745,0.970448,-1.605802,0.540633,0.306158,0.409182,0.765251,2.377412,-1.768414,-0.830468,1.026016,0.353015,1.813582,1.648384,-1.076458,-1.830149,1.290088,-0.321280,0.522018,-0.720160,0.412308,-0.568570,0.661536,0.975841,-1.172335,-1.142428,0.778279,1.526078,-0.422920,-0.352252,0.870726,0.717254,0.914852,-2.028362,1.030640,-0.449103,1.341154,0.661125,0.397046,0.040657,0.547520,0.809972,2.138502,-0.156870,-0.960967,-0.624864,-1.480305,0.168508,-1.053102,-0.174775,-0.381758,-1.304852,-0.057081,-0.449257,0.327530,0.949275,-0.580798,1.915294,-0.482811,-0.658981,0.147835,0.173247,0.541139,0.277799,-0.653735,-1.168723,0.540364,-0.301207,0.647755,-0.480653,0.428893,-1.005869,1.309362,0.235993,0.652125,0.717441,0.875136,0.156760,-0.231497,-0.630515,-0.362267,-0.505543,-1.540877,0.639517,-1.229394,0.392575,-0.091539,-0.698654,-0.317628,0.836837,-0.299131,0.790683,-1.044736,-0.835173,-0.278861,2.287829,1.395450,-0.300536,0.613385,0.609625,0.061141,-1.193306,-0.203143,-0.080978,-0.270965,1.301840,-0.760252,0.832771,1.768992,2.538349,-0.899869,-0.116571,-0.348267,-1.275955,0.245192,0.166728,0.320985,-0.500035,1.682851,0.782335,0.216706,0.646971,-0.499965,0.540870,-0.899950,-0.593642,-0.693595,-0.694605,1.510582,-1.323334,0.634745,0.553090,1.412561,0.617035,1.472513,-2.156491,1.623382,0.716471,0.568394,2.436584,-1.398122,-0.353623,0.383024,-1.262565,-0.285686,0.436375,1.281458,-0.461883,0.164010,0.128340,0.067454,-0.960645,1.502383,0.612702,-2.275102,1.689399,1.062433,1.337289,-1.206029,0.302407,0.178870,0.046435,0.412035,1.110424,-0.462422,-0.504362,-0.809738,0.883617,-0.282764,-1.442379,-0.187121,-1.633802,0.730376,0.289381,-1.633291,1.282281,0.214105,2.125680,0.433060,0.058320,0.927584,-0.792948,0.405493,-0.989563,-0.409785,0.102108,-1.236818,0.435944,1.152166,1.302508,0.291727,0.761200,0.490752,0.395316,0.415469,-0.582631,0.876803,0.054046,-0.092121,-0.574134,-0.110178,-1.550514,-0.363781,-1.828836,-0.503539,1.196251,0.214686,0.896747,-1.146508,1.409912,0.987695,1.193307,-0.586126,-0.870563,-0.654769,0.222614,0.194407,0.163036,-0.244055,-0.195212,1.572398,0.171586,-0.599272,1.384499,1.233297,0.120283,2.010772,0.504732,0.673699,-1.662543,0.392935,1.632057,0.744900,-0.497688,-0.296348,0.779451,-0.414892,-0.632707,-0.219189,-0.050531,0.560491,-0.062139,-0.589589,-0.062727,0.610305,-1.036843,0.025554,-0.400897,-0.669113,1.943684,0.194551,-1.532190,-0.828155,-0.106672,-1.496919,0.384707,0.358459,1.611991,-0.879767,-1.755775,-0.420345,1.199028,0.853541,0.448921,0.059072,-0.857103,0.308299,-0.513848,-0.400323,-1.084698,0.279785,-1.336852,0.574521,-0.687829,-0.904834,0.696367,0.036223,-0.075449,-0.320804,-0.257358,-0.153945,0.801704,-1.853008,-0.363258,-1.466947,-0.169874,-0.938247,0.796368,-0.671802,0.226819,0.051220,-1.473846,0.281841,0.331881,-0.404182,-0.112716,-0.364631,-0.473237,-0.784415,0.749542,-0.275199,1.053305,-0.207303,-1.020583,-1.625803,-0.191668,1.674216,-0.671190,0.575629,1.098929,-0.774466,-0.041663,1.139306,2.365225,-0.725798,-0.038824,1.771020,2.184241,-0.364963,-0.570764,0.241120,-0.748877,0.270378,-3.072989,-1.964752,-0.865815,0.124988,1.186659,-0.778094,0.147189,0.786782,-0.615507,-0.425868,-0.482231,-0.866485,0.088089,0.221273,0.809881,0.117271,0.494233,0.754686,-0.936326,-0.652771,0.626279,2.605196,0.180664,0.530101,0.790702,-1.063561,2.295666,1.408907,1.314155,0.636140,0.647448,-0.421847,-0.789656,2.730378,0.716343,0.174340,0.991440,-0.291910,-1.269087,0.477227,-0.286685,0.972375,1.266528,-0.952068,0.287721,0.552978,2.752558,-0.534099,-1.455067,0.793178,-1.034425,-0.942666,1.422961,-0.296165,-1.005571,-0.215656,1.077140,0.458445,0.497981,-0.071320,-0.197343,0.256981,-0.251169,0.854043,0.003226,-0.423429,0.138318,1.927758,-1.742349,-0.898377,1.339555,1.341884,0.006332,0.564296,0.433987,-0.152611,0.776842,1.755289,2.789081,-0.938301,0.405605,-0.974240,-0.204570,0.389146,0.365617,0.361587,-1.907066,-0.176248,0.205305,0.156245,-0.969140,-0.988435,0.686481,1.582621,0.520144,0.033688,-2.259840,0.931491,0.727572,0.161364,-1.419348,-1.146364,-2.201522,-1.156001,3.526678,-0.351889,-0.364993,-0.243750,1.192930,1.597254,0.208716,1.817943,-0.854934,2.729230,-1.092245,0.458282,-0.564377,0.825264};
    // // constexpr float C_Matlab[M * K] = {1.345760,-0.445204,-3.818317,-7.071449,-2.962649,-0.439156,2.548880,-0.018196,0.528865,2.397087,-4.254694,0.488440,3.091365,2.156695,1.005222,5.516024,5.879253,-0.701959,-0.409589,3.445267,0.244595,-0.477863,-4.517088,-6.635486,-2.851521,0.492737,1.964499,2.763880,1.105394,1.871776,-5.307483,2.445226,1.663239,-1.505096,-0.468993,6.648451,6.919802,2.788480,0.521769,4.636955,-0.092956,-3.720236,-4.211475,-6.354866,-1.712793,1.300743,7.075634,-0.289556,2.813224,2.043292,-1.720242,3.579487,1.574915,-1.999236,-0.459376,9.922272,2.713548,-0.429888,0.003724,2.956175,-0.599332,-2.962303,-5.740487,-7.343363,-1.859670,2.863567,4.171525,1.690147,4.283063,2.184314,-4.270372,1.322532,-0.029939,-2.193159,2.945593,5.722311,5.822718,1.596868,0.614168,7.093074,2.405849,-1.812575,-4.470653,-5.209097,-2.484073,1.325553,3.417810,-0.295977,1.647757,2.142857,-4.976914,0.628519,2.259905,-2.020041,1.626210,5.167320,6.682544,1.424811,-2.126158,3.767716,-0.586026,-2.969169,-6.138661,-8.997524,-3.581949,7.032939,4.818201,-1.940008,5.303719,6.782180,-2.210786,4.496804,2.905834,-2.815321,2.889565,11.085191,6.758179,2.722786,-2.490343,6.546255,-0.076370,-0.694186,-4.675801,-5.053861,0.466741,0.467796,4.788959,-1.188672,3.876452,2.578478,-6.205358,3.236892,3.016353,-4.215972,-0.104144,7.445860,5.890336,-0.558422,0.739564,3.344817,0.378777,-3.953086,-2.740330,-5.903536,-0.121235,5.332105,4.061218,2.695852,6.254270,3.024727,-2.423898,3.609776,1.336833,-3.600400,1.850226,10.814178,5.441546,2.319990,0.739041,3.964396,2.331643,1.070832,-6.250754,-4.721246,-3.749353,-1.142321,6.862478,-0.640946,2.251930,2.757474,-2.504283,2.274251,-0.898213,-0.848626,0.151687,6.162573,7.378494,1.249832,-2.762426,4.179531,0.199258,-1.148828,-5.477040,-8.088164,-3.427472,4.011926,3.980451,1.508168,6.743257,3.150457,0.554059,0.964285,0.521278,0.021978,2.237925,6.752538,8.295572,2.528622,-5.627674,4.835286,3.755869,-2.777474,-0.167612,-4.334915,-2.647611,2.830276,6.392189,0.211307,4.070575,4.553881,-4.954081,3.576782,1.137465,-2.139548,1.916852,9.261957,6.346883,1.742648,-4.068094,4.076083,-2.841688,-2.194540,-2.831188,-5.703318,-0.128937,3.652199,4.119833,0.235635,5.762571,2.515619,-3.113007,2.891446,0.490035,-3.418658,4.981602,7.448514,5.944544,0.517191,-1.964296,5.723610,-1.022172,-2.773105,-6.970033,-7.459031,-1.247338,1.977320,5.367422,-1.504587,3.979340,4.259752,-4.728564,0.106209,-0.158769,-1.623363,2.539200,7.224172,4.446808,-0.644335,0.439936,7.309030,-0.715988,0.504517,-3.862616,-6.359638,1.610877,0.157545,3.597874,-0.200050,5.023803,2.879389,-3.359409,1.552256,0.283059,-2.434287,4.109738,6.296660,6.576332,1.854382,-2.766123,3.720479,-0.481908,-0.950001,-5.572354,-4.617050,-0.817475,1.559366,7.130622,-3.492194,5.214360,5.475186,-5.865790,-0.752974,1.922469,-0.883356,0.070602,8.873040,5.879740,0.872799,-1.090073,7.010184,-0.195483,-2.766362,-2.666138,-1.723362,-2.384783,1.112878,5.533151,-0.447352,6.266007,3.657306,-1.289249,1.960052,1.886466,-0.968540,0.150260,9.526908,1.288673,0.672858,-0.544772,2.985897,2.711884,-1.583626,-5.100007,-5.263050,-2.798309,2.113805,5.235039,1.458644,1.878938,2.668076,-1.753228,1.096741,2.135887,-0.003620,0.846608,6.661644,6.415336,3.626228,-4.708508,5.390811,-1.326920,-2.319596,-6.162217,-7.157856,-2.618475,5.550494,2.775111,-0.258573,6.125362,4.520704,-3.620576,-0.155859,2.241137,-0.702079,1.929843,6.701456,6.750952,1.965016,-2.610636,7.317395,1.735065,-0.940765,-0.512012,-3.511472,-2.028010,-0.196793,2.113285,-0.560079,-0.177933,0.631992,-4.418488,2.199284,0.876753,-2.268795,1.427826,5.306685,7.042498,1.086380,0.405567,2.816671,-0.055701,-0.389130,-5.851556,-5.862928,-1.295746,1.828769,7.065789,-3.631787,1.847077,4.753578,-0.022325,3.098745,2.079338,1.169604,0.126212,8.982183,5.571905,0.754399,-1.490074,4.943506,0.399031,0.429051,-4.859672,-6.413958,-4.246317,2.060524,6.171733,1.764856,5.250731,3.532447,-0.623754,2.859708,-0.495043,0.598131,2.335803,9.698425,6.635998,1.408039,-3.265816,5.764218,-0.700987,-1.631853,-3.484339,-6.350687,-3.681504,5.678374,4.078689,0.421867,1.960467,5.763502,-1.152368,5.994359,2.881854,0.999346,1.171615,10.311854,3.124194,0.918925,-0.884798,4.855608,-0.948022,-3.295305,-2.612484,-1.973828,-0.049434,2.623005,6.176755,-0.853379,5.749740,5.325172,-1.637518,-1.132431,1.198618,-2.196430,1.210761,7.567923,7.793386,1.082332,-1.941502,4.858668,0.807849,2.063232,-4.127200,-3.994300,-2.402945,2.353823,6.086142,-0.795167,-0.487933,1.885520,-0.318516,4.463936,1.038881,-1.221152,1.981562,4.929876,7.403000,0.681313,-5.269679,4.948593};
    // // float C[M * K] = {0};
    // // float C_MatrixUtil[M * K] = {0};
    // // float A_MatrixUtilTransposed[M * N]{0};
    // // float B_MatrixUtilTransposed[N * K]{0};
    // // float A_Ne10Transposed[M * N]{0};
    // // float B_Ne10Transposed[N * K]{0};
    // //
    // // float A_vec[vectorSize] = {0.814724,0.905792,0.126987,0.913376,0.632359,0.097540,0.278498,0.546882,0.957507,0.964889,0.157613,0.970593,0.957167,0.485376,0.800280,0.141886};
    // // float B_matrix[vectorSize * 4] = {-0.124144,-1.068870,1.093266,1.544212,1.489698,-0.809499,1.109273,0.085931,1.409034,-2.944284,-0.863653,-1.491590,1.417192,1.438380,0.077359,-0.742302,0.671497,0.325191,-1.214117,-1.061582,-1.207487,-0.754928,-1.113501,2.350457,0.717239,1.370299,-0.006849,-0.615602,1.630235,-1.711516,1.532630,0.748077,0.488894,-0.102242,-0.769666,-0.192419,1.034693,-0.241447,0.371379,0.888610,0.726885,0.319207,-0.225584,-0.764849,-0.303441,0.312859,1.117356,-1.402269,0.293871,-0.864880,-1.089064,-1.422376,-0.787283,-0.030051,0.032557,0.488194,0.888396,-0.164879,0.552527,-0.177375,-1.147070,0.627707,1.100610,-0.196053};
    // // constexpr float C_vecResult[4] = {5.853615,-1.948750,2.058402,-1.838518};
    // // float C_MatrixUtilvec[vectorSize];
    // // float C_vec[4];
    //
    //
    //
    TimerUtil timer;

    timer.tic();
    for (size_t i = 0; i < 1000; ++i)
    {
        // MatrixUtil::MatrixMultFloat(A, M, N, B, N, K, C_MatrixUtil);
        MatrixUtil::MatrixMult(A, M, N, B, N, K, C_MatrixUtil);
    }
    timer.toc("MatrixUtil A(24x32) B(32x20) Elapsed Time");

    logPRN(LOG_TYPE_GENERAL, "\nDifference1:\n\n");
    printMatrixDifference(C_Matlab, C_MatrixUtil, M, K);

    timer.tic();
    for (size_t i = 0; i < 1000; ++i)
    {
        // MatrixMultFloat(A, M, N, B, N, K, C);
        MatrixMultDouble(A, M, N, B, N, K, C);
    }
    timer.toc("Ne10 A(24x32) B(32x20) Elapsed Time");

    logPRN(LOG_TYPE_GENERAL, "\nDifference2:\n\n");
    printMatrixDifference(C_Matlab, C, M, K);

    logPRN(LOG_TYPE_GENERAL, "\nDifference3:\n\n");
    printMatrixDifference(C_MatrixUtil, C, M, K);

    //
    // // timer.tic();
    // // for (size_t i = 0; i < 1000; ++i)
    // // {
    // // //Vector1x16MatrixMultiplicationFloat16x16(A_vec, B_matrix, C_vec);
    // // //Vector1x16MatrixMultiplicationDouble16x16(A_vec, B_matrix, C_vec);
    // // //VectorMatrixMultiplicationDouble(A_vec, B_matrix, vectorSize, 4, C_vec);
    // // VectorMatrixMultiplicationFloat(A_vec, B_matrix, vectorSize, 4, C_vec);
    // // }
    // // timer.toc("Ne10 A(1x4) B(16x4) Elapsed Time");
    // // logPRN(LOG_TYPE_GENERAL, "\nDifference Vector x Matrix:\n\n");
    // // printMatrixDifference(C_vecResult, C_vec, 1, 4);
    //
    //
    // // timer.tic();
    // // for (size_t i = 0; i < 1000; ++i)
    // // {
    // // MatrixUtil::transpose(A_16x16, vectorSize, vectorSize, A_MatrixUtil_16x16);
    // // //MatrixUtil::transposeFloat(A, M, N, A_MatrixUtilTransposed);
    // // }
    // // timer.toc("MatrixUtil Transpose Elapsed Time");
    // //
    // // timer.tic();
    // // for (size_t i = 0; i < 1000; ++i)
    // // {
    // // MatrixTransposeDouble16x16(A_16x16, A_Ne10_16x16);
    // // //MatrixTransposeFloat(A, M, N, A_Ne10Transposed);
    // // }
    // // timer.toc("Ne10 Transpose Elapsed Time");
    // //
    // //
    // //
    // // logPRN(LOG_TYPE_GENERAL, "\nA:\n\n");
    // // printMatrix(A_16x16, vectorSize, vectorSize);
    // //
    // // logPRN(LOG_TYPE_GENERAL, "\nA_transposed:\n\n");
    // // printMatrix(A_Ne10_16x16, vectorSize, vectorSize);
    // //
    // // logPRN(LOG_TYPE_GENERAL, "\nDifference A_transposed:\n\n");
    // // printMatrixDifference(A_MatrixUtil_16x16, A_Ne10_16x16, vectorSize, vectorSize);
    //
    // // timer.tic();
    // // for (size_t i = 0; i < 1000; ++i)
    // // {
    // // MatrixUtil::transpose(B, N, K, B_MatrixUtilTransposed);
    // // //MatrixUtil::transposeFloat(B, N, K, B_MatrixUtilTransposed);
    // // }
    // // timer.toc("MatrixUtil Transpose Elapsed Time");
    // //
    // // timer.tic();
    // // for (size_t i = 0; i < 1000; ++i)
    // // {
    // // MatrixTransposeDouble(B, N, K, B_Ne10Transposed);
    // // //MatrixTransposeFloat(B, N, K, B_Ne10Transposed);
    // // }
    // // timer.toc("Ne10 Transpose Elapsed Time");
    // //
    // //
    // // logPRN(LOG_TYPE_GENERAL, "\nB:\n\n");
    // // printMatrix(B, N, K);
    // //
    // // logPRN(LOG_TYPE_GENERAL, "\nB_transposed:\n\n");
    // // printMatrix(B_Ne10Transposed, K, N);
    // //
    // // logPRN(LOG_TYPE_GENERAL, "\nDifference B_transposed:\n\n");
    // // printMatrixDifference(B_MatrixUtilTransposed, B_Ne10Transposed, K, M);
    //
}
//
// void Ne10Util::testDotProductsAndConjugate()
// {
// constexpr size_t M = 16;
//
// // float A_real[M] {0.814724,0.905792,0.126987,0.913376,0.632359,0.097540,0.278498,0.546882,0.957507,0.964889,0.157613,0.970593,0.957167,0.485376,0.800280,0.141886};
// // float A_imag[M] {0.421761,0.915736,0.792207,0.959492,0.655741,0.035712,0.849129,0.933993,0.678735,0.757740,0.743132,0.392227,0.655478,0.171187,0.706046,0.031833};
// // float B_real[M] {-1.068870,-0.809499,-2.944284,1.438380,0.325191,-0.754928,1.370299,-1.711516,-0.102242,-0.241447,0.319207,0.312859,-0.864880,-0.030051,-0.164879,0.627707};
// // float B_imag[M] {1.093266,1.109273,-0.863653,0.077359,-1.214117,-1.113501,-0.006849,1.532630,-0.769666,0.371379,-0.225584,1.117356,-1.089064,0.032557,0.552527,1.100610};
// // constexpr float C_Matlab_Pure_real {-1.948749555508579};
// // constexpr float C_Matlab_real {-3.151909};
// // constexpr float C_Matlab_imag {-0.895675};
// // float C_real {0};
// // float C_imag {0};
// // float C_MatrixUtil_real {0};
// // float C_MatrixUtil_imag {0};
// // float A_imagConjugated[M] {0};
// // float B_imagConjugated[M] {0};
//
// double A_real[M] {0.814724,0.905792,0.126987,0.913376,0.632359,0.097540,0.278498,0.546882,0.957507,0.964889,0.157613,0.970593,0.957167,0.485376,0.800280,0.141886};
// double A_imag[M] {0.421761,0.915736,0.792207,0.959492,0.655741,0.035712,0.849129,0.933993,0.678735,0.757740,0.743132,0.392227,0.655478,0.171187,0.706046,0.031833};
// double B_real[M] {-1.068870,-0.809499,-2.944284,1.438380,0.325191,-0.754928,1.370299,-1.711516,-0.102242,-0.241447,0.319207,0.312859,-0.864880,-0.030051,-0.164879,0.627707};
// double B_imag[M] {1.093266,1.109273,-0.863653,0.077359,-1.214117,-1.113501,-0.006849,1.532630,-0.769666,0.371379,-0.225584,1.117356,-1.089064,0.032557,0.552527,1.100610};
// constexpr double C_Matlab_Pure_real {-1.948749555508579};
// constexpr double C_Matlab_real {-3.151909};
// constexpr double C_Matlab_imag {-0.895675};
// double C_real {0};
// double C_imag {0};
// double C_MatrixUtil_real {0};
// double C_MatrixUtil_imag {0};
// double A_imagConjugated[M] {0};
// double B_imagConjugated[M] {0};
//
//
// TimerUtil timer;
//
// // Real Dot Product
// timer.tic();
// for (size_t i = 0; i < 1000; ++i)
// {
// MatrixUtil::MatrixMult(A_real, 1, M, B_real, M, 1, &C_MatrixUtil_real);
// //MatrixUtil::MatrixMultFloat(A_real, 1, M, B_real, M, 1, &C_MatrixUtil_real);
// }
// timer.toc("Real Dot Product MatrixUtil Elapsed Time");
// logSYS(LOG_TYPE_GENERAL, "MATLAB Result - MatrixUtil %lf", C_Matlab_Pure_real - C_MatrixUtil_real);
//
// timer.tic();
// for (size_t i = 0; i < 1000; ++i)
// {
// VectorDotProductDouble(A_real, B_real, M, &C_real);
// //VectorDotProductFloat(A_real, B_real, M, &C_real);
// }
// timer.toc("Real Dot Product (General) Ne10 Elapsed Time");
//
// logSYS(LOG_TYPE_GENERAL, "MATLAB Result - Ne10 (General) %lf", C_Matlab_Pure_real - C_real);
//
// timer.tic();
// for (size_t i = 0; i < 1000; ++i)
// {
// VectorDotProductDouble16x16(A_real, B_real, &C_real);
// //VectorDotProductFloat16x16(A_real, B_real, &C_real);
// }
// timer.toc("Real Dot Product (16x16) Ne10 Elapsed Time");
// logSYS(LOG_TYPE_GENERAL, "MATLAB Result - Ne10 (16x16) %lf", C_Matlab_Pure_real - C_real);
//
//
// // Complex Dot Product
// timer.tic();
// for (size_t i = 0; i < 1000; ++i)
// {
// MatrixUtil::MatrixComplexMult(A_real, A_imag, 1, M, B_real, B_imag, M, 1, &C_MatrixUtil_real, &C_MatrixUtil_imag);
// //MatrixUtil::MatrixComplexMultFloat(A_real, A_imag, 1, M, B_real, B_imag, M, 1, &C_MatrixUtil_real, &C_MatrixUtil_imag);
// }
// timer.toc("Complex Dot Product MatrixUtil Elapsed Time");
// logSYS(LOG_TYPE_GENERAL, "MATLAB Result - MatrixUtil Real: %lf", C_Matlab_real - C_MatrixUtil_real);
// logSYS(LOG_TYPE_GENERAL, "MATLAB Result - MatrixUtil Imag: %lf", C_Matlab_imag - C_MatrixUtil_imag);
//
// timer.tic();
// for (size_t i = 0; i < 1000; ++i)
// {
// //VectorDotProductComplexDouble16x16(A_real, A_imag, B_real, B_imag, &C_real, &C_imag);
// //VectorDotProductComplexFloat16x16(A_real, A_imag, B_real, B_imag, &C_real, &C_imag);
// //VectorDotProductComplexFloat(A_real, A_imag, B_real, B_imag, M, &C_real, &C_imag);
// VectorDotProductComplexDouble(A_real, A_imag, B_real, B_imag, 16, &C_real, &C_imag);
// }
// timer.toc("Complex Dot Product (16x16) Ne10 Elapsed Time");
//
// logSYS(LOG_TYPE_GENERAL, "MATLAB Result - MatrixUtil Real: %lf", C_Matlab_real - C_real);
// logSYS(LOG_TYPE_GENERAL, "MATLAB Result - MatrixUtil Imag: %lf", C_Matlab_imag - C_imag);
//
//
//
// // Conjugate
//
// timer.tic();
// for (size_t i = 0; i < 1000; ++i)
// {
//
// VectorComplexConjugateDouble(A_imag, A_imagConjugated, M);
// VectorComplexConjugateDouble16x16(B_imag, B_imagConjugated);
//
// // VectorComplexConjugateFloat(A_imag, A_imagConjugated, M);
// // VectorComplexConjugateFloat16x16(B_imag, B_imagConjugated);
// }
// timer.toc("Conjugate (16x16) Ne10 Elapsed Time");
//
// logPRN(LOG_TYPE_GENERAL, "\nSum of A_imag and A_imagConjugated\n");
// for (size_t i = 0; i < M; ++i)
// {
// logPRN(LOG_TYPE_GENERAL, "%lf  ", A_imag[i] + A_imagConjugated[i]); // Must be 0
// }
//
// logPRN(LOG_TYPE_GENERAL, "\nSum of B_imag and B_imagConjugated\n");
// for (size_t i = 0; i < M; ++i)
// {
// logPRN(LOG_TYPE_GENERAL, "%lf  ", B_imag[i] + B_imagConjugated[i]); // Must be 0
// }
// }
