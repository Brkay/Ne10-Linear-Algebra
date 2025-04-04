
#pragma once

#include <stdio.h>
#include <arm_neon.h>
#include "Defines/TypeDefines.h"

class Ne10Util
{
public:
    static int32_t MatrixMultFloat(const float* A, const size_t& n, const size_t& m1, const float* B, const size_t& m2, const size_t& k, float* C);
    static int32_t MatrixMultComplexFloat(const float* A_real, const float* A_imag, const size_t& n, const size_t& m1, const float* B_real, const float* B_imag, const size_t& m2, const size_t& k, float* C_real, float* C_imag);

    static int32_t VectorxVectorElementwiseMultiplicationFloat(const float* A, const float* B, const size_t& vectorSize, float* C);
    static int32_t VectorxVectorElementwiseMultiplicationDouble(const double* A, const double* B, const size_t& vectorSize, double* C);

    static int32_t VectorxVectorElementwiseMultiplicationFloat1x16(const float* A, const float* B, float* C);
    static int32_t VectorxVectorElementwiseMultiplicationDouble1x16(const double* A, const double* B, double* C);

    static int32_t VectorMatrixMultiplicationFloat(const float* A, const float* B, const size_t& m1, const size_t& m2, float* C);
    static int32_t VectorMatrixMultiplicationDouble(const double* A, const double* B, const size_t& m1, const size_t& m2, double* C);

    static int32_t Vector1x16MatrixMultiplicationFloat16x16(const float* A, const float* B, float* C);
    static int32_t Vector1x16MatrixMultiplicationDouble16x16(const double* A, const double* B, double* C);


    static int32_t Vector1x16MatrixMultiplicationComplexFloat16x16(const float* A_real, const float* A_imag, const float* B_real, const float* B_imag, float* C_real, float* C_imag);
    static int32_t Vector1x16MatrixMultiplicationComplexDouble16x16(const double* A_real, const double* A_imag, const double* B_real, const double* B_imag, double* C_real, double* C_imag);

// TODO: Inmplement Vector * Matrix Complex Multiplication for generic dimensions.

///////
///
    static int32_t NormalizeComplexVectorFloat1x16(float* A_real, float* A_imag);
    static int32_t NormalizeComplexVectorDouble1x16(double* A_real, double* A_imag);

    //static int32_t NormalizeComplexVectorFloat1x16(float* A_real, float* A_imag);
    static int32_t NormComplexVectorDouble1x16(const double* A_real, const double* A_imag, double* C);
    static int32_t NormSquareComplexVectorDouble1x16(const double* A_real, const double* A_imag, double* C);

    static int32_t NormalizeComplexVectorFloat(float* A_real, float* A_imag, const size_t& vectorSize);
    static int32_t NormalizeComplexVectorDouble(double* A_real, double* A_imag, const size_t& vectorSize);


    static int32_t MatrixMultDouble(const double* A, const size_t& n, const size_t& m1, const double* B, const size_t& m2, const size_t& k, double* C);
    static int32_t MatrixMultComplexDouble(const double* A_real, const double* A_imag, const size_t& n, const size_t& m, const double* B_real, const double* B_imag, const size_t& m2, const size_t& k, double* C_real, double* C_imag);

    static int32_t VectorDotProductFloat(const float* A, const float* B, const size_t& vectorSize, float* C);
    static int32_t VectorDotProductDouble(const double* A, const double* B, const size_t& vectorSize, double* C);

    static int32_t VectorDotProductComplexFloat(const float* A_real, const float* A_imag, const float* B_real, const float* B_imag, const size_t& vectorSize, float* C_real, float* C_imag);
    static int32_t VectorDotProductComplexDouble(const double* A_real, const double* A_imag, const double* B_real, const double* B_imag, const size_t& vectorSize, double* C_real, double* C_imag);

    static int32_t VectorDotProductFloat1x16(const float* A, const float* B, float* C);
    static int32_t VectorDotProductDouble1x16(const double* A, const double* B, double* C);

    static int32_t VectorDotProductComplexFloat1x16(const float* A_real, const float* A_imag, const float* B_real, const float* B_imag, float* C_real, float* C_imag);
    static int32_t VectorDotProductComplexDouble1x16(const double* A_real, const double* A_imag, const double* B_real, const double* B_imag, double* C_real, double* C_imag);

    static int32_t VectorComplexConjugateFloat(const float* A_imag, float* A_conjugatedImag, const size_t& vectorSize);
    static int32_t VectorComplexConjugateDouble(const double* A_imag, double* A_conjugatedImag, const size_t& vectorSize);

    static int32_t VectorComplexConjugateFloat1x16(const float* A_imag, float* A_conjugatedImag);
    static int32_t VectorComplexConjugateDouble1x16(const double* A_imag, double* A_conjugatedImag);

    static int32_t MatrixTransposeFloat(const float* A, const uint32& n, const uint32& m, float* A_transposed);
    static int32_t MatrixTransposeDouble(const double* A, const uint32& n, const uint32& m, double* A_transposed);

    static int32_t MatrixTransposeDouble16x16(const double* A, double* A_transposed);

    template <typename T>
    static void printMatrix(const T* matrix, const size_t& rowSize, const size_t& columnSize);

    template <typename T>
    static void printMatrixDifference(const T* matrix1, const T* matrix2, const size_t& rowSize, const size_t& columnSize);

     static void testComplexMultiplications();
    static void testRealMultiplications();
    // static void testDotProductsAndConjugate();

    static void pinThreadToCore();


};

// Since the function is template, it must be defined in the header!
template <typename T>
void Ne10Util::printMatrix(const T* matrix, const size_t& rowSize, const size_t& columnSize)
{
    for(size_t rowIndex = 0; rowIndex < rowSize; ++rowIndex) {
        for(size_t colIndex = 0; colIndex < columnSize; ++colIndex) {
            printf("%lf, ", matrix[rowIndex * columnSize + colIndex]);
        }
        printf("\n");
    }
}


template <typename T>
void Ne10Util::printMatrixDifference(const T* matrix1, const T* matrix2, const size_t& rowSize, const size_t& columnSize)
{
    for(size_t rowIndex = 0; rowIndex < rowSize; ++rowIndex) {
        for(size_t colIndex = 0; colIndex < columnSize; ++colIndex) {
            printf("%lf, ", matrix1[rowIndex * columnSize + colIndex] - matrix2[rowIndex * columnSize + colIndex]);
        }
        printf("\n");
    }
}
