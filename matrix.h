#ifndef _MATRIX_
#define _MATRIX_

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

#define IterateMatrix(EXPRESSION) \
    for (int i = 0; i < height; ++i) { \
        for (int j = 0; j < width; ++j) { \
            EXPRESSION \
        } \
    }

#define Overload(op) \
    Matrix operator op (const Matrix& other) const { \
        if (width != other.width || height != other.height) { \
            throw std::invalid_argument("Matrix dimensions must match"); \
        } \
        Matrix result(height, width); \
        IterateMatrix(result.mat[i][j] = mat[i][j] op other.mat[i][j];); \
        return result; \
    }

class Matrix {
public:
    int width;
    int height;
    std::vector<std::vector<double>> mat;
    Matrix(int hei = 0, int wid = 0, double std_dev = 0, double mean = 0.0) : height(hei), width(wid) {
        mat.resize(height, std::vector<double>(width, mean));
        if (std_dev) {

            std::default_random_engine generator;
            std::normal_distribution<double> distribution(mean, std_dev);

            IterateMatrix(mat[i][j] = distribution(generator););
        }
    }
    ~Matrix() {}

    // General Operations
    // ------------------
    Matrix T() const {
        Matrix transposed(width, height);
        IterateMatrix(transposed.mat[j][i] = mat[i][j];);
        return transposed;
    }

    Matrix operator |(const Matrix& other) const {
        if (width != other.height) {
            throw std::invalid_argument("Incorrect dimensions");
        }
        Matrix result(height, other.width, 0, 0);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < other.width; j++) {
                for (int k = 0; k < width; k++) {
                    result.mat[i][j] += mat[i][k] * other.mat[k][j];
                }
            }
        }
        return result;
    }

    Matrix operator*(const double scalar) const {
        Matrix result(height, width);
        IterateMatrix(result.mat[i][j] = mat[i][j] * scalar;);
        return result;
    }

    Matrix operator/(const double scalar) const {
        Matrix result(height, width);
        IterateMatrix(result.mat[i][j] = mat[i][j] / scalar;);
        return result;
    }

    Overload(+);
    Overload(-);
    Overload(*);
    Overload(/);

    bool operator == (const Matrix &other) const {
        if (width != other.width || height != other.height) {
            return false;}
        IterateMatrix(if (mat[i][j] != other.mat[i][j]) return false;);
        return true;
    }

    Matrix &operator=(const Matrix &result) {
        width = result.width;
        height = result.height;
        mat = result.mat;

        return *this;
    }

    Matrix &operator=(const std::vector<std::vector<double>> &result) {
        width = result[0].size();
        height = result.size();
        mat = result;
        return *this;
    }

    const double sum(bool l1=false) const {
        double result{};
        if (l1) {
            IterateMatrix(result += std::abs(mat[i][j]););
            return result;
        }
        IterateMatrix(result += mat[i][j];);
        return result;
    }

    const double max(bool arg=false) const {
        double result = 0;
        IterateMatrix( if (result < mat[i][j]) result = mat[i][j];);
        return result;
    }

    const double min(bool give_arg=false) const {
        double result = 0;
        IterateMatrix( if (result > mat[i][j]) result = mat[i][j];);
        return result;
    }

    Matrix matexp() const {
        Matrix result(height, width);
        IterateMatrix(result.mat[i][j] = std::exp(mat[i][j]););
        return result;
    }

    void print() const {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                std::cout << mat[i][j] << " ";
            }
            std::cout << '\n';
        }
    }

    // Machine Learning Functions
    // --------------------------
    const Matrix relu(double coeff = 0) const {
        Matrix result(height, width);
        IterateMatrix(result.mat[i][j] = (0 < mat[i][j]) ? mat[i][j] : (coeff * mat[i][j]););
        return result;
    }

    const Matrix relu_derivative(double coeff = 0) const {
        Matrix result(height, width);
        IterateMatrix(result.mat[i][j] = (0 <= mat[i][j]) ? 1.0 : coeff;);
        return result;
    }

    const Matrix softmax() const {
        Matrix result = matexp();
        return result / result.sum();
    }

    const Matrix softmax_derivative() const {
        Matrix jacobian(height, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < height; j++) {
                if (i == j) jacobian.mat[i][j] = mat[i][0] * (1 - mat[i][0]);
                else jacobian.mat[i][j] = 0; //-mat[i][0] * mat[j][0];
            }
        }
        return jacobian;
    }

    const double l2_normalized() const {
        double result{};
        IterateMatrix(result += std::pow(mat[i][j], 2););
        return std::sqrt(result);
    }


};

#endif