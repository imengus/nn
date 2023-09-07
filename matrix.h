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
        Matrix result(width, height); \
        IterateMatrix(result.mat[i][j] = mat[i][j] op other.mat[i][j];); \
        return result; \
    }

class Matrix {
public:
    int width;
    int height;
    std::vector<std::vector<double>> mat;
    Matrix(int wid = 0, int hei = 0, double std_dev = 1, double mean = 0.0) : width(wid), height(hei) {
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
        Matrix transposed(height, width);
        IterateMatrix(transposed.mat[j][i] = mat[i][j];);
        return transposed;
    }

    Matrix operator*(const double scalar) const {
        Matrix result(width, height);
        IterateMatrix(result.mat[i][j] = mat[i][j] * scalar;);
        return result;
    }

    Overload(+);
    Overload(-);
    Overload(*);
    Overload(/);

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

    Matrix mult(const Matrix& other) const {
        if (width != other.height) {
            throw std::invalid_argument("Incorrect dimensions");
        }
        Matrix result(other.width, height, 0, 0);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < other.width; j++) {
                for (int k = 0; k < width; k++) {
                    result.mat[i][j] += mat[i][k] * other.mat[k][j];
                }
            }
        }
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
    Matrix relu(double coeff = 0) const {
        Matrix result(width, height);
        IterateMatrix(result.mat[i][j] = (0 < mat[i][j]) ? mat[i][j] : (coeff * mat[i][j]););
        return result;
    }

    Matrix relu_derivative(double coeff = 0) const {
        Matrix result(width, height);
        IterateMatrix(result.mat[i][j] = (0 <= mat[i][j]) ? 1.0 : coeff;);
        return result;
    }

    const double l1_normalized() const {
        double result{};
        IterateMatrix(result += std::abs(mat[i][j]););
        return std::sqrt(result);
    }

    const double l2_normalized() const {
        double result{};
        IterateMatrix(result += std::pow(mat[i][j], 2););
        return std::sqrt(result);
    }
};
