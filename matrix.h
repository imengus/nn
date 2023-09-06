#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

#define IterateMatrix(EXPRESSION) {\
    for (int i = 0; i < height; ++i) {\
        for (int j = 0; j < width; ++j) {\
            EXPRESSION\
        }}}
    
#define Overload(op) \
    Matrix operator op (const Matrix& other) const {\
        if (width != other.width || height != other.height) {\
            throw invalid_argument("Matrix dimensions must match");}\
        \
        Matrix result(width, height);\
        IterateMatrix(result.mat[i][j] = mat[i][j] op other.mat[i][j];);\
        return result;\
        }

class Matrix {
    public:
        int width;
        int height;
        vector<vector<double>> mat;
        Matrix(int wid=0, int hei=0, double std_dev=1, double mean=0.0) : width(wid), height(hei) {
            mat.resize(height, vector<double>(width, mean));
            if (std_dev) {
                
                default_random_engine generator;
                normal_distribution<double> distribution(mean, std_dev);

                IterateMatrix(mat[i][j] = distribution(generator););
            }
        }
        ~Matrix() {}

        // General Operations
        // ------------------
        Matrix transpose() const {
            Matrix trans(height, width);
            IterateMatrix(trans.mat[j][i] = mat[i][j];);
            return trans;
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

        Matrix mult(const Matrix& other) const {
            if (width != other.height) {
                throw invalid_argument("Incorrect dimensions");
            }
            Matrix result(other.width, height);
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < other.width; ++j) {
                    for (int k = 0; k < width; ++k) {
                        result.mat[i][j] += mat[i][j]*other.mat[j][k];
                    }
                }
            }
            return result;
        }

        void print() const {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    cout << mat[i][j] << " ";
                }
                cout << '\n';
            }
        }

        // Machine Learning Functions
        // --------------------------
        Matrix relu(double coeff) const {
            Matrix result(width, height);
            IterateMatrix(result.mat[i][j] = (0<mat[i][j]) ? mat[i][j] : (coeff * mat[i][j]););
            return result;
        }

        Matrix relu_derivative(double coeff) const {
            Matrix result(width, height);
            IterateMatrix(result.mat[i][j] = (0<=mat[i][j]) ? 1.0 : coeff;);
            return result;
        }

        const double l1_normalized() const {
            double result{};
            IterateMatrix(result += abs(mat[i][j]););
            return sqrt(result);
        }

        const double l2_normalized() const {
            double result{};
            IterateMatrix(result += pow(mat[i][j], 2););
            return sqrt(result);
        }
};

