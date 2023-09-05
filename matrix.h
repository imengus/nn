#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
using namespace std;

class Matrix {
    public:
        int width;
        int height;
        vector<vector<double>> mat;
        Matrix(int wid=0, int hei=0, bool rand_fill=false, double fill=0.0) : width(wid), height(hei) {
            mat.resize(height, vector<double>(width, fill));
            if (rand_fill) {
                
                default_random_engine generator;
                normal_distribution<double> distribution(0, 1);

                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        mat[i][j] = distribution(generator);
                    }
                }
            }
            // cout << "Created" << endl;
        }
        ~Matrix() {}//cout << "Destroyed " << endl; }

        Matrix transpose() const {
            Matrix trans(width, height);

            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    trans.mat[i][j] = mat[j][i];
                }
            }
            return trans;
        }

        Matrix operator*(const double scalar) const {
            Matrix result(width, height);
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    result.mat[i][j] = mat[i][j] * scalar;
                }
            }
            return result;
        }


        Matrix operator+(const Matrix& other) const {
            if (width != other.width || height != other.height) {
                throw invalid_argument("Matrix dimensions must match for addition");
            }

            Matrix result(width, height);
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    result.mat[i][j] = mat[i][j] + other.mat[i][j];
                }
            }
            return result;
        }

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
};
