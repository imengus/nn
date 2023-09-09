#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "matrix.h"
#include "layer.h"

void read_mnist_into(
    Matrix &X, 
    Matrix &Y, 
    std::string filepath, 
    int upto=1000, 
    bool bycol=true) 
    {
    std::ifstream inputFile(filepath);

    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> labels;
    std::vector<double> col = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    std::string line;

    int count = 0;
    while (std::getline(inputFile, line) && (count <= upto)) {
        std::vector<double> row;
        std::istringstream ss(line);
        std::string cell;

        int cnt = 0;
        
        while (std::getline(ss, cell, ',')) {
            try {
                double value = std::stod(cell);
                if (cnt == 0) {
                    col[value] = 1;
                    if (bycol) labels.push_back(col);
                    else labels.push_back({value});
                    col[value] = 0;
                }
                else row.push_back(value/255);
                cnt++;
            } 
            catch (const std::invalid_argument& e) {
                break;
            }
        }
        data.push_back(row);
        count++;
    }
    data.erase(data.begin());

    inputFile.close();

    X = data;
    Y = labels;
}

int increment(Matrix &X, Matrix &Y, int total_correct) {
    double max_val{};
    int max_index{};
    for (int i = 0; i < X.height; i++) {
        if (X.mat[i][0] > max_val) {
            max_index = i;
            max_val = X.mat[i][0];
        }
    }
    int y = Y.mat[0][0];
    if (max_index == y) total_correct += 1;
    return total_correct;
}

int main () {
    Matrix train_X, train_Y, test_X, test_Y;
    int trn_s{5000}, tst_s{1000};

    read_mnist_into(train_X, train_Y, "data/mnist_train.csv", trn_s);
    read_mnist_into(test_X, test_Y, "data/mnist_test.csv", tst_s);
    
    MLP nn;

    // Sequential layers of neural network
    DenseLayer d0(0, 784, 10, trn_s);
    ReLULayer r1(1, 0.2);
    DenseLayer d2(2, 10, 10, trn_s);
    SoftMaxLayer s3(3);

    nn.add_layer(d0);
    nn.add_layer(r1);
    nn.add_layer(d2);
    nn.add_layer(s3);

    nn.train(train_X, train_Y, 50, 10);
    nn.test(test_X, test_Y, increment);
    return 0;
}