#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "matrix.h"
#include "layer.h"

void read_mnist_into(Matrix &X, Matrix &Y, std::string filepath) {
    std::ifstream inputFile(filepath);

    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> labels;
    std::vector<double> col = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    std::string line;

    int count = 0;
    while (std::getline(inputFile, line) && (count <= 1000)) {
        std::vector<double> row;
        std::istringstream ss(line);
        std::string cell;

        int cnt = 0;
        
        while (std::getline(ss, cell, ',')) {
            try {
                double value = std::stod(cell);
                if (cnt == 0) {
                    col[value] = 1;
                    labels.push_back(col);
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

int main () {
    Matrix train_X, train_Y, test_X, test_Y;

    read_mnist_into(train_X, train_Y, "data/mnist_train.csv");
    read_mnist_into(test_X, test_Y, "data/mnist_test.csv");
    
    MLP nn;

    // Sequential layers of neural network
    DenseLayer d0(0, 784, 32, 0.1);
    ReLULayer r1(1, 0.5);
    DenseLayer d2(2, 32, 10, 0.1);
    SoftMaxLayer s3(3);

    nn.add_layer(d0);
    nn.add_layer(r1);
    nn.add_layer(d2);
    nn.add_layer(s3);

    std::cout << "starting training" << std::endl;
    nn.train(train_X, train_Y, 10, 100);
    return 0;
}