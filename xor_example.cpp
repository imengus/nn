#include "matrix.h"
#include "layer.h"

int argmax(Matrix &X, Matrix &Y, int total_correct) {
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

    Matrix train_X;
    Matrix train_Y;
    train_X = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1}
    };
    train_Y = {
        {1, 0},
        {0, 1},
        {0, 1},
        {1, 0},
        {0, 1},
        {1, 0},
        {1, 0},
        {0, 1}
    };

    MLP nn;

    DenseLayer d0(0, 3, 10);
    ReLULayer r1(1, 0.2);
    DenseLayer d2(2, 10, 2);
    SoftMaxLayer s3(3);

    nn.add_layer(d0);
    nn.add_layer(r1);
    nn.add_layer(d2);
    nn.add_layer(s3);
    nn.print_network();

    nn.train(train_X, train_Y, 4, 100);
    // nn.test(train_Y, train_Y, argmax);
    return 0;
}