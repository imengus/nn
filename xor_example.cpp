#include "matrix.h"
#include "layer.h"

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
        {0},
        {1},
        {1},
        {0},
        {1},
        {0},
        {0},
        {1}
    };

    MLP nn;

    // Sequential layers of neural network
    DenseLayer d0(0, 3, 10, 0.1);
    ReLULayer r1(1, 0.2);
    DenseLayer d2(2, 10, 1, 0.1);
    ReLULayer r3(3, 0.2);

    nn.add_layer(d0);
    nn.add_layer(r1);
    nn.add_layer(d2);
    nn.add_layer(r3);

    // Converges to 0
    nn.train(train_X, train_Y, 4, 25);

    // No training necessary. All cases covered in training
    return 0;
}