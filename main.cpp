#include "matrix.h"
#include <memory>

class Layer {
public:
    virtual void print_parameters() = 0;
    virtual Matrix forward(Matrix inp) = 0;
    virtual Matrix backward(Matrix prev_grad) = 0;
};

class DenseLayer : public Layer {
private:
    double w_sdev, w_mean, b_sdev, b_mean;
    int layer_num;
    double learn_rate;

public:
    Matrix input, output, weight, bias, grad;

    DenseLayer(
        int num,
        double inp_height,
        double out_height,
        double wsdev = 0.25,
        double wmean = 0.5,
        double bsdev = 0,
        double bmean = 0,
        double lr = 0.1
    ) :
        w_sdev(wsdev),
        w_mean(wmean),
        b_sdev(bsdev),
        b_mean(bmean),
        learn_rate(lr),
        layer_num(num) {
        weight = Matrix(inp_height, out_height, w_sdev, w_mean);
        bias = Matrix(1, out_height, b_sdev, b_mean);
    }

    ~DenseLayer() {}

    void print_parameters() override {
        std::cout << "Dense Layer " << layer_num << std::endl;
        std::cout << "___weight:" << std::endl;
        weight.print();
        std::cout << "___bias:" << std::endl;
        bias.print();
        std::cout << std::endl;
    }

    Matrix forward(Matrix inp) override {
        input = inp;
        output = weight.mult(input) - bias;
        return output;
    }

    Matrix backward(Matrix prev_grad) override {
        Matrix grad_weight = prev_grad.mult(input.T());
        Matrix grad_bias = prev_grad * -1;
        grad = (prev_grad.T().mult(weight)).T();

        weight = weight - grad_weight * learn_rate;
        bias = bias - grad_bias * learn_rate;

        return grad;
    }
};

class ReLULayer : public Layer {
private:
public:
    int layer_num;
    double relu_coeff;
    Matrix input, output, grad;

    ReLULayer(int num, double coeff) : layer_num(num), relu_coeff(coeff) {}

    ~ReLULayer() {}

    void print_parameters() override {
        std::cout << "ReLU " << layer_num << std::endl;
        std::cout << "__coeff: " << relu_coeff << std::endl;
        std::cout << std::endl;
    }

    Matrix forward(Matrix inp) override {
        input = inp;
        output = input.relu(relu_coeff);
        return output;
    }

    Matrix backward(Matrix prev_grad) override {
        grad = output.relu_derivative(relu_coeff) * prev_grad;
        return grad;
    }
};

class MLP {
private:
    double mse;
    Matrix gradient;

public:
    std::vector<Layer*> layers;
    Matrix train_X;
    Matrix train_Y;

    MLP(Matrix inp, Matrix out) : train_X(inp), train_Y(out) {}

    void calculate_error(Matrix Y_hat, Matrix Y) {
        mse = (Y_hat - Y).l2_normalized();
        gradient = (Y_hat - Y) * 2;
    }

    void train() {
        DenseLayer d0(0, 3, 10);
        ReLULayer r1(1, 0.2);
        DenseLayer d2(2, 10, 1);
        ReLULayer r3(3, 0.2);
        layers = {&d0, &r1, &d2, &r3};

        Matrix X;
        Matrix Y;
        double total_mse = 0;
        for (int i = 0; i < 1000; i++) {
            total_mse = 0;
            for (int j = 0; j < train_X.height; j++) {

                X = {train_X.mat[j]};
                Y = {train_Y.mat[j]};

                X = X.T();
                for (auto layer : layers) {
                    X = (*layer).forward(X);
                }
                calculate_error(X, Y.T());
                total_mse += mse;

                for (int i = layers.size() - 1; i >= 0; i--) {
                     gradient = (*layers[i]).backward(gradient);
                }
            }
        std::cout << total_mse << std::endl;
        }
    }
};

int main () {

    Matrix train_X(3, 8);
    Matrix train_Y(1, 8);
    train_X.mat = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1}
    };
    train_Y.mat = {
        {0},
        {1},
        {1},
        {0},
        {1},
        {0},
        {0},
        {1}
    };
    MLP nn(train_X, train_Y);
    nn.train();
    return 0;
}