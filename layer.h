#ifndef _LAYER_
#define _LAYER_

#include "matrix.h"
#include <fstream>

class Layer {
public:
    virtual void print_parameters() = 0;
    virtual Matrix forward(Matrix inp) = 0;
    virtual Matrix backward(Matrix prev_grad) = 0;
};

class DenseLayer : public Layer {
private:
    double w_sdev, w_mean, b_sdev, b_mean, learn_rate, momentum;
    int layer_num;

    Matrix prev_weight_delta, prev_bias_delta, weight_delta, bias_delta;

public:
    Matrix input, output, weight, bias, grad;

    DenseLayer(
        int num,
        double inp_height,
        double out_height,
        double mom = 0.5,
        double lr = 0.0000001,
        double wsdev = 0.5,
        double wmean = 0,
        double bsdev = 0,
        double bmean = 0
    ) :
        w_sdev(wsdev),
        w_mean(wmean),
        b_sdev(bsdev),
        b_mean(bmean),
        learn_rate(lr),
        momentum(mom),
        layer_num(num) {

        weight = Matrix(out_height, inp_height, std::sqrt(2/inp_height), 0);
        bias = Matrix(out_height, 1, b_sdev, b_mean);

        prev_weight_delta = Matrix(out_height, inp_height, 0, 0);
        prev_bias_delta = Matrix(out_height, 1, 0, 0);

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

        weight_delta = grad_weight * learn_rate;
        bias_delta = grad_bias * learn_rate;

        weight = weight - weight_delta - prev_weight_delta * momentum;
        bias = bias - bias_delta - prev_bias_delta * momentum;

        prev_weight_delta = weight_delta;
        prev_bias_delta = bias_delta;

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

class SoftMaxLayer : public Layer {
private:
public:
    int layer_num;
    double relu_coeff;
    Matrix input, output, grad;

    SoftMaxLayer(int num) : layer_num(num) {}

    ~SoftMaxLayer() {}

    void print_parameters() override {
        std::cout << "Softmax " << layer_num << std::endl;
        std::cout << std::endl;
    }

    Matrix forward(Matrix inp) override {
        input = inp;
        output = input.softmax();
        return output;
    }

    Matrix backward(Matrix prev_grad) override {
        grad = output.softmax_derivative().mult(prev_grad);
        return grad;
    }
};

class MLP {
public:
    std::vector<Layer*> layers;

    double calculate_error(Matrix Y_hat, Matrix Y, Matrix &gradient) {
        gradient =  gradient + (Y_hat - Y) * 2;
        return (Y_hat - Y).l2_normalized();
    }

    void add_layer(Layer &layer) {layers.push_back(&layer);}

    void print_network() {for (auto layer : layers) layer->print_parameters();}

    void train(Matrix train_X, Matrix train_Y, int mb_size, int n_epoch) {
        std::cout << "Starting training" << std::endl;

        std::ofstream MSEHist("mse_history.csv");

        int len = train_X.height;
        if ((len/mb_size) < 1) mb_size = len;
        int n_batches = len / mb_size;
        int s = train_Y.mat[0].size();

        Matrix X;
        Matrix Y;
        Matrix gradient(s, 1, 0, 0);
        Matrix batch_grad(s, 1, 0, 0);
        Matrix init_grad(s, 1, 0, 0);
        double total_mse{};
        double mse{};
        for (int i = 0; i < n_epoch; i++) {
            for (int j = 0; j < n_batches; j++) {
                for (int k = j * mb_size; k < (j + 1) * mb_size; k++) {

                    X = {train_X.mat[j]};
                    Y = {train_Y.mat[j]};

                    X = X.T();
                    Y = Y.T();

                    for (auto layer : layers) X = (*layer).forward(X);
                    
                    total_mse += calculate_error(X, Y, batch_grad) * 0.01;

                    }
                gradient = batch_grad / len;
                batch_grad = init_grad;

                for (int l = layers.size() - 1; l >= 0; l--) {
                    gradient = (*layers[l]).backward(gradient);
                }
            }
            if (std::isnan(-total_mse)) throw std::invalid_argument("Must adjust parameters as MSE is NaN");
            MSEHist << total_mse << std::endl;
            std::cout << i + 1 << ": " << total_mse << std::endl;
            total_mse = 0;
        }
        MSEHist.close();
        std::cout << std::endl;
    }

    Matrix predict(Matrix X) {
        for (auto layer : layers) X = (*layer).forward(X);
        return X;
    }

    template <typename F>
    void test(Matrix test_X, Matrix test_Y, F&&func) {
        std::cout << "Starting testing" << std::endl << std::endl;
        
        Matrix X, Y;
        int total_correct = 0;
        int total_cases = test_Y.height;
        for (int i = 0; i < total_cases; i++) {
            X = {test_X.mat[i]};
            Y = {test_Y.mat[i]};

            X = X.T();
            Y = Y.T();
            X = predict(X);
            
            total_correct = func(X, Y, total_correct);
        }
        std::cout << "Results: " << std::endl;
        std::cout << total_correct << "/" << total_cases << std::endl;
        float result = total_correct / total_cases;
    }

};
#endif