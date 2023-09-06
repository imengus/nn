#include "matrix.h"
#include <map>

typedef map<int, Matrix> MatMap;
using MatMap = map<int, Matrix>;
using namespace std;

class Network {
    // private:
    public:
        vector<int> layer_dims{5, 4, 10};//{784, 100, 10};
        vector<Matrix> activations;
        vector<double> errors;
        int depth = layer_dims.size();
        double learn_rate = 0.01;
        double relu_coeff = 0.5;
        double w_sdev = 0.25;
        double w_mean = 0.5;
        double b_sdev = 0;
        double b_mean = 0;
        double mse = 0;
        Matrix gradient;
        Matrix Y_hat;
        MatMap weights;
        MatMap biases;

        void initialize_parameters() {

            for (int i = 1; i < depth; i++) {
                Matrix temp_w(layer_dims[i-1], layer_dims[i], w_sdev, w_mean);
                Matrix temp_b(1, layer_dims[i], b_sdev, b_mean);

                weights[i] = temp_w.relu(relu_coeff);
                biases[i] = temp_b;
            }
        }

        void print_parameters() {
            for (int i = 1; i < depth; i++ ) {
                cout << "___weight " << i << endl;
                weights[i].print();
                cout << "___bias " << i << endl;
                biases[i].print();
                cout << endl;
            }
        }

        void calculate_error(Matrix Y_hat, Matrix Y) {
            mse = (Y_hat - Y).l2_normalized();
            gradient = (Y_hat - Y) * 2;
        }

        void forward(Matrix input) {
            Matrix activation = input;
            for (int i = 1; i < depth; i++) {
                activation = weights[i].mult(activation) + biases[i];
                activations.push_back(activation);
            }
            Y_hat = activation;
        }

        void backward(Matrix Y) {
            calculate_error(Y_hat, Y);
            gradient = gradient * Y_hat.relu_derivative(0);
            for (int i = depth-1; i > 0; i--) {
                Matrix gradient_weight = gradient.transpose().mult(activations[i-1]);
                Matrix gradient_bias = gradient * -1;

                weights[i] = weights[i] - gradient_weight * learn_rate;
                biases[i] = biases[i] - gradient_bias * learn_rate;

                gradient = gradient.mult(weights[i]).transpose();

            }
        }
};



int main () {
    Network nn;
    nn.initialize_parameters();
    // nn.print_parameters();
    Matrix input(5, 1);
    // vector<vector<double>> temp{{1., 2., 3., 4., 5.}};
    input.mat = {{1, 2, 3, 4, 5}};
    nn.forward(input.transpose());
    nn.Y_hat.print();
    // input.transpose().print();
    // Matrix first(5, 5);
    // Matrix res = first * -1;
    // first.relu_derivative(0.0).print();
    // cout << endl;
    // res.relu_derivative(0.0).print();
    return 0;
}