#include "matrix.h"
#include <map>
#include <vector>

typedef map<int, Matrix> MatMap;
using MatMap = map<int, Matrix>;
using namespace std;


void init_params(vector<int> layer_dims, MatMap &weights, MatMap &biases) {
    int L = layer_dims.size();

    for (int i = 1; i < L; i++) {
        Matrix temp_w(layer_dims[i-1], layer_dims[i], true);
        temp_w = temp_w * 0.01;
        Matrix temp_b(1, layer_dims[i]);

        weights[i] = temp_w;
        biases[i] = temp_b;

    }
}
int main () {
    MatMap weights;
    MatMap biases;
    init_params({5, 8, 1}, weights, biases);
//     MatMap a = *p;
//     for (int i = 0; i < 2; i++ ) {
//         for (int j = 1; j < 4; j++ ) {
//             cout << "___" << j << endl;
//             (*(p + i))[j].print();
//         }
//    }
    return 0;
}