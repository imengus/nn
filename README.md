# [WIP] Neural Networks from Scratch in C++

## Library contents:
### layer.h
```
class: Layer --> DenseLayer, ReLULayer, SoftmaxLayer
    func: print_parameters
    func: forward
    func: backward
class: MLP
    func: add_layer
    func: print_network
    func: calculate_error
    func: train
    func: predict
    func: test
```

### matrix.h
```
class: Matrix
    func: T
    overload: | (mat mul)
    overload: * (scalar mul)
    overload: */+- (mat ops)
    overload: == (mat equality)
    overload: = (mat assignment)
    overload: = (2d-vector assign.)
    func: sum
    func: max
    func: min
    func: matexp
    func: print

    func: relu
    func: relu_derivative
    func: softmax
    func: softmax_derivative
    func: l2_normalized
```