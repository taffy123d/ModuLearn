#pragma once
#include "matrix.h"

namespace activations {
    /// ReLU 及其导数
    Matrix relu(const Matrix& m);
    Matrix relu_derivative(const Matrix& m);

    /// Sigmoid 及其导数 (用 pre-activation 值)
    Matrix sigmoid(const Matrix& m);
    Matrix sigmoid_derivative(const Matrix& pre_sigmoid);

    /// Tanh 及其导数
    Matrix tanh_act(const Matrix& m);
    Matrix tanh_derivative(const Matrix& pre_tanh);

    /// Leaky ReLU
    Matrix leaky_relu(const Matrix& m, double alpha = 0.01);
    Matrix leaky_relu_derivative(const Matrix& m, double alpha = 0.01);
}