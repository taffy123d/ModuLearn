#include "layer.h"
#include "activations.h"
#include <stdexcept>

DenseLayer::DenseLayer(size_t input_size, size_t output_size,
                       const std::string& activation)
    : input_size_(input_size), output_size_(output_size),
      activation_type_(activation),
      weights_(input_size, output_size, 0.0),
      bias_(1, output_size, 0.0)
{
    // 选择激活函数
    if (activation == "relu") {
        activate_      = activations::relu;
        activate_deriv_ = activations::relu_derivative;
    } else if (activation == "sigmoid") {
        activate_      = activations::sigmoid;
        activate_deriv_ = activations::sigmoid_derivative;
    } else if (activation == "tanh") {
        activate_      = activations::tanh_act;
        activate_deriv_ = activations::tanh_derivative;
    } else if (activation == "leaky_relu") {
        activate_ = [](const Matrix& m) { return activations::leaky_relu(m, 0.01); };
        activate_deriv_ = [](const Matrix& m) { return activations::leaky_relu_derivative(m, 0.01); };
    } else if (activation == "none") {
        activate_ = [](const Matrix& m) { return m; };
        activate_deriv_ = [](const Matrix& m) {
            return Matrix(m.rows, m.cols, 1.0);
        };
    } else {
        throw std::runtime_error("Unknown activation: " + activation);
    }
}

void DenseLayer::init_weights(std::mt19937& rng) {
    weights_.xavier_init(rng);
    // bias 初始化为 0
    bias_ = Matrix(1, output_size_, 0.0);
}

Matrix DenseLayer::forward(const Matrix& input) {
    last_input_ = input;
    last_z_ = Matrix::multiply(input, weights_);
    last_z_.add_bias(bias_);
    last_output_ = activate_(last_z_);
    return last_output_;
}

Matrix DenseLayer::backward(const Matrix& grad_output,
                            double learning_rate,
                            double batch_size)
{
    // dL/dz = dL/da * f'(z)
    Matrix dz = Matrix::hadamard(grad_output, activate_deriv_(last_z_));

    // dL/dW = input^T * dz  (除以 batch_size)
    Matrix dW = Matrix::multiply(last_input_.transpose(), dz) * (1.0 / batch_size);

    // dL/db = sum_rows(dz) / batch_size
    Matrix db = dz.sum_rows() * (1.0 / batch_size);

    // dL/dinput = dz * W^T (传给上一层)
    Matrix grad_input = Matrix::multiply(dz, weights_.transpose());

    // 梯度下降更新
    weights_ = Matrix::subtract(weights_, dW * learning_rate);
    bias_    = Matrix::subtract(bias_, db * learning_rate);

    return grad_input;
}