#pragma once
#include "matrix.h"
#include <string>
#include <random>
#include <functional>

/// 全连接层
class DenseLayer {
public:
    DenseLayer(size_t input_size, size_t output_size,
               const std::string& activation = "relu");

    /// 前向传播
    Matrix forward(const Matrix& input);

    /// 反向传播，返回传递给上一层的梯度
    Matrix backward(const Matrix& grad_output, double learning_rate, double batch_size);

    size_t input_size() const { return input_size_; }
    size_t output_size() const { return output_size_; }

    void init_weights(std::mt19937& rng);

private:
    size_t input_size_;
    size_t output_size_;
    std::string activation_type_;

    Matrix weights_;   // input_size × output_size
    Matrix bias_;      // 1 × output_size

    // 缓存用于反向传播
    Matrix last_input_;
    Matrix last_z_;        // pre-activation
    Matrix last_output_;   // post-activation

    // 激活函数
    std::function<Matrix(const Matrix&)> activate_;
    std::function<Matrix(const Matrix&)> activate_deriv_;
};