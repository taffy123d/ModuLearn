#include "network.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>

// ============================
//    DenseLayer 实现
// ============================
void DenseLayer::init_weights(std::mt19937& rng) {
    // He 初始化
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / in_features));
    for (size_t i = 0; i < W.size(); ++i) W.data[i] = dist(rng);
    for (size_t i = 0; i < b.size(); ++i) b.data[i] = 0.0;
}

Matrix DenseLayer::forward(const Matrix& input) {
    input_cache = input;
    // z = X * W^T + b
    Matrix W_T = W.transpose();
    Matrix z = input * W_T;
    
    // 加上偏置 b
    for (size_t i = 0; i < z.rows; ++i) {
        for (size_t j = 0; j < z.cols; ++j) {
            z(i, j) += b.data[j];
        }
    }
    z_cache = z;

    // 激活函数
    Matrix out(z.rows, z.cols);
    if (activation == "relu") {
        for (size_t i = 0; i < z.size(); ++i)
            out.data[i] = std::max(0.0, z.data[i]);
    } else if (activation == "leaky_relu") {
        for (size_t i = 0; i < z.size(); ++i)
            out.data[i] = z.data[i] > 0 ? z.data[i] : 0.01 * z.data[i];
    } else if (activation == "sigmoid") {
        for (size_t i = 0; i < z.size(); ++i)
            out.data[i] = 1.0 / (1.0 + std::exp(-z.data[i]));
    } else {
        out = z; // linear
    }
    return out;
}

Matrix DenseLayer::backward(const Matrix& grad_output, double lr, double batch_size) {
    // 1. 激活函数导数
    Matrix delta(grad_output.rows, grad_output.cols);
    if (activation == "relu") {
        for (size_t i = 0; i < delta.size(); ++i)
            delta.data[i] = z_cache.data[i] > 0 ? grad_output.data[i] : 0.0;
    } else if (activation == "leaky_relu") {
        for (size_t i = 0; i < delta.size(); ++i)
            delta.data[i] = z_cache.data[i] > 0 ? grad_output.data[i] : 0.01 * grad_output.data[i];
    } else if (activation == "sigmoid") {
        for (size_t i = 0; i < delta.size(); ++i) {
            double s = 1.0 / (1.0 + std::exp(-z_cache.data[i]));
            delta.data[i] = grad_output.data[i] * s * (1.0 - s);
        }
    } else {
        delta = grad_output;
    }

    // 2. 计算梯度
    // dW = delta^T * input
    Matrix delta_T = delta.transpose();
    Matrix dW = delta_T * input_cache;
    // db = sum(delta, axis=0)
    Matrix db(1, out_features);
    for (size_t i = 0; i < delta.rows; ++i)
        for (size_t j = 0; j < delta.cols; ++j)
            db.data[j] += delta(i, j);

    // 3. 传递给上一层的梯度: grad_input = delta * W
    Matrix grad_input = delta * W;

    // 4. 更新权重 (SGD)
    double scale = lr / batch_size;
    for (size_t i = 0; i < W.size(); ++i) W.data[i] -= scale * dW.data[i];
    for (size_t i = 0; i < b.size(); ++i) b.data[i] -= scale * db.data[i];

    return grad_input;
}

// ============================
//       Network 实现
// ============================
Network::Network() : rng_(42) {}

void Network::add_layer(size_t input_size, size_t output_size,
                        const std::string& activation) {
    layers_.emplace_back(input_size, output_size, activation);
}

void Network::init(unsigned seed) {
    rng_.seed(seed);
    for (auto& layer : layers_)
        layer.init_weights(rng_);
}

Matrix Network::forward(const Matrix& input) {
    Matrix current = input;
    for (auto& layer : layers_)
        current = layer.forward(current);
    return current;
}

double Network::train_batch(const Matrix& X, const Matrix& Y,
                            double learning_rate)
{
    Matrix output = forward(X);
    double batch_size = static_cast<double>(X.rows);
    Matrix diff = Matrix::subtract(output, Y);

    double loss = 0.0;
    for (size_t i = 0; i < diff.size(); ++i)
        loss += diff.data[i] * diff.data[i];
    loss /= batch_size;

    Matrix grad = diff * (2.0 / batch_size);

    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i)
        grad = layers_[i].backward(grad, learning_rate, batch_size);

    return loss;
}

double Network::predict(double dividend, double divisor, double quotient_norm) {
    Matrix input(1, 3);
    input(0, 0) = dividend / 2500.0;
    input(0, 1) = divisor / 30.0;
    input(0, 2) = quotient_norm;
    Matrix output = forward(input);
    return output(0, 0);
}

// ============================
//      保存与加载权重
// ============================
void Network::save_weights(const std::string& filepath) const {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) throw std::runtime_error("无法打开文件保存权重: " + filepath);

    ofs << layers_.size() << "\n";

    for (const auto& layer : layers_) {
        // 写入 W
        ofs << layer.W.rows << " " << layer.W.cols << "\n";
        for (size_t i = 0; i < layer.W.size(); ++i) {
            ofs << std::setprecision(10) << layer.W.data[i]
                << (i + 1 == layer.W.size() ? "\n" : " ");
        }
        // 写入 b
        ofs << layer.b.rows << " " << layer.b.cols << "\n";
        for (size_t i = 0; i < layer.b.size(); ++i) {
            ofs << std::setprecision(10) << layer.b.data[i]
                << (i + 1 == layer.b.size() ? "\n" : " ");
        }
    }
    std::cout << "✓ 权重已保存到: " << filepath << std::endl;
}

void Network::load_weights(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) throw std::runtime_error("无法打开权重文件: " + filepath);

    size_t num_layers;
    ifs >> num_layers;

    if (num_layers != layers_.size()) {
        throw std::runtime_error("权重文件层数与当前网络结构不匹配！");
    }

    for (auto& layer : layers_) {
        // 读取 W
        size_t w_rows, w_cols;
        ifs >> w_rows >> w_cols;
        if (w_rows != layer.W.rows || w_cols != layer.W.cols)
            throw std::runtime_error("权重矩阵 W 维度不匹配！");
        for (size_t i = 0; i < layer.W.size(); ++i) ifs >> layer.W.data[i];

        // 读取 b
        size_t b_rows, b_cols;
        ifs >> b_rows >> b_cols;
        if (b_rows != layer.b.rows || b_cols != layer.b.cols)
            throw std::runtime_error("偏置矩阵 b 维度不匹配！");
        for (size_t i = 0; i < layer.b.size(); ++i) ifs >> layer.b.data[i];
    }
    std::cout << "✓ 成功加载权重: " << filepath << std::endl;
}