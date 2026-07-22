#include "network.h"
#include "utils.h"
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>

void Network::add_layer(size_t in_dim, size_t out_dim, const std::string& activation) {
    Layer layer;
    // W 仍然是 out_dim × in_dim
    layer.W = Matrix(out_dim, in_dim);
    layer.b = Matrix(1, out_dim); // 【修改】bias 改为 1 × out_dim，方便广播
    layer.dW = Matrix(out_dim, in_dim);
    layer.db = Matrix(1, out_dim);
    layer.activation = activation;

    // Adam 动量
    layer.mW = Matrix(out_dim, in_dim);
    layer.vW = Matrix(out_dim, in_dim);
    layer.mb = Matrix(1, out_dim);
    layer.vb = Matrix(1, out_dim);
    layer.t = 0;

    layers.push_back(layer);
}

void Network::init(int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (auto& layer : layers) {
        double scale = std::sqrt(2.0 / layer.W.cols); // He initialization
        for (size_t i = 0; i < layer.W.rows; ++i)
            for (size_t j = 0; j < layer.W.cols; ++j)
                layer.W(i, j) = dist(rng) * scale;

        for (size_t i = 0; i < layer.b.cols; ++i)
            layer.b(0, i) = 0.0;
    }
}

Matrix Network::activate(const Matrix& X, const std::string& type) {
    Matrix result(X.rows, X.cols);
    if (type == "leaky_relu") {
        for (size_t i = 0; i < X.size(); ++i)
            result.data[i] = X.data[i] > 0 ? X.data[i] : 0.01 * X.data[i];
    } else if (type == "sigmoid") {
        for (size_t i = 0; i < X.size(); ++i) {
            double val = std::max(-500.0, std::min(500.0, X.data[i]));
            result.data[i] = 1.0 / (1.0 + std::exp(-val));
        }
    } else if (type == "tanh") {
        for (size_t i = 0; i < X.size(); ++i)
            result.data[i] = std::tanh(X.data[i]);
    } else {
        result = X;
    }
    return result;
}



Matrix Network::activate_deriv(const Matrix& z, const std::string& type) {
    Matrix result(z.rows, z.cols);
    if (type == "leaky_relu") {
        // 传入的是 z（激活前的值），根据 z 的正负判断导数
        for (size_t i = 0; i < z.size(); ++i)
            result.data[i] = z.data[i] > 0 ? 1.0 : 0.01;
    } else if (type == "sigmoid") {
        // 传入的是 z，先算 sigmoid 再求导
        for (size_t i = 0; i < z.size(); ++i) {
            double val = std::max(-500.0, std::min(500.0, z.data[i]));
            double s = 1.0 / (1.0 + std::exp(-val));
            result.data[i] = s * (1.0 - s);
        }
    } else if (type == "tanh") {
        for (size_t i = 0; i < z.size(); ++i) {
            double t = std::tanh(z.data[i]);
            result.data[i] = 1.0 - t * t;
        }
    } else {
        for (size_t i = 0; i < z.size(); ++i)
            result.data[i] = 1.0;
    }
    return result;
}

// ============================
//  【核心修改】前向传播 - 样本按行排
// ============================
// X: N × D (N个样本，D个特征)
// W: out × in
// Z = X * W^T + b  →  N × out
Matrix Network::forward(const Matrix& X) {
    Matrix current = X;
    for (auto& layer : layers) {
        layer.input = current; // 缓存输入: N × in
        
        // Z = X * W^T
        Matrix Wt = layer.W.transpose(); // in × out
        Matrix z = current * Wt;         // (N × in) * (in × out) = N × out
        
        // 加上 bias (广播: b 是 1 × out，加到每一行)
        for (size_t i = 0; i < z.rows; ++i)
            for (size_t j = 0; j < z.cols; ++j)
                z(i, j) += layer.b(0, j);
        
        current = activate(z, layer.activation);
    }
    return current; // N × out_dim
}

// ============================
//  【核心修改】训练 - 反向传播 + Adam
// ============================

double Network::train_batch(const Matrix& X, const Matrix& Y, double lr) {
    size_t N = X.rows; // batch_size
    
    // 前向传播
    Matrix output = forward(X); // N × 1

    // 计算损失 (MSE) 和初始梯度
    double loss = 0.0;
    Matrix delta(N, output.cols); // N × 1
    for (size_t i = 0; i < output.size(); ++i) {
        double err = output.data[i] - Y.data[i];
        loss += err * err;
        delta.data[i] = 2.0 * err / N;
    }
    loss /= N;

    // 反向传播
    for (int l = layers.size() - 1; l >= 0; --l) {
        Layer& layer = layers[l];
        
        // 重新计算 z = input * W^T + b
        Matrix Wt = layer.W.transpose(); // in × out
        Matrix z = layer.input * Wt;     // N × out
        for (size_t i = 0; i < z.rows; ++i)
            for (size_t j = 0; j < z.cols; ++j)
                z(i, j) += layer.b(0, j);
        
        // 计算激活导数 - 传入 z（不是 activated）
        Matrix d_act = activate_deriv(z, layer.activation);
        
        // delta_act = delta ⊙ f'(z)
        Matrix delta_act(N, layer.W.rows); // N × out
        for (size_t i = 0; i < delta.size(); ++i)
            delta_act.data[i] = delta.data[i] * d_act.data[i];

        // dW = delta_act^T * input / N → out × in
        Matrix delta_act_T = delta_act.transpose(); // out × N
        layer.dW = delta_act_T * layer.input;
        for (size_t i = 0; i < layer.dW.size(); ++i)
            layer.dW.data[i] /= N;

        // db = sum(delta_act, axis=0) / N → 1 × out
        layer.db = Matrix(1, layer.W.rows);
        for (size_t j = 0; j < layer.W.rows; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < N; ++i)
                sum += delta_act(i, j);
            layer.db(0, j) = sum / N;
        }

        // 【新增】梯度裁剪 - 防止梯度爆炸
        double grad_norm = 0.0;
        for (size_t i = 0; i < layer.dW.size(); ++i)
            grad_norm += layer.dW.data[i] * layer.dW.data[i];
        for (size_t i = 0; i < layer.db.size(); ++i)
            grad_norm += layer.db.data[i] * layer.db.data[i];
        grad_norm = std::sqrt(grad_norm);
        
        double max_norm = 1.0;
        if (grad_norm > max_norm) {
            double scale = max_norm / grad_norm;
            for (size_t i = 0; i < layer.dW.size(); ++i)
                layer.dW.data[i] *= scale;
            for (size_t i = 0; i < layer.db.size(); ++i)
                layer.db.data[i] *= scale;
        }

        // Adam 优化器更新
        layer.t++;
        double bc1 = 1.0 - std::pow(beta1, layer.t);
        double bc2 = 1.0 - std::pow(beta2, layer.t);

        for (size_t i = 0; i < layer.W.size(); ++i) {
            layer.mW.data[i] = beta1 * layer.mW.data[i] + (1 - beta1) * layer.dW.data[i];
            layer.vW.data[i] = beta2 * layer.vW.data[i] + (1 - beta2) * layer.dW.data[i] * layer.dW.data[i];
            double m_hat = layer.mW.data[i] / bc1;
            double v_hat = layer.vW.data[i] / bc2;
            layer.W.data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        for (size_t i = 0; i < layer.b.size(); ++i) {
            layer.mb.data[i] = beta1 * layer.mb.data[i] + (1 - beta1) * layer.db.data[i];
            layer.vb.data[i] = beta2 * layer.vb.data[i] + (1 - beta2) * layer.db.data[i] * layer.db.data[i];
            double m_hat = layer.mb.data[i] / bc1;
            double v_hat = layer.vb.data[i] / bc2;
            layer.b.data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        // 传播 delta 到上一层
        if (l > 0) {
            delta = delta_act * layer.W;
        }
    }

    return loss;
}

// ============================
//  预测 - 单个样本
// ============================
double Network::predict(const std::vector<double>& features) {
    // 单个样本: 1 × D
    Matrix input(1, features.size());
    for (size_t i = 0; i < features.size(); ++i)
        input(0, i) = features[i];

    Matrix output = forward(input); // 1 × 1
    return output(0, 0);
}

void Network::save_weights(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    ofs << layers.size() << "\n";
    for (const auto& layer : layers) {
        ofs << layer.W.rows << " " << layer.W.cols << "\n";
        for (size_t i = 0; i < layer.W.rows; ++i) {
            for (size_t j = 0; j < layer.W.cols; ++j)
                ofs << layer.W(i, j) << " ";
            ofs << "\n";
        }
        ofs << layer.b.cols << "\n";
        for (size_t i = 0; i < layer.b.cols; ++i)
            ofs << layer.b(0, i) << "\n";
    }
}

void Network::load_weights(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    size_t num_layers;
    ifs >> num_layers;
    layers.resize(num_layers);

    for (auto& layer : layers) {
        size_t rows, cols;
        ifs >> rows >> cols;
        layer.W = Matrix(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                ifs >> layer.W(i, j);

        size_t b_size;
        ifs >> b_size;
        layer.b = Matrix(1, b_size);
        for (size_t i = 0; i < b_size; ++i)
            ifs >> layer.b(0, i);
    }
}