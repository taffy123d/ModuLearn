#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include <string>
#include <vector>
#include <cmath>

struct Layer {
    Matrix W, b;           // 权重和偏置
    Matrix dW, db;         // 梯度
    Matrix input;          // 前向传播缓存
    std::string activation;

    // Adam 优化器状态
    Matrix mW, vW;         // W 的一阶和二阶动量
    Matrix mb, vb;         // b 的一阶和二阶动量
    int t = 0;             // 时间步
};

class Network {
public:
    std::vector<Layer> layers;

    void add_layer(size_t in_dim, size_t out_dim, const std::string& activation);
    void init(int seed = 42);

    Matrix forward(const Matrix& X);
    double train_batch(const Matrix& X, const Matrix& Y, double lr);
    double predict(const std::vector<double>& features);

    size_t num_layers() const { return layers.size(); }

    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);

private:
    Matrix activate(const Matrix& X, const std::string& type);
    Matrix activate_deriv(const Matrix& activated, const std::string& type);
    
    // Adam 参数
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
};

#endif