#pragma once
#include "matrix.h"
#include <vector>
#include <string>
#include <random>

// ============================
//      全连接层 (DenseLayer)
// ============================
struct DenseLayer {
    size_t in_features;
    size_t out_features;
    std::string activation;

    // 【关键】这里明确声明为 W 和 b
    Matrix W;  
    Matrix b;  

    // 缓存，用于反向传播
    Matrix input_cache;
    Matrix z_cache; // 线性变换结果 (Wx + b)

    DenseLayer() : in_features(0), out_features(0) {}
    DenseLayer(size_t in_f, size_t out_f, const std::string& act)
        : in_features(in_f), out_features(out_f), activation(act),
          W(out_f, in_f), b(1, out_f) {}

    void init_weights(std::mt19937& rng);
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output, double lr, double batch_size);
};

// ============================
//       神经网络类 (Network)
// ============================
class Network {
public:
    Network();

    void add_layer(size_t input_size, size_t output_size,
                   const std::string& activation = "relu");
    void init(unsigned seed = 42);

    Matrix forward(const Matrix& input);
    double train_batch(const Matrix& X, const Matrix& Y, double learning_rate);
    
    // 单样本预测接口
    double predict(double dividend, double divisor, double quotient_norm);

    size_t num_layers() const { return layers_.size(); }

    // 保存和加载权重
    void save_weights(const std::string& filepath) const;
    void load_weights(const std::string& filepath);

private:
    std::vector<DenseLayer> layers_;
    std::mt19937 rng_;
};