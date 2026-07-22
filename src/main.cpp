#include "network.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <set>

// ============================
//       配置参数
// ============================
struct Config {
    std::string train_input  = "data/train_input.csv";
    std::string train_output = "data/train_output.csv";
    std::string test_input   = "data/test_input.csv";
    std::string test_output  = "data/test_output.csv";

    int    epochs          = 25;
    int    batch_size      = 128;      // 更大 batch 减少震荡
    double learning_rate   = 0.001;    // Adam 用 0.001 更安全
    double lr_decay        = 0.999;

    std::vector<size_t> hidden_layers = {256, 256, 128, 64}; // 稍微缩小

    int eval_interval = 1;
};

// ============================
//    【修正版】特征工程函数
// ============================
std::vector<double> build_features(int dividend, int divisor) {
    double d = static_cast<double>(dividend);
    double v = static_cast<double>(divisor);
    const double PI = 3.14159265358979323846;

    std::vector<double> f;

    // 1. 基础归一化特征 (2 个)
    f.push_back(d / 2500.0);
    f.push_back(v / 30.0);

    // 2. 核心相位特征 (8 个) - 傅里叶级数
    double phase = 2.0 * PI * d / v;   
    
    f.push_back(std::sin(phase));
    f.push_back(std::cos(phase));
    f.push_back(std::sin(2.0 * phase));
    f.push_back(std::cos(2.0 * phase));
    f.push_back(std::sin(3.0 * phase));
    f.push_back(std::cos(3.0 * phase));
    f.push_back(std::sin(4.0 * phase));
    f.push_back(std::cos(4.0 * phase));

    // 3. 辅助特征 (2 个)
    f.push_back((d / v) / 1250.0);
    f.push_back(std::fmod(d, v) / v);

    return f;  // 共 12 个特征
}

// ============================
//       测试用例
// ============================
struct TestCase {
    int dividend;
    int divisor;
    int expected_remainder;
    std::vector<double> features;
};

std::vector<TestCase> build_test_cases(const Matrix& X_test, size_t max_cases = 1000) {
    std::vector<TestCase> cases;
    size_t n = std::min(X_test.rows, max_cases);

    for (size_t i = 0; i < n; ++i) {
        double d_norm = X_test(i, 0);
        double v_norm = X_test(i, 1);

        int dividend = static_cast<int>(std::round(d_norm * 2500.0));
        int divisor  = static_cast<int>(std::round(v_norm * 30.0));

        if (divisor < 2) divisor = 2;
        if (dividend < 1) dividend = 1;

        int quotient  = dividend / divisor;
        int remainder = dividend - divisor * quotient;

        auto features = build_features(dividend, divisor);
        cases.push_back({dividend, divisor, remainder, features});
    }
    return cases;
}

// ============================
//       主函数
// ============================
int main() {
    Config cfg;

    std::cout << "============================================\n";
    std::cout << "   神经网络学习取模运算 (Modulo via NN) v2\n";
    std::cout << "============================================\n\n";

    // ---- 1. 加载数据 ----
    std::cout << "[1] 加载数据..." << std::endl;
    Matrix X_train_raw, Y_train, X_test_raw, Y_test;
    try {
        X_train_raw = utils::load_csv(cfg.train_input);
        Y_train     = utils::load_csv(cfg.train_output);
        X_test_raw  = utils::load_csv(cfg.test_input);
        Y_test      = utils::load_csv(cfg.test_output);
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    // ---- 2. 特征工程 ----
    std::cout << "[2] 特征工程 (3维 → 12维)..." << std::endl;

    size_t num_features = 12;
    Matrix X_train(X_train_raw.rows, num_features);
    for (size_t i = 0; i < X_train_raw.rows; ++i) {
        int dividend = static_cast<int>(std::round(X_train_raw(i, 0) * 2500.0));
        int divisor  = static_cast<int>(std::round(X_train_raw(i, 1) * 30.0));
        if (divisor < 2) divisor = 2;
        if (dividend < 1) dividend = 1;
        auto f = build_features(dividend, divisor);
        for (size_t j = 0; j < num_features; ++j)
            X_train(i, j) = f[j];
    }

    Matrix X_test(X_test_raw.rows, num_features);
    for (size_t i = 0; i < X_test_raw.rows; ++i) {
        int dividend = static_cast<int>(std::round(X_test_raw(i, 0) * 2500.0));
        int divisor  = static_cast<int>(std::round(X_test_raw(i, 1) * 30.0));
        if (divisor < 2) divisor = 2;
        if (dividend < 1) dividend = 1;
        auto f = build_features(dividend, divisor);
        for (size_t j = 0; j < num_features; ++j)
            X_test(i, j) = f[j];
    }

    std::cout << "  训练集: " << X_train.rows << " 样本, "
              << X_train.cols << " 特征" << std::endl;
    std::cout << "  测试集: " << X_test.rows << " 样本, "
              << X_test.cols << " 特征" << std::endl;

    auto test_cases = build_test_cases(X_test_raw, 1000);
    std::cout << "  测试用例: " << test_cases.size() << " 个\n" << std::endl;

    // ---- 3. 构建网络 ----
    std::cout << "[3] 构建神经网络..." << std::endl;
    Network net;

    size_t input_dim = num_features;

    net.add_layer(input_dim, cfg.hidden_layers[0], "leaky_relu");
    for (size_t i = 0; i + 1 < cfg.hidden_layers.size(); ++i)
        net.add_layer(cfg.hidden_layers[i], cfg.hidden_layers[i + 1], "leaky_relu");
    net.add_layer(cfg.hidden_layers.back(), 1, "sigmoid");

    net.init(12345);

    std::cout << "  网络结构: " << input_dim;
    for (auto h : cfg.hidden_layers) std::cout << " → " << h;
    std::cout << " → 1" << std::endl;
    std::cout << "  总层数: " << net.num_layers() << "\n" << std::endl;

    // ---- 4. 训练 ----
    std::cout << "[4] 开始训练 (Adam Optimizer)..." << std::endl;
    std::cout << "  Epochs: " << cfg.epochs
              << " | Batch: " << cfg.batch_size
              << " | LR: " << cfg.learning_rate << "\n" << std::endl;

    std::mt19937 rng(42);
    double lr = cfg.learning_rate;
    double best_loss = 1e9;

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        int num_batches = static_cast<int>(X_train.rows) / cfg.batch_size;
        double epoch_loss = 0.0;

        for (int b = 0; b < num_batches; ++b) {
            auto [X_batch, Y_batch] = utils::get_batch(
                X_train, Y_train, cfg.batch_size, rng);
            double loss = net.train_batch(X_batch, Y_batch, lr);
            epoch_loss += loss;
        }
        epoch_loss /= num_batches;

        lr *= cfg.lr_decay;
        if (epoch_loss < best_loss) best_loss = epoch_loss;

        if ((epoch + 1) % cfg.eval_interval == 0 || epoch == 0) {
            int correct = 0, total = 0;
            double test_loss_sum = 0.0;

            for (auto& tc : test_cases) {
                double pred_norm = net.predict(tc.features);
                int pred_r = utils::denormalize_remainder(pred_norm, tc.divisor);
                if (pred_r == tc.expected_remainder) correct++;
                total++;

                double target_norm = static_cast<double>(tc.expected_remainder) / tc.divisor;
                double err = pred_norm - target_norm;
                test_loss_sum += err * err;
            }

            double accuracy = static_cast<double>(correct) / total;
            double test_loss = test_loss_sum / total;

            std::cout << "  Epoch " << std::setw(4) << (epoch + 1)
                      << "/" << cfg.epochs
                      << " | Loss: " << std::fixed << std::setprecision(6) << epoch_loss
                      << " | Test: " << std::fixed << std::setprecision(6) << test_loss
                      << " | Acc: " << std::fixed << std::setprecision(1)
                      << (accuracy * 100) << "%"
                      << " | LR: " << std::scientific << std::setprecision(3) << lr
                      << std::endl;
        }
    }

    // ---- 5. 最终评估 ----
    std::cout << "\n[5] 最终评估" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    int correct = 0, total = 0;

    std::vector<size_t> show_indices;
    {
        std::mt19937 show_rng(99);
        std::uniform_int_distribution<size_t> d(0, test_cases.size() - 1);
        std::set<size_t> picked;
        while (picked.size() < 20) picked.insert(d(show_rng));
        show_indices.assign(picked.begin(), picked.end());
    }

    std::cout << "\n  部分预测示例:\n";
    std::cout << "  " << std::setw(14) << "表达式"
              << std::setw(8) << "真实值"
              << std::setw(8) << "预测值"
              << std::setw(6) << "结果" << "\n";
    std::cout << "  " << std::string(36, '-') << "\n";

    int show_count = 0;
    for (size_t i = 0; i < test_cases.size(); ++i) {
        auto& tc = test_cases[i];
        double pred_norm = net.predict(tc.features);
        int pred_r = utils::denormalize_remainder(pred_norm, tc.divisor);
        bool ok = (pred_r == tc.expected_remainder);
        if (ok) correct++;
        total++;

        if (show_count < 20 &&
            std::find(show_indices.begin(), show_indices.end(), i) != show_indices.end())
        {
            std::string expr = std::to_string(tc.dividend) + " % " +
                               std::to_string(tc.divisor);
            std::cout << "  " << std::setw(14) << expr
                      << std::setw(8) << tc.expected_remainder
                      << std::setw(8) << pred_r
                      << std::setw(6) << (ok ? "✓" : "✗") << "\n";
            show_count++;
        }
    }

    double final_accuracy = static_cast<double>(correct) / total;
    std::cout << "\n----------------------------------------------" << std::endl;
    std::cout << "  总测试数: " << total << std::endl;
    std::cout << "  正确数:   " << correct << std::endl;
    std::cout << "  准确率:   " << std::fixed << std::setprecision(2)
              << (final_accuracy * 100) << "%" << std::endl;
    std::cout << "============================================\n";

    // ---- 6. 保存权重 ----
    std::string weight_file = "model_weights.txt";
    try { net.save_weights(weight_file); }
    catch (const std::exception& e) { std::cerr << "保存失败: " << e.what() << std::endl; }

    // ---- 7. 交互预测 ----
    std::cout << "\n[7] 交互式预测 (输入 0 0 退出)\n";
    int user_d, user_v;
    while (true) {
        std::cout << "\n> ";
        if (!(std::cin >> user_d >> user_v)) break;
        if (user_d == 0 && user_v == 0) break;
        if (user_v == 0) { std::cout << "除数不能为0！\n"; continue; }

        auto features = build_features(user_d, user_v);
        double pred_norm = net.predict(features);
        int pred_r = utils::denormalize_remainder(pred_norm, user_v);
        int real_r = user_d - user_v * (user_d / user_v);

        std::cout << "  " << user_d << " % " << user_v << " = " << pred_r;
        if (pred_r == real_r) std::cout << " (✓ 真实值: " << real_r << ")";
        else std::cout << " (✗ 真实值: " << real_r << ")";
        std::cout << std::endl;
    }

    return 0;
}