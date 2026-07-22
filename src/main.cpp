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
    // 数据路径
    std::string train_input  = "data/train_input.csv";
    std::string train_output = "data/train_output.csv";
    std::string test_input   = "data/test_input.csv";
    std::string test_output  = "data/test_output.csv";

    // 训练参数
    int    epochs          = 100;       // ← 100 个 epoch
    int    batch_size      = 256;
    double learning_rate   = 0.005;
    double lr_decay        = 0.998;     // 每 epoch 学习率衰减

    // 网络结构: 输入3 → 128 → 128 → 64 → 1
    std::vector<size_t> hidden_layers = {128, 128, 64};

    // 评估
    int eval_interval = 10;  // 每 N 个 epoch 评估一次
};


// ============================
//   反归一化测试辅助
// ============================
struct TestCase {
    int dividend;
    int divisor;
    int expected_remainder;
};

/// 从归一化数据中还原出一些测试用例
std::vector<TestCase> build_test_cases(
    const Matrix& X_test, size_t max_cases = 500)
{
    std::vector<TestCase> cases;
    size_t n = std::min(X_test.rows, max_cases);

    for (size_t i = 0; i < n; ++i) {
        double d_norm = X_test(i, 0);
        double v_norm = X_test(i, 1);

        int dividend = static_cast<int>(std::round(d_norm * 2500.0));  // MAX_DIVIDEND
        int divisor  = static_cast<int>(std::round(v_norm * 30.0));    // MAX_DIVISOR

        if (divisor < 2) divisor = 2;
        if (dividend < 1) dividend = 1;

        // 手动计算取模（不用 % 运算符）
        int quotient  = dividend / divisor;
        int remainder = dividend - divisor * quotient;

        cases.push_back({dividend, divisor, remainder});
    }
    return cases;
}


// ============================
//       主函数
// ============================
int main() {
    Config cfg;

    std::cout << "============================================\n";
    std::cout << "   神经网络学习取模运算 (Modulo via NN)\n";
    std::cout << "============================================\n\n";

    // ---- 1. 加载数据 ----
    std::cout << "[1] 加载数据..." << std::endl;
    Matrix X_train, Y_train, X_test, Y_test;
    try {
        X_train = utils::load_csv(cfg.train_input);
        Y_train = utils::load_csv(cfg.train_output);
        X_test  = utils::load_csv(cfg.test_input);
        Y_test  = utils::load_csv(cfg.test_output);
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        std::cerr << "请先运行 python3 generate_data.py 生成数据！" << std::endl;
        return 1;
    }

    std::cout << "  训练集: " << X_train.rows << " 样本, "
              << X_train.cols << " 特征" << std::endl;
    std::cout << "  测试集: " << X_test.rows << " 样本, "
              << X_test.cols << " 特征" << std::endl;

    // 构建测试用例
    auto test_cases = build_test_cases(X_test, 1000);
    std::cout << "  测试用例: " << test_cases.size() << " 个\n" << std::endl;

    // ---- 2. 构建网络 ----
    std::cout << "[2] 构建神经网络..." << std::endl;
    Network net;

    size_t input_dim = X_train.cols;  // 3

    // 输入层 → 隐藏层
    net.add_layer(input_dim, cfg.hidden_layers[0], "leaky_relu");
    // 隐藏层之间
    for (size_t i = 0; i + 1 < cfg.hidden_layers.size(); ++i)
        net.add_layer(cfg.hidden_layers[i], cfg.hidden_layers[i + 1], "leaky_relu");
    // 最后一个隐藏层 → 输出层 (sigmoid 输出 [0,1])
    net.add_layer(cfg.hidden_layers.back(), 1, "sigmoid");

    net.init(12345);

    std::cout << "  网络结构: " << input_dim;
    for (auto h : cfg.hidden_layers) std::cout << " → " << h;
    std::cout << " → 1" << std::endl;
    std::cout << "  总层数: " << net.num_layers() << "\n" << std::endl;

    // ---- 3. 训练 ----
    std::cout << "[3] 开始训练..." << std::endl;
    std::cout << "  Epochs: " << cfg.epochs
              << " | Batch: " << cfg.batch_size
              << " | LR: " << cfg.learning_rate << "\n" << std::endl;

    std::mt19937 rng(42);
    double lr = cfg.learning_rate;
    double best_loss = 1e9;

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        // 每个 epoch 内多个 batch
        int num_batches = static_cast<int>(X_train.rows) / cfg.batch_size;
        double epoch_loss = 0.0;

        for (int b = 0; b < num_batches; ++b) {
            auto [X_batch, Y_batch] = utils::get_batch(
                X_train, Y_train, cfg.batch_size, rng);
            double loss = net.train_batch(X_batch, Y_batch, lr);
            epoch_loss += loss;
        }
        epoch_loss /= num_batches;

        // 学习率衰减
        lr *= cfg.lr_decay;

        if (epoch_loss < best_loss)
            best_loss = epoch_loss;

        // 每 N 个 epoch 评估一次
        if ((epoch + 1) % cfg.eval_interval == 0 || epoch == 0) {
            // 在测试集上评估准确率
            int correct = 0;
            int total   = 0;
            double test_loss_sum = 0.0;

            for (auto& tc : test_cases) {
                // 商的最大值 = MAX_DIVIDEND / 2 = 2500 / 2 = 1250
                double q_norm = static_cast<double>(tc.dividend / tc.divisor) / 1250.0;
                double pred_norm = net.predict(
                    static_cast<double>(tc.dividend),
                    static_cast<double>(tc.divisor),
                    q_norm);

                int pred_remainder = utils::denormalize_remainder(
                    pred_norm, tc.divisor);

                if (pred_remainder == tc.expected_remainder)
                    correct++;
                total++;

                // 测试 loss
                double target_norm = static_cast<double>(tc.expected_remainder) / tc.divisor;
                double err = pred_norm - target_norm;
                test_loss_sum += err * err;
            }

            double accuracy = static_cast<double>(correct) / total;
            double test_loss = test_loss_sum / total;

            std::cout << "  Epoch " << std::setw(4) << (epoch + 1)
                      << "/" << cfg.epochs
                      << " | Train Loss: " << std::fixed << std::setprecision(6) << epoch_loss
                      << " | Test Loss: " << std::fixed << std::setprecision(6) << test_loss
                      << " | Accuracy: " << std::fixed << std::setprecision(2)
                      << (accuracy * 100) << "%"
                      << " | LR: " << std::scientific << std::setprecision(4) << lr
                      << std::endl;
        }
    }

    // ---- 4. 最终评估 ----
    std::cout << "\n[4] 最终评估" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    int correct = 0;
    int total   = 0;

    // 随机抽取一些样例展示
    std::vector<size_t> show_indices;
    {
        std::mt19937 show_rng(99);
        std::uniform_int_distribution<size_t> d(0, test_cases.size() - 1);
        std::set<size_t> picked;
        while (picked.size() < 20)
            picked.insert(d(show_rng));
        show_indices.assign(picked.begin(), picked.end());
    }

    std::cout << "\n  部分预测示例:\n";
    std::cout << "  " << std::setw(12) << "表达式"
              << std::setw(10) << "真实值"
              << std::setw(10) << "预测值"
              << std::setw(8) << "结果" << "\n";
    std::cout << "  " << std::string(40, '-') << "\n";

    int show_count = 0;
    for (size_t i = 0; i < test_cases.size(); ++i) {
        auto& tc = test_cases[i];
        // 商的最大值 = MAX_DIVIDEND / 2 = 2500 / 2 = 1250
        double q_norm = static_cast<double>(tc.dividend / tc.divisor) / 1250.0;
        double pred_norm = net.predict(
            static_cast<double>(tc.dividend),
            static_cast<double>(tc.divisor),
            q_norm);

        int pred_remainder = utils::denormalize_remainder(pred_norm, tc.divisor);
        bool is_correct = (pred_remainder == tc.expected_remainder);
        if (is_correct) correct++;
        total++;

        // 显示部分
        if (show_count < 20 &&
            std::find(show_indices.begin(), show_indices.end(), i) != show_indices.end())
        {
            std::string expr = std::to_string(tc.dividend) + " % " +
                               std::to_string(tc.divisor);
            std::cout << "  " << std::setw(12) << expr
                      << std::setw(10) << tc.expected_remainder
                      << std::setw(10) << pred_remainder
                      << std::setw(8) << (is_correct ? "✓" : "✗")
                      << "\n";
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

    // ============================
    //  5. 保存权重文件
    // ============================
    std::string weight_file = "model_weights.txt";
    try {
        net.save_weights(weight_file);
    } catch (const std::exception& e) {
        std::cerr << "保存权重失败: " << e.what() << std::endl;
    }

    // ============================
    //  6. 交互式自定义预测
    // ============================
    std::cout << "\n[6] 进入自定义预测模式 (输入 0 0 退出)\n";
    int user_d, user_v;
    while (true) {
        std::cout << "\n请输入 被除数 和 除数 (例如: 1234 7): ";
        if (!(std::cin >> user_d >> user_v)) break;
        if (user_d == 0 && user_v == 0) break;
        if (user_v == 0) {
            std::cout << "除数不能为0！\n";
            continue;
        }

        // 计算归一化输入
        double q_norm = static_cast<double>(user_d / user_v) / 1250.0;

        double pred_norm = net.predict(user_d, user_v, q_norm);
        int pred_r = utils::denormalize_remainder(pred_norm, user_v);
        int real_r = user_d - user_v * (user_d / user_v); // 手动算真实取模

        std::cout << "  " << user_d << " % " << user_v << " = ";
        std::cout << pred_r;
        if (pred_r == real_r) std::cout << " (✓ 正确, 真实值: " << real_r << ")";
        else std::cout << " (✗ 错误, 真实值: " << real_r << ")";
        std::cout << std::endl;
    }

    return 0;
}