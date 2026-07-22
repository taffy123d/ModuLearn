#include "network.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <vector>

std::vector<double> build_features(int dividend, int divisor) {
    double d = static_cast<double>(dividend);
    double v = static_cast<double>(divisor);
    const double PI = 3.14159265358979323846;

    std::vector<double> f;

    f.push_back(d / 2500.0);
    f.push_back(v / 30.0);

    double phase = 2.0 * PI * d / v;   
    
    f.push_back(std::sin(phase));
    f.push_back(std::cos(phase));
    f.push_back(std::sin(2.0 * phase));
    f.push_back(std::cos(2.0 * phase));
    f.push_back(std::sin(3.0 * phase));
    f.push_back(std::cos(3.0 * phase));
    f.push_back(std::sin(4.0 * phase));
    f.push_back(std::cos(4.0 * phase));

    f.push_back((d / v) / 1250.0);
    f.push_back(std::fmod(d, v) / v);

    return f;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  取模运算神经网络 - 独立预测工具\n";
    std::cout << "========================================\n\n";

    // ✅ 先搭建和训练时完全一样的网络结构
    Network net;
    std::vector<size_t> hidden_layers = {256, 256, 128, 64};
    size_t input_dim = 12;

    net.add_layer(input_dim, hidden_layers[0], "leaky_relu");
    for (size_t i = 0; i + 1 < hidden_layers.size(); ++i)
        net.add_layer(hidden_layers[i], hidden_layers[i + 1], "leaky_relu");
    net.add_layer(hidden_layers.back(), 1, "sigmoid");

    // ✅ 然后再加载权重
    try {
        net.load_weights("model_weights.txt");
        std::cout << "[✓] 成功加载模型权重\n\n";
    } catch (const std::exception& e) {
        std::cerr << "[✗] 加载权重失败: " << e.what() << "\n";
        return 1;
    }

    std::cout << "输入 被除数 和 除数 (输入 0 0 退出):\n";

    int dividend, divisor;
    while (true) {
        std::cout << "\n> ";
        if (!(std::cin >> dividend >> divisor)) break;
        if (dividend == 0 && divisor == 0) break;
        if (divisor == 0) { std::cout << "除数不能为0！\n"; continue; }

        auto features = build_features(dividend, divisor);
        double pred_norm = net.predict(features);
        int pred_r = utils::denormalize_remainder(pred_norm, divisor);
        int real_r = dividend - divisor * (dividend / divisor);

        std::cout << "  预测: " << dividend << " % " << divisor << " = " << pred_r << "\n";
        std::cout << "  真实: " << dividend << " % " << divisor << " = " << real_r << "\n";
        std::cout << "  状态: " << (pred_r == real_r ? "✓ 正确" : "✗ 错误") << "\n";
    }

    return 0;
}