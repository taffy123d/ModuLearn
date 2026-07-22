#include "network.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

int main(int argc, char* argv[]) {
    std::string weight_file = "model_weights.txt";

    // 允许通过命令行传入权重路径: ./predict my_weights.txt
    if (argc > 1) weight_file = argv[1];

    std::cout << "============================================\n";
    std::cout << "   神经网络取模运算 - 推理模式 (Predict)\n";
    std::cout << "============================================\n\n";

    // 1. 构建与训练时完全相同的网络结构
    Network net;
    std::vector<size_t> hidden_layers = {128, 128, 64}; // 必须与 main.cpp 中的结构一致
    size_t input_dim = 3;

    net.add_layer(input_dim, hidden_layers[0], "leaky_relu");
    for (size_t i = 0; i + 1 < hidden_layers.size(); ++i)
        net.add_layer(hidden_layers[i], hidden_layers[i + 1], "leaky_relu");
    net.add_layer(hidden_layers.back(), 1, "sigmoid");

    net.init(12345); // 初始化一下占位，后面会被 load_weights 覆盖

    // 2. 加载权重
    try {
        net.load_weights(weight_file);
    } catch (const std::exception& e) {
        std::cerr << "加载权重失败: " << e.what() << std::endl;
        std::cerr << "请确保当前目录下有 " << weight_file << " 文件，"
                  << "或者先运行主程序进行训练。\n";
        return 1;
    }

    // 3. 批量或交互预测
    std::cout << "\n输入 被除数 和 除数 (输入 0 0 退出):\n";
    int d, v;
    while (true) {
        std::cout << "> ";
        if (!(std::cin >> d >> v)) break;
        if (d == 0 && v == 0) break;
        if (v == 0) { std::cout << "除数不能为0！\n"; continue; }

        double q_norm = static_cast<double>(d / v) / 1250.0;
        double pred_norm = net.predict(d, v, q_norm);
        int pred_r = utils::denormalize_remainder(pred_norm, v);

        std::cout << "  [预测] " << d << " % " << v << " = " << pred_r << "\n";
    }

    std::cout << "再见！\n";
    return 0;
}