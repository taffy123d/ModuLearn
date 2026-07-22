#pragma once
#include "matrix.h"
#include <string>
#include <vector>
#include <tuple>

namespace utils {

    /// 从 CSV 加载数据
    Matrix load_csv(const std::string& path);

    /// 从 CSV 中取随机 batch
    /// 返回 (X_batch, Y_batch)
    std::tuple<Matrix, Matrix> get_batch(
        const Matrix& X_all, const Matrix& Y_all,
        size_t batch_size, std::mt19937& rng);

    /// 将归一化的 remainder_norm 还原为整数 remainder
    int denormalize_remainder(double r_norm, int divisor);

    /// 简单进度条
    void print_progress(int current, int total, double loss, double accuracy);
}