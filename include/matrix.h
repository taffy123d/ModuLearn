#pragma once
#include <vector>
#include <cstddef>
#include <string>
#include <random>

/// 简单的矩阵类，支持基本运算
class Matrix {
public:
    size_t rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(size_t r, size_t c, double val = 0.0);

    double& operator()(size_t r, size_t c);
    double  operator()(size_t r, size_t c) const;

    // 矩阵乘法
    static Matrix multiply(const Matrix& a, const Matrix& b);
    Matrix operator*(const Matrix& other) const;       // a * b 运算符重载

    // 逐元素运算
    static Matrix add(const Matrix& a, const Matrix& b);
    static Matrix subtract(const Matrix& a, const Matrix& b);
    static Matrix hadamard(const Matrix& a, const Matrix& b); // 逐元素乘

    // 转置
    Matrix transpose() const;

    // Xavier 初始化
    void xavier_init(std::mt19937& rng);

    // 标量运算
    Matrix operator*(double scalar) const;
    Matrix operator+(double scalar) const;

    // 广播: 将 bias (1×cols) 加到每一行
    void add_bias(const Matrix& bias);

    // 对每行求和 → (1×cols)
    Matrix sum_rows() const;

    // 打印
    void print(const std::string& name = "", size_t max_r = 5, size_t max_c = 5) const;

    size_t size() const { return rows * cols; }
};