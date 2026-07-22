#include "matrix.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <algorithm>

Matrix::Matrix(size_t r, size_t c, double val)
    : rows(r), cols(c), data(r * c, val) {}

double& Matrix::operator()(size_t r, size_t c) {
    return data[r * cols + c];
}

double Matrix::operator()(size_t r, size_t c) const {
    return data[r * cols + c];
}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows)
        throw std::runtime_error("Matrix multiply dimension mismatch");

    Matrix result(a.rows, b.cols, 0.0);
    // 简单 ijk 循环 (可优化为 ikj)
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t k = 0; k < a.cols; ++k) {
            double aik = a(i, k);
            for (size_t j = 0; j < b.cols; ++j) {
                result(i, j) += aik * b(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    return multiply(*this, other);
}

Matrix Matrix::add(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols)
        throw std::runtime_error("Matrix add dimension mismatch");
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < result.size(); ++i)
        result.data[i] = a.data[i] + b.data[i];
    return result;
}

Matrix Matrix::subtract(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols)
        throw std::runtime_error("Matrix subtract dimension mismatch");
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < result.size(); ++i)
        result.data[i] = a.data[i] - b.data[i];
    return result;
}

Matrix Matrix::hadamard(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols)
        throw std::runtime_error("Matrix hadamard dimension mismatch");
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < result.size(); ++i)
        result.data[i] = a.data[i] * b.data[i];
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result(j, i) = (*this)(i, j);
    return result;
}

void Matrix::xavier_init(std::mt19937& rng) {
    double limit = std::sqrt(6.0 / (rows + cols));
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (auto& v : data)
        v = dist(rng);
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < size(); ++i)
        result.data[i] = data[i] * scalar;
    return result;
}

Matrix Matrix::operator+(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < size(); ++i)
        result.data[i] = data[i] + scalar;
    return result;
}

void Matrix::add_bias(const Matrix& bias) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            (*this)(i, j) += bias(0, j);
}

Matrix Matrix::sum_rows() const {
    Matrix result(1, cols, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result(0, j) += (*this)(i, j);
    return result;
}

void Matrix::print(const std::string& name, size_t max_r, size_t max_c) const {
    if (!name.empty())
        std::cout << name << " (" << rows << "×" << cols << "):\n";
    size_t r = std::min(rows, max_r);
    size_t c = std::min(cols, max_c);
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j)
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(10) << (*this)(i, j);
        if (cols > max_c) std::cout << " ...";
        std::cout << "\n";
    }
    if (rows > max_r) std::cout << " ...\n";
    std::cout << std::endl;
}