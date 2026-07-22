#include "activations.h"
#include <cmath>
#include <algorithm>

namespace activations {

Matrix relu(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (size_t i = 0; i < m.size(); ++i)
        result.data[i] = std::max(0.0, m.data[i]);
    return result;
}

Matrix relu_derivative(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (size_t i = 0; i < m.size(); ++i)
        result.data[i] = (m.data[i] > 0.0) ? 1.0 : 0.0;
    return result;
}

Matrix sigmoid(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (size_t i = 0; i < m.size(); ++i)
        result.data[i] = 1.0 / (1.0 + std::exp(-m.data[i]));
    return result;
}

Matrix sigmoid_derivative(const Matrix& pre_sigmoid) {
    // sigmoid_deriv(z) = sigmoid(z) * (1 - sigmoid(z))
    Matrix s = sigmoid(pre_sigmoid);
    Matrix result(s.rows, s.cols);
    for (size_t i = 0; i < s.size(); ++i)
        result.data[i] = s.data[i] * (1.0 - s.data[i]);
    return result;
}

Matrix tanh_act(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (size_t i = 0; i < m.size(); ++i)
        result.data[i] = std::tanh(m.data[i]);
    return result;
}

Matrix tanh_derivative(const Matrix& pre_tanh) {
    Matrix t = tanh_act(pre_tanh);
    Matrix result(t.rows, t.cols);
    for (size_t i = 0; i < t.size(); ++i)
        result.data[i] = 1.0 - t.data[i] * t.data[i];
    return result;
}

Matrix leaky_relu(const Matrix& m, double alpha) {
    Matrix result(m.rows, m.cols);
    for (size_t i = 0; i < m.size(); ++i)
        result.data[i] = (m.data[i] > 0.0) ? m.data[i] : alpha * m.data[i];
    return result;
}

Matrix leaky_relu_derivative(const Matrix& m, double alpha) {
    Matrix result(m.rows, m.cols);
    for (size_t i = 0; i < m.size(); ++i)
        result.data[i] = (m.data[i] > 0.0) ? 1.0 : alpha;
    return result;
}

} // namespace activations