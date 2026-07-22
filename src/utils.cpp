#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace utils {

Matrix load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    std::string line;
    std::vector<std::vector<double>> rows;

    // 跳过表头
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> row;
        while (std::getline(ss, token, ',')) {
            row.push_back(std::stod(token));
        }
        if (!row.empty())
            rows.push_back(row);
    }

    if (rows.empty())
        throw std::runtime_error("CSV file is empty: " + path);

    size_t r = rows.size();
    size_t c = rows[0].size();
    Matrix mat(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            mat(i, j) = rows[i][j];

    return mat;
}

std::tuple<Matrix, Matrix> get_batch(
    const Matrix& X_all, const Matrix& Y_all,
    size_t batch_size, std::mt19937& rng)
{
    size_t n = X_all.rows;
    if (batch_size > n) batch_size = n;

    std::uniform_int_distribution<size_t> dist(0, n - 1);

    Matrix X_batch(batch_size, X_all.cols);
    Matrix Y_batch(batch_size, Y_all.cols);

    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = dist(rng);
        for (size_t j = 0; j < X_all.cols; ++j)
            X_batch(i, j) = X_all(idx, j);
        for (size_t j = 0; j < Y_all.cols; ++j)
            Y_batch(i, j) = Y_all(idx, j);
    }

    return {X_batch, Y_batch};
}

int denormalize_remainder(double r_norm, int divisor) {
    double r = r_norm * divisor;
    return static_cast<int>(std::round(r));
}

void print_progress(int current, int total, double loss, double accuracy) {
    int bar_width = 40;
    double progress = static_cast<double>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] "
              << static_cast<int>(progress * 100) << "% "
              << "| Loss: " << std::fixed << std::setprecision(6) << loss
              << " | Acc: " << std::fixed << std::setprecision(2)
              << (accuracy * 100) << "%"
              << std::flush;
}

} // namespace utils