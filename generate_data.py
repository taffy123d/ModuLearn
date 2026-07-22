#!/usr/bin/env python3
"""
生成训练集和测试集的脚本。
训练集: dividend 在 [1, 2000], divisor 在 [2, 30]
测试集: dividend 在 [2001, 2500], divisor 在 [2, 30]
保证两个集合完全不重叠。
输出: (dividend, divisor, dividend/divisor商(归一化)) -> remainder/divisor(归一化结果)
"""

import os
import random
import csv

random.seed(42)

# ---- 参数 ----
TRAIN_DIVIDEND_RANGE = (1, 2000)
TEST_DIVIDEND_RANGE  = (2001, 2500)
DIVISOR_RANGE        = (2, 30)

TRAIN_SAMPLES = 10000   # 总组合 2000×29=58,000 >> 10,000 ✓
TEST_SAMPLES  = 2000    # 总组合 500×29=14,500  >> 2,000  ✓

MAX_DIVIDEND = 2500     # 全局最大被除数（用于归一化）
MAX_DIVISOR  = 30       # 全局最大除数（用于归一化）

# ---- 输出目录 ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_dataset(dividend_range, divisor_range, num_samples):
    """生成数据集，返回 list of (dividend, divisor, remainder)"""
    # 安全检查：组合数必须大于所需样本数
    total_combos = (dividend_range[1] - dividend_range[0] + 1) * \
                   (divisor_range[1] - divisor_range[0] + 1)
    if num_samples > total_combos:
        raise ValueError(
            f"需要 {num_samples} 个样本，但只有 {total_combos} 种组合！"
            f"请扩大数据范围或减少样本数。"
        )

    samples = []
    seen = set()
    while len(samples) < num_samples:
        dividend = random.randint(*dividend_range)
        divisor  = random.randint(*divisor_range)
        key = (dividend, divisor)
        if key in seen:
            continue
        seen.add(key)
        remainder = dividend - divisor * (dividend // divisor)  # 手动算取模
        samples.append((dividend, divisor, remainder))
    return samples


def write_csv(samples, input_path, output_path):
    """
    写入 CSV。
    输入: [dividend/MAX_DIVIDEND, divisor/MAX_DIVISOR, quotient_norm]
    输出: [remainder/divisor]  (归一化到 [0, 1))
    """
    max_quotient = MAX_DIVIDEND / 2.0  # 商的最大值约为 MAX_DIVIDEND / 最小除数(2) = 1250

    with open(input_path, 'w', newline='') as fi, \
         open(output_path, 'w', newline='') as fo:

        wi = csv.writer(fi)
        wo = csv.writer(fo)

        wi.writerow(["dividend_norm", "divisor_norm", "quotient_norm"])
        wo.writerow(["remainder_norm"])

        for dividend, divisor, remainder in samples:
            d_norm = dividend / MAX_DIVIDEND
            v_norm = divisor / MAX_DIVISOR
            q_norm = (dividend // divisor) / max_quotient
            r_norm = remainder / divisor  # 归一化到 [0, 1)

            wi.writerow([f"{d_norm:.6f}", f"{v_norm:.6f}", f"{q_norm:.6f}"])
            wo.writerow([f"{r_norm:.6f}"])


# ---- 生成 ----
print("生成训练集...")
train_data = generate_dataset(TRAIN_DIVIDEND_RANGE, DIVISOR_RANGE, TRAIN_SAMPLES)
print(f"  训练集大小: {len(train_data)}")

print("生成测试集...")
test_data = generate_dataset(TEST_DIVIDEND_RANGE, DIVISOR_RANGE, TEST_SAMPLES)
print(f"  测试集大小: {len(test_data)}")

# ---- 验证不重叠 ----
train_keys = set((d, v) for d, v, _ in train_data)
test_keys  = set((d, v) for d, v, _ in test_data)
overlap = train_keys & test_keys
assert len(overlap) == 0, f"训练集和测试集有 {len(overlap)} 个重叠样本!"
print("✓ 训练集与测试集无重叠")

# ---- 写入文件 ----
write_csv(train_data,
          os.path.join(DATA_DIR, "train_input.csv"),
          os.path.join(DATA_DIR, "train_output.csv"))

write_csv(test_data,
          os.path.join(DATA_DIR, "test_input.csv"),
          os.path.join(DATA_DIR, "test_output.csv"))

print(f"✓ 数据已写入 {DATA_DIR}/")
print("  - train_input.csv, train_output.csv")
print("  - test_input.csv,  test_output.csv")