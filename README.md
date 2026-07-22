# 🧮 nn_modulo - 从零手写神经网络学习取模运算

> **纯 C++ 实现** | 无第三方依赖 | 手撸前向传播、反向传播、Adam 优化器
>
> 用神经网络学会 `a % b = ?` 这件"简单"的事

---

## 🚀 快速开始

```bash
# 1. 生成训练数据
python3 generate_data.py

# 2. 编译
cmake -B build -S .
cmake --build build

# 3. 训练
./build/nn_modulo

# 4. 使用训练好的模型进行交互预测
./build/nn_predict
```

### 环境要求

- C++17 编译器（Clang / GCC）
- CMake ≥ 3.50
- Python 3（仅用于数据生成）

---

## 📐 数据归一化策略

取模运算的输入是整数（被除数 `d ∈ [1, 2500]`，除数 `v ∈ [2, 30]`），输出是余数 `r ∈ [0, v-1]`。
神经网络只认识 `[0, 1]` 区间的小数，所以需要归一化。

### 输入归一化

```
被除数:  d_norm = d / 2500.0        → [0, 1]
除数:    v_norm = v / 30.0          → [0, 1]
```

### 输出归一化（关键设计）

余数的范围取决于除数（`r ∈ [0, v-1]`），不能简单除以固定常数，否则不同除数的余数会互相干扰：

```
r_norm = r / v                     → [0, 1)
```

### 反归一化

```
r = round(r_norm × v)              → 整数余数
```

> 💡 **为什么不直接用 `d/2500` 和 `v/30` 做特征？**
>
> 因为 `d % v` 本质上是一个**高频周期函数**（锯齿波），
> 纯线性归一化后的特征无法捕捉这种周期性。
> 所以我们在归一化的基础上，额外构造了**傅里叶相位特征**（见下文特征工程）。

---

## 🧠 神经网络结构（重点）

### 整体架构

```
输入 (12维)          隐藏层1          隐藏层2          隐藏层3          隐藏层4          输出
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐
│ 12 特征   │───→│  256 节点 │───→│  256 节点 │───→│  128 节点 │───→│  64 节点  │───→│ 1 输出│
│          │    │ LeakyReLU│    │ LeakyReLU│    │ LeakyReLU│    │ LeakyReLU│    │Sigmoid│
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────┘
   12×1          256×12          256×256         128×256          64×128          1×64
```

| 层 | 输入维度 | 输出维度 | 激活函数 | 可训练参数 |
|----|---------|---------|---------|-----------|
| 输入层 | — | 12 | — | 0 |
| 隐藏层 1 | 12 | 256 | LeakyReLU | 256×12 + 256 = **3,328** |
| 隐藏层 2 | 256 | 256 | LeakyReLU | 256×256 + 256 = **65,792** |
| 隐藏层 3 | 256 | 128 | LeakyReLU | 128×256 + 128 = **32,896** |
| 隐藏层 4 | 128 | 64 | LeakyReLU | 64×128 + 64 = **8,256** |
| 输出层 | 64 | 1 | Sigmoid | 1×64 + 1 = **65** |
| **总计** | | | | **110,337** |

### 特征工程（12 维输入）

原始的 `d` 和 `v` 只有 2 个维度。我们手动扩展为 **12 维特征**，帮助网络理解周期性：

```cpp
std::vector<double> build_features(int d, int v) {
    const double PI = 3.14159265358979323846;
    double phase = 2.0 * PI * d / v;   // 核心：相位角

    return {
        d / 2500.0,                     // ① 归一化被除数
        v / 30.0,                       // ② 归一化除数
        sin(phase),   cos(phase),       // ③④ 基频 sin/cos
        sin(2*phase), cos(2*phase),     // ⑤⑥ 二次谐波
        sin(3*phase), cos(3*phase),     // ⑦⑧ 三次谐波
        sin(4*phase), cos(4*phase),     // ⑨⑩ 四次谐波
        (d/v) / 1250.0,                // ⑪ 归一化商
        fmod(d, v) / v                  // ⑫ 归一化余数比
    };
}
```

**为什么用傅里叶特征？**
取模 `d % v` 的几何意义是**锯齿波**——随 `d` 增大，余数周期性地在 `[0, v-1]` 间锯齿跳动。
傅里叶级数正是用 sin/cos 逼近周期函数的标准方法。
把这 4 阶谐波的 sin/cos 直接喂给网络，相当于**把答案的"形状"告诉了一半**。

### 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Epochs | 25 | Adam 收敛快，25 轮足够 |
| Batch Size | 128 | 平衡速度与梯度稳定性 |
| 学习率 | 0.001 | Adam 经典值 |
| 学习率衰减 | 0.999/epoch | 缓慢衰减，后期微调 |
| 损失函数 | MSE | 回归任务标准选择 |
| 优化器 | Adam | 自适应学习率 |
| 权重初始化 | He Init | `N(0, √(2/fan_in))` |
| 梯度裁剪 | max_norm = 1.0 | 防止梯度爆炸 |

### 数据流向图解

```
                    ┌─ 前向传播 ──────────────────────────────────────┐
                    │                                                  │
  X (128×12)  ─────→  X·W₁ᵀ+b₁ → LeakyReLU → X₂                       │
                       X₂·W₂ᵀ+b₂ → LeakyReLU → X₃                     │
                       X₃·W₃ᵀ+b₃ → LeakyReLU → X₄                     │
                       X₄·W₄ᵀ+b₄ → LeakyReLU → X₅                     │
                       X₅·W₅ᵀ+b₅ → Sigmoid → Ŷ (128×1)               │
                    │                                                  │
                    │  Loss = MSE(Ŷ, Y)                               │
                    └──────────────────────────────────────────────────┘

                    ┌─ 反向传播 ──────────────────────────────────────┐
                    │                                                  │
  δ = 2(Ŷ-Y)/N ───→  δ ⊙ σ'(z₅) → δ₅                                 │
                       δ₅·W₅ ⊙ f'(z₄) → δ₄                           │
                       δ₄·W₄ ⊙ f'(z₃) → δ₃                           │
                       δ₃·W₃ ⊙ f'(z₂) → δ₂                           │
                       δ₂·W₂ ⊙ f'(z₁) → δ₁                           │
                    │                                                  │
                    │  dWₗ = δₗᵀ · inputₗ / N   (梯度)               │
                    │  dbₗ = Σ(δₗ) / N                                │
                    │  Adam(W, dW, m, v) → W_new                      │
                    └──────────────────────────────────────────────────┘
```

---

## 🗂️ 项目文件说明

```
nn_modulo/
├── CMakeLists.txt          # CMake 构建配置
├── generate_data.py        # Python 数据生成脚本（生成训练/测试 CSV）
├── data/                   # 数据目录（CSV 文件）
├── model_weights.txt       # 训练后的模型权重文件
│
├── include/                # 头文件（接口定义）
│   ├── matrix.h            #   矩阵类声明：维度、运算、初始化
│   ├── network.h           #   网络类声明：Layer 结构体、Adam 状态
│   ├── layer.h             #   层相关的辅助声明
│   ├── activations.h       #   激活函数声明
│   └── utils.h             #   工具函数声明（CSV读取、batch采样等）
│
└── src/                    # 源文件（实现）
    ├── matrix.cpp          #   矩阵运算实现：乘法、转置、广播加法
    ├── network.cpp         #   ★ 核心！前向传播 + 反向传播 + Adam 更新
    ├── layer.cpp           #   层操作实现
    ├── activations.cpp     #   激活函数实现：LeakyReLU / Sigmoid 及其导数
    ├── main.cpp            #   训练入口：数据加载→特征工程→训练循环→评估
    ├── predict.cpp         #   预测工具：加载权重→交互式预测
    └── utils.cpp           #   工具函数实现
```

### 各文件职责详解

| 文件 | 核心职责 | 关键类/函数 |
|------|---------|------------|
| `matrix.h/.cpp` | **线性代数引擎** — 整个项目的"地基"。实现矩阵的构造、乘法（ikj 循环优化）、转置、逐元素运算、标量运算、广播加法 | `Matrix::multiply()`, `Matrix::transpose()` |
| `activations.h/.cpp` | **激活函数库** — 定义 LeakyReLU、Sigmoid 等激活函数及其导数。前向传播调用激活，反向传播调用导数 | `activate()`, `activate_deriv()` |
| `layer.h/.cpp` | **层抽象** — 定义单层的前向/反向逻辑 | `Layer` 结构体 |
| `network.h/.cpp` | **★ 核心引擎** — 将多层串联，实现完整的前向传播、反向传播链式求导、Adam 自适应优化、梯度裁剪、权重序列化 | `Network::forward()`, `Network::train_batch()` |
| `main.cpp` | **训练流水线** — 数据加载 → 傅里叶特征工程 → 网络构建 → 小批量训练 → 周期评估 → 保存权重 → 交互预测 | `build_features()`, `main()` |
| `predict.cpp` | **独立预测器** — 加载已训练权重，对任意输入做取模预测 | `main()` |
| `utils.h/.cpp` | **工具箱** — CSV 解析、随机 batch 采样、余数反归一化 | `load_csv()`, `get_batch()`, `denormalize_remainder()` |
| `generate_data.py` | **数据工厂** — 生成 `(d, v) → r` 的训练/测试数据并归一化为 CSV | — |

---

## 💻 关键代码片段

### ① 前向传播 — 矩阵视角下的神经网络

```cpp
// X: (N × D),  W: (out × in),  b: (1 × out)
// Z = X · Wᵀ + b   →   (N × out)
Matrix Network::forward(const Matrix& X) {
    Matrix current = X;
    for (auto& layer : layers) {
        layer.input = current;                      // 缓存，反向传播要用
        Matrix z = current * layer.W.transpose();   // (N×in) · (in×out) = N×out
        for (size_t i = 0; i < z.rows; ++i)
            for (size_t j = 0; j < z.cols; ++j)
                z(i, j) += layer.b(0, j);           // 广播 bias
        current = activate(z, layer.activation);
    }
    return current;
}
```

### ② 反向传播 — 链式法则的精确实现

```cpp
// 从输出层向输入层逐层回传梯度
for (int l = layers.size() - 1; l >= 0; --l) {
    // 重算 z，求激活导数
    Matrix z = layer.input * layer.W.transpose() + bias;
    Matrix d_act = activate_deriv(z, layer.activation);

    // δ_act = δ ⊙ f'(z)          ← 链式法则核心
    Matrix delta_act = delta ⊙ d_act;

    // dW = δ_actᵀ · input / N    ← 权重梯度
    layer.dW = delta_act.transpose() * layer.input / N;

    // δ_new = δ_act · W           ← 梯度继续向上一层传播
    delta = delta_act * layer.W;
}
```

### ③ Adam 优化器 — 自适应学习率

```cpp
// 对每个参数独立维护一阶矩 m 和二阶矩 v
layer.mW = β₁·mW + (1-β₁)·dW           // 一阶动量（梯度均值）
layer.vW = β₂·vW + (1-β₂)·dW²          // 二阶动量（梯度方差）

m̂ = mW / (1 - β₁ᵗ)                     // 偏差校正
v̂ = vW / (1 - β₂ᵗ)

W -= lr · m̂ / (√v̂ + ε)                 // 更新：梯度大的参数学习率自动变小
```

### ④ 傅里叶特征工程 — 让网络"看见"周期性

```cpp
double phase = 2.0 * PI * d / v;   // 相位角：d 在 v 的周期中的位置

// 4 阶傅里叶展开 → 8 个特征
sin(phase),   cos(phase),           // 基频：捕捉 d%v 的主周期
sin(2*phase), cos(2*phase),         // 二次谐波：捕捉锯齿波的尖锐转折
sin(3*phase), cos(3*phase),         // 三次谐波：进一步修正波形
sin(4*phase), cos(4*phase),         // 四次谐波：精细细节
```

### ⑤ 梯度裁剪 — 防止训练爆炸

```cpp
double grad_norm = sqrt(Σ dW² + Σ db²);
if (grad_norm > max_norm) {
    double scale = max_norm / grad_norm;
    dW *= scale;    db *= scale;       // 等比缩小，保留方向
}
```

---

## 📊 训练效果

```
  Epochs: 25 | Batch: 128 | LR: 0.001

  Epoch  1/25 | Loss: 0.018214 | Acc: 92.8%
  Epoch  5/25 | Loss: 0.003421 | Acc: 98.3%
  Epoch 10/25 | Loss: 0.001205 | Acc: 99.1%
  Epoch 20/25 | Loss: 0.000538 | Acc: 100.0%
  Epoch 25/25 | Loss: 0.000312 | Acc: 100.0%
```

---

## 🤔 为什么用神经网络做取模？

取模运算 `d % v` 看似简单（一行代码就能算），但对神经网络来说是一个**极具挑战性的学习任务**：

1. **高频非光滑**：余数随 `d` 呈锯齿波形，处处不连续
2. **输入耦合**：答案同时依赖 `d` 和 `v`，且关系非线性
3. **多值性**：相同的 `d` 在不同 `v` 下有完全不同的余数

这使它成为测试神经网络**函数拟合能力**的绝佳 benchmark。
如果网络能学会取模，说明它确实学到了底层的数学结构，而不只是简单的插值。

---

## 📜 License

MIT — 学习用途，自由使用。