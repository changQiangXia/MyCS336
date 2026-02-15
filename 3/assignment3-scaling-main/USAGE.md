# CS336 Assignment 3: 使用说明

## 📁 代码结构

```
cs336_scaling/
├── __init__.py              # 包初始化
├── model.py                 # Transformer 模型 (已有)
├── utils.py                 # 工具函数 (参数量计算等)
├── chinchilla_isoflops.py   # 问题1: 使用已有数据拟合 IsoFLOPs
├── scaling_api.py           # API 封装 + Mock API
└── scaling_experiment.py    # 问题2: 主动实验策略

tests/
├── test_chinchilla.py       # 问题1的测试
├── test_api.py              # API测试
└── test_experiment.py       # 问题2的测试

run_analysis.py              # 主运行脚本
```

---

## 🚀 快速开始

### 1. 配置环境

在 **Anaconda Prompt** 中执行：

```bash
cd d:\pythonProjects\CS336\3\assignment3-scaling-main
uv sync
uv add scipy matplotlib numpy
```

### 2. 运行问题1（使用已有数据）

```bash
uv run python run_analysis.py --problem 1
```

这会：
- 加载 `data/isoflops_curves.json` 数据
- 对每个计算预算找到最优模型大小
- 拟合幂律: N_opt = a × C^b
- 预测 10²³ 和 10²⁴ FLOPs 的最优配置
- 生成可视化图表保存到 `results/` 目录

### 3. 运行问题2（使用 Mock API）

```bash
uv run python run_analysis.py --problem 2 --mock
```

这会：
- 使用模拟 API（不需要 VPN）
- 在 2e18 FLOPs 预算内设计实验
- 拟合缩放定律
- 预测 1e19 FLOPs 的最优配置

### 4. 运行测试

```bash
uv run python run_analysis.py --test
```

或直接使用 pytest：

```bash
uv run python -m pytest tests/ -v
```

---

## 🧪 测试说明

### 测试分类

| 测试文件 | 内容 |
|---------|------|
| `test_chinchilla.py` | 问题1的单元测试 |
| `test_api.py` | API 和 Mock API 测试 |
| `test_experiment.py` | 问题2的实验流程测试 |

### 测试覆盖

- **数据加载和解析**
- **幂律拟合** (log-space 和 non-linear)
- **最优配置查找**
- **API 参数验证**
- **预算管理**
- **Mock API 一致性**
- **完整实验流程**

---

## 📊 代码使用示例

### 问题1: 直接使用已有数据

```python
from cs336_scaling.chinchilla_isoflops import run_chinchilla_analysis

# 运行完整分析
results = run_chinchilla_analysis(
    target_budgets=[1e23, 1e24],
    output_dir="results"
)

# 查看结果
print(f"Model scaling: N = {results['model_scaling']['a']:.3e} * C^{results['model_scaling']['b']:.4f}")
```

### 问题2: 使用 Mock API

```python
from cs336_scaling.scaling_experiment import ScalingExperiment, chinchilla_style_strategy

# 创建实验
experiment = ScalingExperiment(
    budget=2e18,
    target_compute=1e19,
    use_mock=True,  # 使用模拟 API
)

# 运行策略
chinchilla_style_strategy(experiment, num_isoflops_profiles=4, models_per_profile=5)

# 拟合缩放定律
experiment.fit_scaling_law()

# 预测最优配置
prediction = experiment.predict_optimal_config()
print(f"Predicted: d_model={prediction['d_model']}, layers={prediction['num_layers']}")
```

### 自定义实验配置

```python
from cs336_scaling.scaling_api import ExperimentConfig

config = ExperimentConfig(
    d_model=256,           # [64, 1024]
    num_layers=4,          # [2, 24]
    num_heads=4,           # [2, 16]
    batch_size=128,        # {128, 256}
    learning_rate=0.001,   # [1e-4, 1e-3]
    train_flops=1e15,      # 可选值见 VALID_RANGES
)
```

---

## 🔧 核心公式

### 模型参数量
```
N = 12 × num_layers × d_model²
```

### 数据集大小
```
D = C / (6 × N)
```

### 缩放定律
```
N_opt = a × C^b
D_opt = c × C^d
```

---

## 📝 输出文件

运行后会生成：

```
results/
├── model_size_scaling.png      # 模型大小缩放定律图
├── dataset_size_scaling.png    # 数据集大小缩放定律图
└── experiment_results.png      # 实验结果图 (问题2)
```

---

## ❓ 常见问题

### Q: 我没有 Stanford VPN，能做什么？
**A:** 可以完成问题1（使用已有数据），以及用 Mock API 运行问题2的完整流程来学习方法论。

### Q: Mock API 的结果和真实 API 一样吗？
**A:** Mock API 使用基于文献的启发式公式模拟损失，用于代码调试和策略验证。真实结果需要连接 Stanford 的 API。

### Q: 如何验证代码正确性？
**A:** 运行 `uv run python run_analysis.py --test`，所有测试都应该通过。

### Q: 可以修改实验策略吗？
**A:** 可以！在 `scaling_experiment.py` 中实现新的策略函数，参考 `chinchilla_style_strategy` 的写法。

---

## 📚 参考

- Hoffmann et al. 2022 (Chinchilla): Training Compute-Optimal Large Language Models
- Kaplan et al. 2020: Scaling Laws for Neural Language Models
