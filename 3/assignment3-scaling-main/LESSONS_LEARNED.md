# CS336 Assignment 3: Scaling Laws 经验教训

## 概览

本次作业实现了 Chinchilla 论文中的 IsoFLOPs 方法，用于拟合语言模型的缩放定律并预测最优模型配置。

### 完成情况

| 问题 | 描述 | 状态 | 测试 |
|------|------|------|------|
| Problem 1 | 使用已有数据拟合 IsoFLOPs | ✅ 完成 | ✅ 通过 |
| Problem 2 | 主动实验策略 + Mock API | ✅ 完成 | ✅ 通过 |
| 测试覆盖 | 47 个单元/集成测试 | ✅ 100% | ✅ 全部通过 |

---

## 1. 幂律拟合的数值稳定性

### 问题描述
直接在原始空间使用 `scipy.optimize.curve_fit` 拟合 `y = a * x^b` 可能遇到数值稳定性问题。

### 解决方案

**方法1: Log 空间线性拟合（推荐）**
```python
# 转换到 log 空间，问题变为线性拟合
log_x = np.log(compute_budgets)
log_y = np.log(optimal_params)
coeffs = np.polyfit(log_x, log_y, 1)

# 还原参数
b = coeffs[0]  # 斜率就是幂指数
a = np.exp(coeffs[1])  # 截距取 exp 得到系数
```

**方法2: 原始空间非线性拟合（备选）**
```python
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * np.power(x, b)

popt, pcov = curve_fit(power_law, x, y, p0=[initial_a, initial_b])
```

### 经验
- Log 空间拟合更稳定，对数据噪声更鲁棒
- 对于 2-3 个数据点，log 空间线性拟合足够好
- 对于更多数据点，可以先用 log 空间得到初始值，再用非线性拟合精调

---

## 2. API 参数验证与离散值处理

### 问题描述
训练 API 接受严格的参数限制，特别是 `train_flops` 必须是特定离散值之一：
```python
VALID_FLOPS = {1e13, 3e13, 6e13, 1e14, 3e14, 6e14, 1e15, 3e15, 6e15, 
               1e16, 3e16, 6e16, 1e17, 3e17, 6e17, 1e18}
```

### 问题成因
策略生成的连续值 FLOPs（如 `5.77e17`）不在允许集合中，导致 API 调用失败。

### 解决方案
```python
def _round_to_allowed_flops(flops: float) -> int:
    """将 FLOPs 四舍五入到最近的允许值"""
    allowed = sorted(VALID_RANGES['train_flops'])
    return min(allowed, key=lambda x: abs(x - flops))

# 使用示例
compute_budgets = [_round_to_allowed_flops(10**x) for x in log_budgets]
```

### 经验
- **永远验证 API 参数**，不要假设输入合法
- **离散值需要显式处理**，不能依赖近似
- 使用 `min(key=...)` 模式优雅地找到最近值

---

## 3. Mock API 设计

### 设计目标
在没有 Stanford VPN 的情况下，能够：
1. 完整测试代码逻辑
2. 获得可复现的结果
3. 快速迭代开发

### 实现要点

**1. 基于启发式的损失模拟**
```python
def _simulate_loss(self, config):
    N = self._compute_model_params(config)
    D = config.train_flops / (6 * N)
    
    # 基于 Chinchilla scaling law 的启发式
    base_loss = A / (N ** alpha) + B / (D ** beta) + E
    
    # 超参数影响
    lr_factor = 1.0 + 0.5 * abs(np.log10(config.learning_rate) - np.log10(0.001))
    
    # 添加可控噪声
    noise = self.rng.normal(0, 0.02)
    
    return base_loss * lr_factor + noise
```

**2. 结果缓存确保一致性**
```python
def _config_to_key(self, config):
    return f"{config.d_model}_{config.num_layers}_..."

def get_loss(self, config):
    cache_key = self._config_to_key(config)
    if cache_key in self._cache:
        return self._cache[cache_key]
    # 计算并缓存...
```

**3. FLOPs 预算追踪**
```python
# 只对新查询计数 FLOPs
if not any(r['train_flops'] == config.train_flops and ...
           for r in self.query_history):
    self.total_flops_used += config.train_flops
```

### 经验
- Mock API 应该模拟真实 API 的行为模式，而不仅仅是返回随机值
- 使用固定 seed 确保结果可复现，便于调试
- 缓存机制模拟真实 API 的行为（相同配置返回相同结果）

---

## 4. 实验设计策略

### Chinchilla-style IsoFLOPs 策略

**核心思想**：对于每个计算预算 C，测试多个模型大小 N，找到损失最小的最优 N_opt(C)。

**实现**:
```python
def chinchilla_style_strategy(experiment, num_profiles=4, models_per_profile=5):
    # 1. 选择计算预算（在对数空间均匀分布）
    log_budgets = np.linspace(log_min, log_max, num_profiles)
    compute_budgets = [_round_to_allowed_flops(10**x) for x in log_budgets]
    
    for flops in compute_budgets:
        # 2. 对每个预算测试多个模型大小
        target_params = np.logspace(6, 9, models_per_profile)
        
        for target_n in target_params:
            # 3. 找到最接近的 d_model 和 num_layers 组合
            config = find_best_config(target_n, flops)
            experiment.query(config)
```

### 预算分配经验

| 预算 | 建议用途 |
|------|---------|
| 2e18 FLOPs | 总预算 |
| 4 个计算预算 | 每个预算测试 5 个模型 = 20 次查询 |
| 平均 1e17 FLOPs/查询 | 合理分配 |
| 预留 10-20% | 防止超支 |

---

## 5. 可视化与结果分析

### IsoFLOPs 曲线解读
- X轴：模型大小（对数）
- Y轴：训练损失
- 曲线呈 U 型，最小值对应最优模型大小
- 不同计算预算的曲线位置不同（预算越大，整体损失越低）

### 缩放定律图解读
- X轴：计算预算（对数）
- Y轴：最优模型大小（对数）
- 幂律拟合在对数坐标下呈直线
- 斜率 b ≈ 0.4 表示模型大小随计算预算亚线性增长

---

## 6. 测试策略

### 测试分类

| 测试类型 | 文件 | 覆盖内容 |
|---------|------|---------|
| 单元测试 | `test_chinchilla.py` | 数据加载、分组、拟合函数 |
| API 测试 | `test_api.py` | 参数验证、Mock API 行为 |
| 集成测试 | `test_experiment.py` | 完整实验流程、策略执行 |

### 关键测试用例
```python
# 测试幂律拟合准确性
def test_fit_scaling_law_perfect_power_law():
    a_true, b_true = 0.5, 0.6
    C = np.array([1e15, 1e16, 1e17, 1e18])
    N = power_law(C, a_true, b_true)
    
    (a_fit, b_fit), _ = fit_scaling_law(C, N)
    
    assert a_fit == pytest.approx(a_true, rel=0.01)
    assert b_fit == pytest.approx(b_true, rel=0.01)

# 测试预算管理
def test_budget_tracking_accuracy():
    # 验证 FLOPs 计数正确
    # 验证重复查询不计入预算
```

---

## 7. 核心公式速查

| 公式 | 用途 |
|------|------|
| `N = 12 × L × d²` | 计算模型参数量 |
| `D = C / (6N)` | 计算数据集大小 |
| `C = 6ND` | 计算预算关系 |
| `N_opt = a × C^b` | 模型大小缩放定律 |
| `D_opt = c × C^d` | 数据集大小缩放定律 |
| `L(N,D) = A/N^α + B/D^β + E` | 损失预测公式（Chinchilla）|

---

## 总结

### 关键收获
1. **Scaling laws 是大模型训练的重要工具**：可以在实际训练前预测最优配置
2. **实验设计很重要**：有限的预算需要精心分配
3. **Mock API 是开发的好帮手**：无需真实环境即可验证代码
4. **幂律拟合要注意数值稳定性**：log 空间拟合是标准做法

### 可改进之处
1. 可以尝试更多实验策略（如自适应采样）
2. 可以添加模型架构搜索（不仅是大小，还有 depth/width 比例）
3. 可以探索超参数对学习率、batch size 的影响
